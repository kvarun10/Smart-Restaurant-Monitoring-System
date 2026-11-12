"""End-to-end logic on a synthetic fixed-camera scene (no detector, no GPU)."""
import numpy as np
import pytest

from tablewatch.cleanliness import ReferenceScorer, changed_cells, prepare
from tablewatch.evaluate import dirty_episodes, predicted_intervals, time_confusion
from tablewatch.monitor import MonitorConfig, TableMonitor, overlap_share
from tablewatch.tracker import CLEAN, DIRTY, OCCUPIED, TrackerConfig
from tablewatch.zones import Zone, cluster_boxes, load_zones, make_zones, save_zones

T1, T2 = (100, 100, 300, 250), (400, 100, 600, 250)
ZONES = [Zone("T1", T1), Zone("T2", T2)]
FPS = 10


def scene(rng, brightness=1.0, dishes_on=(), preset=True):
    """Grey floor, two white tables. T2 always has its standard place setting (a dark tray)."""
    img = np.full((360, 720), 90.0)
    for box in (T1, T2):
        img[box[1]:box[3], box[0]:box[2]] = 215
    if preset:
        img[150:200, 470:530] = 60                      # pre-set tray on T2: part of "clean"
    for box in dishes_on:                                # leftovers: plates and a cup
        x, y = box[0], box[1]
        yy, xx = np.ogrid[:360, :720]
        for cx, cy, r, v in [(x + 50, y + 50, 22, 120), (x + 130, y + 95, 26, 140), (x + 160, y + 40, 12, 40)]:
            img[(xx - cx) ** 2 + (yy - cy) ** 2 < r * r] = v
    img = img * brightness + rng.normal(0, 2.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def person_at(box):   # someone sitting on the near side of the table
    return (box[0] + 40, box[1] + 60, box[0] + 130, box[3] + 60)


def test_overlap_share():
    assert overlap_share(T1, person_at(T1)) > 0.5
    assert overlap_share(T1, person_at(T2)) == 0.0


def test_change_score_reacts_to_dishes_not_to_lighting():
    rng = np.random.default_rng(0)
    crop = lambda img, b: img[b[1]:b[3], b[0]:b[2]]
    scorer = ReferenceScorer(warmup_frames=3)
    for _ in range(3):
        scorer.warm_up("T1", crop(scene(rng), T1))
        scorer.warm_up("T2", crop(scene(rng), T2))
    assert scorer.ready("T1")

    same = scorer.score("T1", crop(scene(rng), T1))
    darker = scorer.score("T1", crop(scene(rng, brightness=0.7), T1))
    brighter = scorer.score("T2", crop(scene(rng, brightness=1.15), T2))
    dirty = scorer.score("T1", crop(scene(rng, dishes_on=[T1]), T1))
    dirty_dark = scorer.score("T1", crop(scene(rng, brightness=0.7, dishes_on=[T1]), T1))
    preset_clean = scorer.score("T2", crop(scene(rng), T2))
    preset_removed = scorer.score("T2", crop(scene(rng, preset=False), T2))

    assert max(same, darker, brighter, preset_clean) < 0.05     # lighting / a table's own setting: clean
    assert min(dirty, dirty_dark) > 0.15                         # leftovers: clearly different
    assert preset_removed > 0.05                                 # the setting itself going missing is a change
    assert changed_cells(prepare(crop(scene(rng), T1)), prepare(crop(scene(rng), T1))).shape == (8, 8)


def run_story(dirty_alert_s=20.0):
    """T1: clean 10 s -> guests 30 s -> they leave dishes 40 s -> waiter clears -> clean 20 s.
       T2: pre-set and untouched throughout; lights dim half way."""
    rng = np.random.default_rng(1)
    mon = TableMonitor(ZONES, MonitorConfig(tracker=TrackerConfig(dirty_alert_s=dirty_alert_s)))
    t = 0.0
    while t < 100:
        guests, leftovers = 10 <= t < 40, 40 <= t < 80
        people = [person_at(T1)] if guests else []
        if 60 <= t < 61:
            people.append((280, 60, 340, 300))          # waiter walks past, clipping T1's edge
        frame = scene(rng, brightness=0.8 if t >= 50 else 1.0,
                      dishes_on=[T1] if (guests or leftovers) else [])
        mon.update(t, frame, people)
        t = round(t + 1 / FPS, 3)
    return mon, t


def test_full_story_states_alert_and_events():
    mon, t_end = run_story()
    events = mon.tracker.events
    t1 = [(e["type"], e.get("to")) for e in events if e["table"] == "T1"]
    assert t1 == [("state_change", CLEAN), ("state_change", OCCUPIED), ("state_change", DIRTY),
                  ("dirty_alert", None), ("state_change", CLEAN)]
    t2 = [(e["type"], e.get("to")) for e in events if e["table"] == "T2"]
    assert t2 == [("state_change", CLEAN)]              # pre-set table + dimmed lights: never flagged

    went_dirty = next(e["t"] for e in events if e.get("to") == DIRTY)
    alert = next(e for e in events if e["type"] == "dirty_alert")
    assert 40 < went_dirty < 44                          # noticed within a few seconds of guests leaving
    assert alert["t"] == pytest.approx(went_dirty + 20, abs=0.3)
    assert mon.snapshot(t_end)[0]["state"] == CLEAN


def test_evaluation_against_ground_truth():
    mon, t_end = run_story()
    gt = {"T1": [(0, 10, CLEAN), (10, 40, OCCUPIED), (40, 80, DIRTY), (80, 100, CLEAN)],
          "T2": [(0, 100, CLEAN)]}
    pred = predicted_intervals(mon.tracker.events, t_end)
    conf = time_confusion(gt, pred)
    assert conf["accuracy"] > 0.93 and conf["annotated_seconds"] == 200
    assert conf["per_state"][DIRTY]["recall"] > 0.9

    ep = dirty_episodes(gt, pred)
    assert ep["true_dirty_episodes"] == 1 and ep["detected"] == 1 and ep["false_dirty_flags"] == 0
    assert 0 < ep["mean_delay_s"] < 4

    # a system that never says dirty: missed episode; one that cries wolf on T2: false flag
    assert dirty_episodes(gt, {"T1": [(0, 100, CLEAN)]})["recall"] == 0.0
    wolf = dict(pred, T2=[(0, 50, CLEAN), (50, 60, DIRTY), (60, 100, CLEAN)])
    assert dirty_episodes(gt, wolf)["false_dirty_flags"] == 1


def test_zone_discovery_and_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    jitter = lambda b: tuple(v + rng.normal(0, 3) for v in b)
    frames = [[jitter(T2), jitter(T1)] + ([(10, 10, 60, 40)] if k == 7 else []) for k in range(20)]
    frames[3] = [jitter(T1)]                              # T2 hidden in one frame
    zones = make_zones(cluster_boxes(frames))
    assert [z.id for z in zones] == ["T1", "T2"]         # one-off false box dropped, reading order
    assert zones[0].box == pytest.approx(T1, abs=4) and zones[1].box == pytest.approx(T2, abs=4)

    save_zones(zones, tmp_path / "zones.json")
    assert load_zones(tmp_path / "zones.json") == zones
    assert load_zones({"A": [0, 0, 5, 5]}) == [Zone("A", (0.0, 0.0, 5.0, 5.0))]
