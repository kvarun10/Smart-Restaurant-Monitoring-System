import pytest

from tablewatch.tracker import (CLEAN, DIRTY, OCCUPIED, Detection, TableTracker, TrackerConfig, iou,
                                normalize_state)

BOX_A = (100, 100, 300, 300)
BOX_B = (500, 100, 700, 300)
FPS = 10


def feed(tracker, t0, seconds, dets_fn):
    """Feed `seconds` of frames starting at t0; dets_fn(frame_index) -> detections."""
    events = []
    n = int(seconds * FPS)
    for i in range(n):
        events += tracker.update(t0 + i / FPS, dets_fn(i))
    return events, t0 + n / FPS


def test_iou():
    assert iou(BOX_A, BOX_A) == pytest.approx(1.0)
    assert iou(BOX_A, BOX_B) == 0.0
    assert iou((0, 0, 2, 2), (1, 0, 3, 2)) == pytest.approx(1 / 3)


def test_normalize_state():
    assert normalize_state("Unoccupied-Dirty") == DIRTY
    with pytest.raises(ValueError):
        normalize_state("chair")


def test_tables_are_discovered_and_keep_their_ids():
    tr = TableTracker()
    feed(tr, 0, 2, lambda i: [Detection(BOX_A, CLEAN), Detection(BOX_B, OCCUPIED)])
    assert [t.id for t in tr.tables] == ["T1", "T2"]
    assert {t.id: t.state for t in tr.tables} == {"T1": CLEAN, "T2": OCCUPIED}
    # jittered boxes still map to the same two tables
    feed(tr, 2, 2, lambda i: [Detection((104, 97, 305, 296), CLEAN), Detection((497, 103, 696, 304), OCCUPIED)])
    assert len(tr.tables) == 2


def test_one_off_false_detection_does_not_create_a_table():
    tr = TableTracker()
    feed(tr, 0, 2, lambda i: [Detection(BOX_A, CLEAN)] + ([Detection(BOX_B, DIRTY)] if i == 3 else []))
    assert len(tr.tables) == 1


def test_flicker_does_not_change_state():
    tr = TableTracker()
    feed(tr, 0, 3, lambda i: [Detection(BOX_A, CLEAN)])
    # every 5th frame the detector wrongly says dirty
    events, _ = feed(tr, 3, 10, lambda i: [Detection(BOX_A, DIRTY if i % 5 == 0 else CLEAN)])
    assert events == []
    assert tr.tables[0].state == CLEAN


def test_real_change_switches_after_a_short_delay():
    tr = TableTracker()
    feed(tr, 0, 3, lambda i: [Detection(BOX_A, OCCUPIED)])
    events, _ = feed(tr, 3, 5, lambda i: [Detection(BOX_A, DIRTY)])
    changes = [e for e in events if e["type"] == "state_change"]
    assert len(changes) == 1 and changes[0]["to"] == DIRTY
    assert 0 < changes[0]["t"] - 3 <= tr.cfg.vote_window_s


def test_dirty_alert_fires_once_and_cleaning_is_logged():
    tr = TableTracker(TrackerConfig(dirty_alert_s=20))
    feed(tr, 0, 3, lambda i: [Detection(BOX_A, OCCUPIED)])
    events, t = feed(tr, 3, 40, lambda i: [Detection(BOX_A, DIRTY)])
    alerts = [e for e in events if e["type"] == "dirty_alert"]
    assert len(alerts) == 1
    assert alerts[0]["dirty_for_s"] == pytest.approx(20, abs=0.2)

    events, t = feed(tr, t, 5, lambda i: [Detection(BOX_A, CLEAN)])
    cleaned = [e for e in events if e.get("to") == CLEAN]
    assert len(cleaned) == 1 and cleaned[0]["dirty_for_s"] > 35
    assert tr.tables[0].alerted is False

    s = tr.summary(t)["tables"]["T1"]
    assert s["alerts"] == 1 and s["dirty_episodes_finished"] == 1
    assert sum(s["seconds_in_state"].values()) == pytest.approx(t - tr.events[0]["t"], abs=0.2)


def test_state_holds_while_table_is_hidden():
    tr = TableTracker(TrackerConfig(dirty_alert_s=10))
    feed(tr, 0, 3, lambda i: [Detection(BOX_A, DIRTY)])
    # a waiter blocks the camera for 20 s: no detections at all
    events, t = feed(tr, 3, 20, lambda i: [])
    assert tr.tables[0].state == DIRTY
    assert [e["type"] for e in events] == ["dirty_alert"]
    assert tr.snapshot(t)[0]["last_seen_s_ago"] > 19


def test_timing_is_independent_of_processing_rate():
    def run(step):
        tr = TableTracker(TrackerConfig(dirty_alert_s=15))
        t = 0.0
        while t < 40:
            tr.update(t, [Detection(BOX_A, OCCUPIED if t < 5 else DIRTY)])
            t += step
        return [e for e in tr.events if e["type"] == "dirty_alert"][0]["t"]
    assert run(0.1) == pytest.approx(run(0.5), abs=1.5)
