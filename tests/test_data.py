import numpy as np
import pytest
from PIL import Image

from tablewatch.data_audit import (audit, dhash, leakage_report, load_yolo_dataset, near_duplicate_report,
                                   source_group, venue_group)
from tablewatch.split import group_split, write_split

NAMES = ["occupied", "unoccupied_clean", "unoccupied_dirty"]


@pytest.mark.parametrize("stem,group", [
    ("cam6_1hour_frame_0175_jpg.rf.a832d26d3ae80c953e4390cac8c2f50c", "cam6_1hour"),
    ("CRI_CH05_1_jpg.rf.2412d1f8cb01e33460d04289b5170b1d", "cri_ch05"),
    ("restaurant.4", "restaurant"),
    ("cafe-video (12)", "cafe-video"),
    ("0001", "0001"),
])
def test_source_group(stem, group):
    assert source_group(stem) == group


@pytest.mark.parametrize("stem,venue", [
    ("BCP_N1-D2_1_jpg.rf.a832d26d3ae80c953e4390cac8c2f50c", "bcp"),
    ("bnm_n2-ipc_2", "bnm"),
    ("CRI_CH05_1", "cri"),
    ("pks_d12_1", "pks"),
    ("jh-nma_n3-d5_2", "jh-nma"),
    ("cam6_1hour_frame_0175", "1hour"),
    ("whatsapp-image-2025-03-26-at-19_35_27_d98", "whatsapp-image-2025-03-26-at-19_35_27_d98"),
])
def test_venue_group(stem, venue):
    assert venue_group(stem) == venue


def _scene(seed):
    """A smooth random 'scene'; frames of the same scene differ only by faint noise."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (6, 6, 3), dtype=np.uint8)
    return np.asarray(Image.fromarray(small).resize((64, 64), Image.BICUBIC))


def _write(root, split, name, pixels, label_lines):
    (root / split / "images").mkdir(parents=True, exist_ok=True)
    (root / split / "labels").mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(root / split / "images" / f"{name}.png")
    (root / split / "labels" / f"{name}.txt").write_text("\n".join(label_lines))


@pytest.fixture
def leaky_dataset(tmp_path):
    """5 videos x 10 frames, split at random by frame -> every video is in every split."""
    rng = np.random.default_rng(0)
    for v in range(5):
        scene = _scene(v)
        for f in range(10):
            noise = rng.integers(-3, 4, scene.shape)
            px = np.clip(scene.astype(int) + noise, 0, 255).astype(np.uint8)
            split = ["train", "train", "train", "valid", "test"][f % 5]
            labels = ["0 0.5 0.5 0.2 0.2", "1 0.2 0.2 0.1 0.1"] + (["2 0.8 0.8 0.1 0.1"] if v < 3 else [])
            _write(tmp_path, split, f"cam{v}_video_frame_{f:04d}", px, labels)
    return tmp_path


def test_dhash_separates_scenes_but_not_frames(tmp_path):
    a, b = _scene(1), _scene(2)
    for name, px in [("a", a), ("a2", np.clip(a.astype(int) + 2, 0, 255).astype(np.uint8)), ("b", b)]:
        Image.fromarray(px).save(tmp_path / f"{name}.png")
    h = {n: dhash(tmp_path / f"{n}.png") for n in ("a", "a2", "b")}
    assert bin(h["a"] ^ h["a2"]).count("1") <= 2
    assert bin(h["a"] ^ h["b"]).count("1") > 10


def test_audit_flags_leakage(leaky_dataset):
    report = audit(leaky_dataset, NAMES)
    assert report["n_images"] == 50
    assert report["group_leakage"]["valid"]["share"] == 1.0
    assert report["near_duplicates"]["test"]["share"] > 0.9
    assert report["class_table"]["train"]["classes"]["unoccupied_dirty"]["instances"] == 18


def test_group_split_removes_leakage(leaky_dataset, tmp_path_factory):
    records = load_yolo_dataset(leaky_dataset)
    assignment = group_split(records, n_classes=3, ratios=(0.6, 0.2, 0.2), seed=0)
    assert set(assignment.values()) == {"train", "valid", "test"}

    out = tmp_path_factory.mktemp("resplit")
    yaml_path = write_split(records, assignment, out, NAMES)
    assert "unoccupied_dirty" in yaml_path.read_text()

    new = load_yolo_dataset(out)
    assert len(new) == 50
    assert all(v["share"] == 0.0 for v in leakage_report(new).values())
    assert all(v["share"] == 0.0 for v in near_duplicate_report(new).values())


def test_group_split_needs_enough_groups(tmp_path):
    for f in range(4):
        _write(tmp_path, "train", f"only_video_frame_{f}", _scene(0), ["0 0.5 0.5 0.2 0.2"])
    with pytest.raises(ValueError, match="source group"):
        group_split(load_yolo_dataset(tmp_path), n_classes=3)


def test_polygon_labels_become_boxes(tmp_path):
    _write(tmp_path, "train", "poly_1", _scene(0), ["2 0.1 0.1 0.5 0.1 0.5 0.3 0.1 0.3"])
    (cls, cx, cy, w, h), = load_yolo_dataset(tmp_path)[0].boxes
    assert (cls, cx, cy, w, h) == pytest.approx((2, 0.3, 0.2, 0.4, 0.2))
