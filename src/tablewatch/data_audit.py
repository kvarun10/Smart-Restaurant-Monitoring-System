"""Audit a YOLO-format dataset: class balance, train/eval leakage, near-duplicate images.

Only needs numpy and Pillow, so it runs anywhere (no GPU, no ultralytics).

Why this exists: the images in this project are mostly frames cut from a handful of
CCTV videos. If frames from one video land in both train and validation, the model is
scored on scenes it has effectively memorised and the reported mAP is inflated.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_DIRS = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}

_ROBOFLOW_HASH = re.compile(r"\.rf\.[0-9a-f]{8,}$", re.I)
_EXT_SUFFIX = re.compile(r"_(jpe?g|png|bmp|webp)$", re.I)
# trailing frame counters: "_frame_0175", "_12", ".4", " (3)", "-0007"
_FRAME_COUNTER = re.compile(r"(?:[\s_\-.]+(?:frame|img|image)?[\s_\-.]*\(?\d+\)?)+$", re.I)


def source_group(stem: str) -> str:
    """Best-effort source (video / camera / photo set) of an image, from its file name.

    >>> source_group("cam6_1hour_frame_0175_jpg.rf.a832d26d3ae80c953e4390cac8c2f50c")
    'cam6_1hour'
    """
    s = _ROBOFLOW_HASH.sub("", stem)
    s = _EXT_SUFFIX.sub("", s)
    g = _FRAME_COUNTER.sub("", s)
    return (g or s).lower()


@dataclass
class ImageRecord:
    split: str
    image_path: Path
    label_path: Path | None
    stem: str
    group: str
    boxes: list[tuple[int, float, float, float, float]] = field(default_factory=list)  # cls, cx, cy, w, h


def _parse_label_file(path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        vals = [float(v) for v in parts[1:]]
        if len(vals) == 4:
            cx, cy, w, h = vals
        else:  # polygon export: x1 y1 x2 y2 ... -> enclosing box
            xs, ys = vals[0::2], vals[1::2]
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            w, h = max(xs) - min(xs), max(ys) - min(ys)
        boxes.append((cls, cx, cy, w, h))
    return boxes


def load_yolo_dataset(root: str | Path, group_fn=source_group) -> list[ImageRecord]:
    """Read every split of a YOLO dataset laid out as <root>/<split>/{images,labels}."""
    root = Path(root)
    records = []
    for dirname, split in SPLIT_DIRS.items():
        img_dir = root / dirname / "images"
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            label = root / dirname / "labels" / (img.stem + ".txt")
            has_label = label.is_file()
            records.append(ImageRecord(
                split=split, image_path=img, label_path=label if has_label else None,
                stem=img.stem, group=group_fn(img.stem),
                boxes=_parse_label_file(label) if has_label else [],
            ))
    if not records:
        raise FileNotFoundError(f"No images found under {root}/<train|valid|test>/images")
    return records


def class_table(records: list[ImageRecord], names: list[str]) -> dict:
    """{split: {class_name: {"instances": n, "images": n}}} plus image totals per split."""
    out: dict = {}
    for split in sorted({r.split for r in records}):
        recs = [r for r in records if r.split == split]
        inst, imgs = Counter(), Counter()
        for r in recs:
            classes = [b[0] for b in r.boxes]
            inst.update(classes)
            imgs.update(set(classes))
        out[split] = {
            "n_images": len(recs),
            "n_background_images": sum(1 for r in recs if not r.boxes),
            "classes": {name: {"instances": inst[i], "images": imgs[i]} for i, name in enumerate(names)},
        }
    return out


def group_table(records: list[ImageRecord]) -> list[dict]:
    """One row per source group: how many images it contributes to each split."""
    per_group: dict = defaultdict(Counter)
    for r in records:
        per_group[r.group][r.split] += 1
    rows = [{"group": g, "total": sum(c.values()), **dict(c)} for g, c in per_group.items()]
    return sorted(rows, key=lambda row: -row["total"])


def leakage_report(records: list[ImageRecord]) -> dict:
    """For each eval split: share of images whose source group is also in train."""
    train_groups = {r.group for r in records if r.split == "train"}
    out = {}
    for split in ("valid", "test"):
        recs = [r for r in records if r.split == split]
        if not recs:
            continue
        leaked = [r for r in recs if r.group in train_groups]
        out[split] = {
            "n_images": len(recs),
            "n_from_groups_also_in_train": len(leaked),
            "share": round(len(leaked) / len(recs), 4),
            "shared_groups": sorted({r.group for r in leaked}),
        }
    return out


def dhash(path: str | Path, size: int = 8) -> int:
    """64-bit difference hash; near-identical images differ in only a few bits."""
    with Image.open(path) as im:
        px = np.asarray(im.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _hamming_to_all(h: int, others: np.ndarray) -> np.ndarray:
    xor = np.bitwise_xor(others, np.uint64(h))
    return np.unpackbits(xor.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def near_duplicate_report(records: list[ImageRecord], max_distance: int = 4) -> dict:
    """For each eval split: share of images with a near-identical twin in train.

    Independent of file names, so it also catches leakage when names carry no source info.
    """
    hashes = {id(r): dhash(r.image_path) for r in records}
    train = [r for r in records if r.split == "train"]
    train_h = np.array([hashes[id(r)] for r in train], dtype=np.uint64)
    out = {}
    for split in ("valid", "test"):
        recs = [r for r in records if r.split == split]
        if not recs or not train:
            continue
        pairs = []
        for r in recs:
            d = _hamming_to_all(hashes[id(r)], train_h)
            j = int(d.argmin())
            if d[j] <= max_distance:
                pairs.append({"eval": r.image_path.name, "train": train[j].image_path.name, "distance": int(d[j])})
        out[split] = {
            "n_images": len(recs),
            "n_with_near_duplicate_in_train": len(pairs),
            "share": round(len(pairs) / len(recs), 4),
            "max_distance": max_distance,
            "examples": pairs[:20],
        }
    return out


def audit(root: str | Path, names: list[str], group_fn=source_group, max_distance: int = 4) -> dict:
    """Run every check and return one JSON-serialisable report."""
    records = load_yolo_dataset(root, group_fn)
    return {
        "n_images": len(records),
        "class_table": class_table(records, names),
        "groups": group_table(records),
        "group_leakage": leakage_report(records),
        "near_duplicates": near_duplicate_report(records, max_distance),
    }
