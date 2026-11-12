"""Group-aware train/valid/test split.

Every image from one source (video / camera) goes to exactly one split, so the model is
always evaluated on scenes it never saw in training.
"""
from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data_audit import ImageRecord

SPLITS = ("train", "valid", "test")


def group_split(
    records: list[ImageRecord],
    n_classes: int,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> dict[str, str]:
    """Assign each source group to a split. Returns {group: split}.

    Greedy: biggest groups first, each goes to the split that is furthest below its
    target share, measured jointly on image count and per-class instance counts so the
    rare class gets spread across splits when the groups allow it.
    """
    vectors: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(n_classes + 1))
    for r in records:
        v = vectors[r.group]
        v[0] += 1
        for cls, *_ in r.boxes:
            v[1 + cls] += 1
    if len(vectors) < len(SPLITS):
        raise ValueError(
            f"Only {len(vectors)} source group(s) found; need at least {len(SPLITS)} to split by group. "
            "Check that source_group() is reading your file names correctly."
        )

    total = np.maximum(sum(vectors.values()), 1)
    current = {s: np.zeros(n_classes + 1) for s in SPLITS}
    groups = sorted(vectors)
    random.Random(seed).shuffle(groups)
    groups.sort(key=lambda g: -vectors[g][0])  # stable: ties keep the shuffled order

    assignment = {}
    for g in groups:
        def need(split_ratio):  # mean remaining need, relative to that split's own target
            s, ratio = split_ratio
            return float(np.mean((ratio * total - current[s]) / (ratio * total)))
        best = max(zip(SPLITS, ratios), key=need)[0]
        assignment[g] = best
        current[best] += vectors[g]
    return assignment


def write_split(
    records: list[ImageRecord],
    assignment: dict[str, str],
    out_dir: str | Path,
    names: list[str],
) -> Path:
    """Copy images/labels into <out_dir>/<split>/{images,labels} and write data.yaml."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for s in SPLITS:
        (out_dir / s / "images").mkdir(parents=True)
        (out_dir / s / "labels").mkdir(parents=True)
    for r in records:
        s = assignment[r.group]
        shutil.copy2(r.image_path, out_dir / s / "images" / r.image_path.name)
        label_dst = out_dir / s / "labels" / (r.stem + ".txt")
        if r.label_path is not None:
            shutil.copy2(r.label_path, label_dst)
        else:
            label_dst.touch()
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(
        f"path: {out_dir.resolve().as_posix()}\n"
        "train: train/images\nval: valid/images\ntest: test/images\n"
        f"nc: {len(names)}\nnames: {json.dumps(names)}\n"
    )
    return yaml_path
