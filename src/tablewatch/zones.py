"""Table zones: where each table is in a fixed camera's view.

Tables do not move, so they are located once per camera - either automatically, by
clustering "dining table" detections from many frames, or from a hand-edited JSON file.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from .tracker import Box, iou


@dataclass(frozen=True)
class Zone:
    id: str
    box: Box  # x1, y1, x2, y2 in pixels


def cluster_boxes(per_frame_boxes: list[list[Box]], match_iou: float = 0.5, min_share: float = 0.3) -> list[Box]:
    """Merge detections from many frames into one stable box per table.

    A table is kept only if it was detected in at least `min_share` of the frames, which
    drops one-off false detections. Each kept box is the per-coordinate median, so a few
    bad frames (someone standing in front of the table) do not shift it.
    """
    clusters: list[list[Box]] = []
    for boxes in per_frame_boxes:
        for box in boxes:
            best, best_iou = None, match_iou
            for cluster in clusters:
                v = iou(box, cluster[0])
                if v >= best_iou:
                    best, best_iou = cluster, v
            if best is None:
                clusters.append([box])
            else:
                best.append(box)
    need = max(1, min_share * len(per_frame_boxes))
    kept = [tuple(statistics.median(b[i] for b in c) for i in range(4)) for c in clusters if len(c) >= need]
    kept.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    unique: list[Box] = []
    for box in kept:  # drop a smaller duplicate of a table that is already kept
        if all(iou(box, u) < match_iou for u in unique):
            unique.append(box)
    return unique


def make_zones(boxes: list[Box]) -> list[Zone]:
    """Number tables T1, T2, ... in reading order (top rows first, left to right)."""
    if not boxes:
        return []
    row_height = statistics.median(b[3] - b[1] for b in boxes)
    ordered = sorted(boxes, key=lambda b: (int((b[1] + b[3]) / 2 // row_height), b[0]))
    return [Zone(f"T{i + 1}", tuple(round(v, 1) for v in b)) for i, b in enumerate(ordered)]


def save_zones(zones: list[Zone], path: str | Path) -> None:
    Path(path).write_text(json.dumps({z.id: list(z.box) for z in zones}, indent=2))


def load_zones(source: str | Path | dict) -> list[Zone]:
    """From a {"T1": [x1, y1, x2, y2], ...} dict or a JSON file containing one."""
    data = source if isinstance(source, dict) else json.loads(Path(source).read_text())
    return [Zone(str(k), tuple(float(v) for v in box)) for k, box in data.items()]
