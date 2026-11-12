"""Per-frame decision for every table, fed into the TableTracker.

    people at the table           -> occupied
    nobody near, matches clean    -> unoccupied_clean
    nobody near, differs          -> unoccupied_dirty
    someone partly in front       -> no vote this frame (the tracker holds the last state)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cleanliness import ReferenceScorer
from .tracker import CLEAN, DIRTY, OCCUPIED, Box, Detection, TableTracker, TrackerConfig
from .zones import Zone


def _expand(box: Box, frac: float) -> Box:
    dx, dy = frac * (box[2] - box[0]), frac * (box[3] - box[1])
    return (box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy)


def overlap_share(table: Box, person: Box) -> float:
    """Intersection area as a share of the smaller of the two boxes."""
    iw = min(table[2], person[2]) - max(table[0], person[0])
    ih = min(table[3], person[3]) - max(table[1], person[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    smaller = min((table[2] - table[0]) * (table[3] - table[1]), (person[2] - person[0]) * (person[3] - person[1]))
    return iw * ih / smaller


@dataclass
class MonitorConfig:
    min_overlap: float = 0.2       # person/table overlap that counts as "sitting at it"
    zone_expand: float = 0.1       # people sit beside a table, so look slightly beyond its box
    dirty_threshold: float = 0.10  # share of the table that must differ from its clean reference
    tracker: TrackerConfig = field(default_factory=TrackerConfig)


class TableMonitor:
    """Assumes every table is clean the first time it is seen unoccupied (see README: limitations)."""

    def __init__(self, zones: list[Zone], config: MonitorConfig | None = None,
                 scorer: ReferenceScorer | None = None):
        self.zones, self.cfg = zones, config or MonitorConfig()
        self.scorer = scorer or ReferenceScorer()
        self.tracker = TableTracker(self.cfg.tracker)
        for z in zones:
            self.tracker.add_table(z.id, z.box)
        self.scores: dict[str, float | None] = {z.id: None for z in zones}

    def update(self, t: float, frame: np.ndarray, person_boxes: list[Box]) -> list[dict]:
        """`frame` is the full image (H x W or H x W x 3, uint8). Returns new tracker events."""
        cfg, detections = self.cfg, []
        height, width = frame.shape[:2]
        for z in self.zones:
            reach = _expand(z.box, cfg.zone_expand)
            overlaps = [overlap_share(reach, p) for p in person_boxes]
            if any(v >= cfg.min_overlap for v in overlaps):
                state, self.scores[z.id] = OCCUPIED, None
            elif any(overlap_share(z.box, p) > 0 for p in person_boxes):
                continue  # table partly hidden by a passer-by: cannot judge it, do not vote
            else:
                x1, y1 = max(0, int(z.box[0])), max(0, int(z.box[1]))
                x2, y2 = min(width, int(z.box[2])), min(height, int(z.box[3]))
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                if not self.scorer.ready(z.id):
                    self.scorer.warm_up(z.id, crop)
                    state, self.scores[z.id] = CLEAN, 0.0
                else:
                    score = self.scorer.score(z.id, crop)
                    state = DIRTY if score >= cfg.dirty_threshold else CLEAN
                    self.scores[z.id] = score
                    if state == CLEAN and self.tracker.table(z.id).state == CLEAN:
                        self.scorer.adapt(z.id, crop)
            detections.append(Detection(z.box, state, 1.0))
        return self.tracker.update(t, detections)

    def snapshot(self, t: float) -> list[dict]:
        return [{**s, "score": self.scores.get(s["id"])} for s in self.tracker.snapshot(t)]
