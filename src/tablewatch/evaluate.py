"""Score the system against hand-written ground truth for a video.

Ground truth is a CSV with one row per interval:

    table,start_s,end_s,state
    T1,0,95,occupied
    T1,95,260,unoccupied_dirty
    T1,260,600,unoccupied_clean

Two views of quality:
* time_confusion  - for how many seconds was each table's state right or wrong?
* dirty_episodes  - the product question: was each dirty table noticed, how fast,
                    and how many dirty flags were false?
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from .tracker import DIRTY, STATES, normalize_state

Intervals = dict[str, list[tuple[float, float, str]]]  # table -> [(start_s, end_s, state)]
UNKNOWN = "unknown"


def load_annotations(path: str | Path) -> Intervals:
    out: Intervals = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["table"].strip()].append((float(row["start_s"]), float(row["end_s"]),
                                              normalize_state(row["state"])))
    return {k: sorted(v) for k, v in out.items()}


def predicted_intervals(events: list[dict], t_end: float) -> Intervals:
    """Rebuild each table's state timeline from the tracker's state_change events."""
    changes = defaultdict(list)
    for e in events:
        if e["type"] == "state_change":
            changes[e["table"]].append((e["t"], e["to"]))
    out: Intervals = {}
    for table, seq in changes.items():
        seq.sort()
        out[table] = [(t, seq[i + 1][0] if i + 1 < len(seq) else t_end, state)
                      for i, (t, state) in enumerate(seq)]
    return out


def _state_at(intervals: list[tuple[float, float, str]], t: float) -> str:
    for start, end, state in intervals:
        if start <= t < end:
            return state
    return UNKNOWN


def time_confusion(gt: Intervals, pred: Intervals, step: float = 0.5) -> dict:
    """Seconds spent in each (true state, predicted state) pair, over all annotated time."""
    labels = list(STATES) + [UNKNOWN]
    matrix = {g: {p: 0.0 for p in labels} for g in STATES}
    for table, intervals in gt.items():
        for start, end, state in intervals:
            t = start
            while t < end:
                matrix[state][_state_at(pred.get(table, []), t)] += step
                t += step
    total = sum(sum(row.values()) for row in matrix.values())
    correct = sum(matrix[s][s] for s in STATES)
    per_state = {}
    for s in STATES:
        true_s = sum(matrix[s].values())
        pred_s = sum(matrix[g][s] for g in STATES)
        per_state[s] = {"recall": round(matrix[s][s] / true_s, 4) if true_s else None,
                        "precision": round(matrix[s][s] / pred_s, 4) if pred_s else None,
                        "true_seconds": round(true_s, 1)}
    return {"seconds": matrix, "accuracy": round(correct / total, 4) if total else None,
            "per_state": per_state, "annotated_seconds": round(total, 1)}


def _overlaps(a: tuple, b: tuple) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def dirty_episodes(gt: Intervals, pred: Intervals) -> dict:
    """Episode-level: each true dirty period should be flagged once, quickly, with no false flags."""
    delays, missed, false_flags = [], 0, 0
    for table, intervals in gt.items():
        true_dirty = [iv for iv in intervals if iv[2] == DIRTY]
        pred_dirty = [iv for iv in pred.get(table, []) if iv[2] == DIRTY]
        for ep in true_dirty:
            hits = [p for p in pred_dirty if _overlaps(ep, p)]
            if hits:
                delays.append(max(0.0, min(p[0] for p in hits) - ep[0]))
            else:
                missed += 1
        annotated = [iv for iv in intervals]
        for p in pred_dirty:
            inside_annotation = any(_overlaps(p, iv) for iv in annotated)
            if inside_annotation and not any(_overlaps(p, ep) for ep in true_dirty):
                false_flags += 1
    n = len(delays) + missed
    hours = sum(end - start for ivs in gt.values() for start, end, _ in ivs) / 3600
    return {
        "true_dirty_episodes": n,
        "detected": len(delays),
        "recall": round(len(delays) / n, 4) if n else None,
        "mean_delay_s": round(sum(delays) / len(delays), 2) if delays else None,
        "max_delay_s": round(max(delays), 2) if delays else None,
        "false_dirty_flags": false_flags,
        "false_flags_per_table_hour": round(false_flags / hours, 3) if hours else None,
        "annotated_table_hours": round(hours, 3),
    }


def evaluate(annotation_csv: str | Path, events: list[dict], t_end: float) -> dict:
    gt = load_annotations(annotation_csv)
    pred = predicted_intervals(events, t_end)
    return {"time": time_confusion(gt, pred), "dirty_episodes": dirty_episodes(gt, pred)}
