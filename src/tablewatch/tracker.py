"""Turn noisy per-frame detections into stable per-table states, timers and alerts.

Pure Python (no torch / OpenCV) so the logic is unit-tested without a GPU.

Design:
* Tables are static furniture seen by a fixed camera, so tables are *discovered* from
  the detections themselves (no hand-written coordinates per restaurant) and matched
  frame-to-frame by box overlap.
* A detector flickers: one frame says "dirty", the next says "clean". Each table keeps a
  short window of recent votes and only switches state when one state clearly wins.
* All timing uses the video timestamp passed in by the caller, never the wall clock, so
  results are identical whether the video is processed at 5 FPS or 500 FPS.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

OCCUPIED, CLEAN, DIRTY = "occupied", "unoccupied_clean", "unoccupied_dirty"
STATES = (OCCUPIED, CLEAN, DIRTY)

Box = tuple[float, float, float, float]  # x1, y1, x2, y2


def normalize_state(name: str) -> str:
    """'Unoccupied-Dirty' -> 'unoccupied_dirty' (class names differ between exports)."""
    s = name.strip().lower().replace("-", "_").replace(" ", "_")
    if s not in STATES:
        raise ValueError(f"Unknown table state {name!r}; expected one of {STATES}")
    return s


def iou(a: Box, b: Box) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


@dataclass(frozen=True)
class Detection:
    box: Box
    state: str
    conf: float = 1.0


@dataclass
class TrackerConfig:
    match_iou: float = 0.3         # min overlap for a detection to belong to a known table
    confirm_hits: int = 5          # detections needed before a new table is registered...
    confirm_window_s: float = 3.0  # ...within this many seconds (filters one-off false boxes)
    vote_window_s: float = 3.0     # how much recent history decides a table's state
    min_votes: int = 3             # never decide on fewer votes than this
    switch_share: float = 0.6      # confidence-weighted vote share needed to switch state
    dirty_alert_s: float = 300.0   # alert when a table has been dirty this long
    box_momentum: float = 0.9      # smoothing of the stored table box


@dataclass
class Table:
    id: str
    box: Box
    state: str | None = None
    state_since: float = 0.0
    last_seen: float = 0.0
    alerted: bool = False
    votes: deque = field(default_factory=deque)          # (t, state, conf)
    time_in_state: dict = field(default_factory=lambda: {s: 0.0 for s in STATES})
    dirty_durations: list = field(default_factory=list)  # finished dirty episodes, seconds


class TableTracker:
    def __init__(self, config: TrackerConfig | None = None):
        self.cfg = config or TrackerConfig()
        self.tables: list[Table] = []
        self._candidates: list[dict] = []  # {"box": Box, "hits": deque[(t, state, conf)]}
        self.events: list[dict] = []

    # ------------------------------------------------------------------ update
    def update(self, t: float, detections: list[Detection]) -> list[dict]:
        """Feed all detections of the frame at video time `t` (seconds). Returns new events."""
        cfg, new_events = self.cfg, []

        for det in detections:
            table = self._best_match(det.box, self.tables)
            if table is not None:
                table.votes.append((t, det.state, det.conf))
                table.last_seen = t
                m = cfg.box_momentum
                table.box = tuple(m * a + (1 - m) * b for a, b in zip(table.box, det.box))
            else:
                self._add_to_candidates(t, det)

        for table in self.tables:
            while table.votes and table.votes[0][0] < t - cfg.vote_window_s:
                table.votes.popleft()
            winner = self._vote(table)
            if winner is not None and winner != table.state:
                new_events.append(self._transition(table, winner, t))
            if (table.state == DIRTY and not table.alerted
                    and t - table.state_since >= cfg.dirty_alert_s):
                table.alerted = True
                new_events.append({"t": round(t, 2), "table": table.id, "type": "dirty_alert",
                                   "dirty_for_s": round(t - table.state_since, 2)})

        self.events.extend(new_events)
        return new_events

    # ----------------------------------------------------------------- helpers
    def _best_match(self, box: Box, items):
        best, best_iou = None, self.cfg.match_iou
        for item in items:
            other = item.box if isinstance(item, Table) else item["box"]
            v = iou(box, other)
            if v >= best_iou:
                best, best_iou = item, v
        return best

    def _add_to_candidates(self, t: float, det: Detection) -> None:
        cfg = self.cfg
        self._candidates = [c for c in self._candidates if c["hits"][-1][0] >= t - cfg.confirm_window_s]
        cand = self._best_match(det.box, self._candidates)
        if cand is None:
            cand = {"box": det.box, "hits": deque()}
            self._candidates.append(cand)
        cand["hits"].append((t, det.state, det.conf))
        while cand["hits"][0][0] < t - cfg.confirm_window_s:
            cand["hits"].popleft()
        if len(cand["hits"]) >= cfg.confirm_hits:
            self._candidates.remove(cand)
            self.tables.append(Table(id=f"T{len(self.tables) + 1}", box=cand["box"],
                                     state_since=t, last_seen=t, votes=cand["hits"]))

    def _vote(self, table: Table) -> str | None:
        if len(table.votes) < self.cfg.min_votes:
            return None
        weight = {s: 0.0 for s in STATES}
        for _, state, conf in table.votes:
            weight[state] += conf
        winner = max(weight, key=weight.get)
        total = sum(weight.values())
        return winner if total > 0 and weight[winner] / total >= self.cfg.switch_share else None

    def _transition(self, table: Table, new_state: str, t: float) -> dict:
        event = {"t": round(t, 2), "table": table.id, "type": "state_change",
                 "from": table.state, "to": new_state}
        if table.state is not None:
            elapsed = t - table.state_since
            table.time_in_state[table.state] += elapsed
            if table.state == DIRTY:
                table.dirty_durations.append(elapsed)
                event["dirty_for_s"] = round(elapsed, 2)
        table.state, table.state_since, table.alerted = new_state, t, False
        return event

    # ----------------------------------------------------------------- outputs
    def snapshot(self, t: float) -> list[dict]:
        """Current state of every table, for drawing or a dashboard."""
        return [{
            "id": tb.id, "box": tuple(round(v, 1) for v in tb.box), "state": tb.state,
            "state_for_s": round(t - tb.state_since, 2) if tb.state else 0.0,
            "alert": tb.alerted, "last_seen_s_ago": round(t - tb.last_seen, 2),
        } for tb in self.tables]

    def summary(self, t_end: float) -> dict:
        """Per-table totals up to video time `t_end`."""
        tables = {}
        for tb in self.tables:
            time_in = dict(tb.time_in_state)
            if tb.state is not None:
                time_in[tb.state] += t_end - tb.state_since
            finished = tb.dirty_durations
            tables[tb.id] = {
                "seconds_in_state": {s: round(v, 2) for s, v in time_in.items()},
                "dirty_episodes_finished": len(finished),
                "mean_dirty_duration_s": round(sum(finished) / len(finished), 2) if finished else None,
                "alerts": sum(1 for e in self.events if e["type"] == "dirty_alert" and e["table"] == tb.id),
            }
        return {"duration_s": round(t_end, 2), "n_tables": len(self.tables), "tables": tables}
