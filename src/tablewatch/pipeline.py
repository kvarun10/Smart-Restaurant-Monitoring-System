"""Run TableWatch over a video from a fixed camera.

    python -m tablewatch.pipeline --source cafe.mp4 --out runs/demo [--zones zones.json]

Uses an off-the-shelf COCO detector (people + dining tables); nothing is trained.
Writes annotated.mp4, zones.json, zones_preview.jpg, events.json and summary.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .monitor import MonitorConfig, TableMonitor
from .tracker import CLEAN, DIRTY, OCCUPIED, TrackerConfig
from .zones import Zone, cluster_boxes, load_zones, make_zones, save_zones

COCO_PERSON, COCO_DINING_TABLE = 0, 60
COLORS = {OCCUPIED: (0, 165, 255), CLEAN: (0, 200, 0), DIRTY: (0, 0, 255), None: (160, 160, 160)}  # BGR
SHORT = {OCCUPIED: "occupied", CLEAN: "clean", DIRTY: "DIRTY", None: "?"}


def _mmss(seconds: float) -> str:
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def _boxes(result) -> list[tuple]:
    return [tuple(b.xyxy[0].tolist()) for b in result.boxes]


def discover_zones(model, source: str, n_frames: int = 40, conf: float = 0.2, imgsz: int = 960,
                   max_seconds: float | None = None) -> list[Zone]:
    """Locate tables by detecting 'dining table' in frames spread across the video."""
    import cv2
    cap = cv2.VideoCapture(source)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_seconds is not None:
        total = min(total, int(max_seconds * (cap.get(cv2.CAP_PROP_FPS) or 25.0)))
    per_frame = []
    for k in range(n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k * max(total - 1, 0) / max(n_frames - 1, 1)))
        ok, frame = cap.read()
        if ok:
            per_frame.append(_boxes(model.predict(frame, classes=[COCO_DINING_TABLE], conf=conf,
                                                  imgsz=imgsz, verbose=False)[0]))
    cap.release()
    return make_zones(cluster_boxes(per_frame))


def draw(frame, snapshot: list[dict]):
    import cv2
    for tb in snapshot:
        x1, y1, x2, y2 = map(int, tb["box"])
        color = COLORS[tb["state"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4 if tb["alert"] else 2)
        label = f'{tb["id"]} {SHORT[tb["state"]]} {_mmss(tb["state_for_s"])}'
        if tb.get("score") is not None:
            label += f' d={tb["score"]:.2f}'
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - h - 8), (x1 + w + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    alerts = [tb["id"] for tb in snapshot if tb["alert"]]
    if alerts:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (0, 0, 255), -1)
        cv2.putText(frame, "CLEAN NOW: " + ", ".join(alerts), (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def run_video(
    source: str,
    out_dir: str,
    zones: list[Zone] | str | dict | None = None,
    model: str = "yolo11s.pt",
    config: MonitorConfig | None = None,
    person_conf: float = 0.3,
    imgsz: int = 960,
    stride: int = 3,
    max_seconds: float | None = None,
) -> dict:
    """Process `source`; returns the summary dict (also written to <out_dir>/summary.json)."""
    import cv2
    from ultralytics import YOLO

    detector = YOLO(model)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if zones is None:
        zones = discover_zones(detector, source, imgsz=imgsz, max_seconds=max_seconds)
        if not zones:
            raise RuntimeError("No tables found automatically. Pass zones={'T1': [x1, y1, x2, y2], ...}.")
    elif not isinstance(zones, list):
        zones = load_zones(zones)
    save_zones(zones, out / "zones.json")

    monitor = TableMonitor(zones, config)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    idx, t = 0, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (max_seconds is not None and idx / fps > max_seconds):
                break
            t = idx / fps
            if idx == 0:
                cv2.imwrite(str(out / "zones_preview.jpg"), draw(frame.copy(), monitor.snapshot(0.0)))
            if idx % max(stride, 1) == 0:
                people = _boxes(detector.predict(frame, classes=[COCO_PERSON], conf=person_conf,
                                                 imgsz=imgsz, verbose=False)[0])
                monitor.update(t, frame, people)
            writer.write(draw(frame, monitor.snapshot(t)))
            idx += 1
    finally:
        cap.release()
        writer.release()

    summary = monitor.tracker.summary(t)
    summary.update(frames=idx, fps=round(fps, 2), model=model)
    (out / "events.json").write_text(json.dumps(monitor.tracker.events, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True)
    p.add_argument("--out", default="runs/demo")
    p.add_argument("--zones", default=None, help="JSON file {table_id: [x1, y1, x2, y2]}; default: auto-detect")
    p.add_argument("--model", default="yolo11s.pt")
    p.add_argument("--stride", type=int, default=3, help="analyse every Nth frame")
    p.add_argument("--dirty-threshold", type=float, default=0.10)
    p.add_argument("--dirty-alert-s", type=float, default=300.0)
    p.add_argument("--max-seconds", type=float, default=None)
    a = p.parse_args()
    cfg = MonitorConfig(dirty_threshold=a.dirty_threshold, tracker=TrackerConfig(dirty_alert_s=a.dirty_alert_s))
    print(json.dumps(run_video(a.source, a.out, a.zones, a.model, cfg, stride=a.stride,
                               max_seconds=a.max_seconds), indent=2))


if __name__ == "__main__":
    main()
