"""Run the detector + tracker over a video and write an annotated video, events and summary.

    python -m tablewatch.pipeline --weights best.pt --source cafe.mp4 --out runs/demo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .tracker import CLEAN, DIRTY, OCCUPIED, Detection, TableTracker, TrackerConfig, normalize_state

COLORS = {OCCUPIED: (0, 165, 255), CLEAN: (0, 200, 0), DIRTY: (0, 0, 255), None: (160, 160, 160)}  # BGR
SHORT = {OCCUPIED: "occupied", CLEAN: "clean", DIRTY: "DIRTY", None: "?"}


def _mmss(seconds: float) -> str:
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def draw(frame, snapshot: list[dict]):
    import cv2
    for tb in snapshot:
        x1, y1, x2, y2 = map(int, tb["box"])
        color = COLORS[tb["state"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4 if tb["alert"] else 2)
        label = f'{tb["id"]} {SHORT[tb["state"]]} {_mmss(tb["state_for_s"])}'
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - h - 8), (x1 + w + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    alerts = [tb["id"] for tb in snapshot if tb["alert"]]
    if alerts:
        text = "CLEAN NOW: " + ", ".join(alerts)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (0, 0, 255), -1)
        cv2.putText(frame, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


def run_video(
    weights: str,
    source: str,
    out_dir: str,
    config: TrackerConfig | None = None,
    conf: float = 0.35,
    imgsz: int = 640,
    stride: int = 2,
    max_seconds: float | None = None,
) -> dict:
    """Process `source`; returns the summary dict (also written to <out_dir>/summary.json)."""
    import cv2
    from ultralytics import YOLO

    model = YOLO(weights)
    names = {i: normalize_state(n) for i, n in model.names.items()}
    tracker = TableTracker(config)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    idx, t = 0, 0.0
    try:
        while True:
            ok, frame = cap.read()
            t = idx / fps
            if not ok or (max_seconds is not None and t > max_seconds):
                break
            if idx % max(stride, 1) == 0:
                result = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
                dets = [Detection(tuple(b.xyxy[0].tolist()), names[int(b.cls[0])], float(b.conf[0]))
                        for b in result.boxes]
                tracker.update(t, dets)
            writer.write(draw(frame, tracker.snapshot(t)))
            idx += 1
    finally:
        cap.release()
        writer.release()

    summary = tracker.summary(t)
    summary["frames"] = idx
    summary["fps"] = round(fps, 2)
    (out / "events.json").write_text(json.dumps(tracker.events, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--out", default="runs/demo")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--stride", type=int, default=2, help="run the detector on every Nth frame")
    p.add_argument("--dirty-alert-s", type=float, default=300.0)
    p.add_argument("--max-seconds", type=float, default=None)
    a = p.parse_args()
    summary = run_video(a.weights, a.source, a.out, TrackerConfig(dirty_alert_s=a.dirty_alert_s),
                        conf=a.conf, stride=a.stride, max_seconds=a.max_seconds)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
