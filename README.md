# TableWatch - dirty-table detection for restaurants

Finds every table in a fixed-camera restaurant video, classifies it as **occupied**,
**unoccupied-clean** or **unoccupied-dirty**, and raises an alert when a table has stayed
dirty for too long, so staff can clean it before the next guests arrive.

> **Status: rebuild in progress.** The code and tests in this repo are complete and passing.
> The results table below is filled in only from runs whose evidence is saved in `report/`.
> Nothing in this README is a number that was not measured.

## Results

| | test set | mAP50 | dirty-table recall | dirty tables in test set |
|---|---|---|---|---|
| Run A - random split (leaky, for comparison) | *pending* | | | |
| Run B - group-aware split (**the real result**) | *pending* | | | |

Produced by `notebooks/02_train_eval.ipynb`; raw numbers land in `report/02_train_eval/`.

## Why this is a rebuild

This started as a 5-person college project in 2025
([original repo](https://github.com/vanshika-0305/Smart-Restaurant-Monitoring-System-)).
It reported 0.974 mAP50. Going back over it, that number does not mean what it appears to:

| Problem in the original | Evidence | What this repo does |
|---|---|---|
| The class the product exists for was almost absent | The validation set had **12** dirty tables out of 999 labelled tables (705 occupied, 282 clean) | `01_data_audit` counts every class per split and exports a contact sheet of every dirty label for review |
| Train and validation shared the same videos | Validation files are consecutive frames of one camera (`cam6_1hour_frame_0175`, `_0177`, `_0178`...) while the split was random | `tablewatch.split` puts each source video in exactly one split; a pixel-hash check confirms no near-duplicates cross over |
| No held-out test result was ever reported | The 342-image test split was never evaluated | `02_train_eval` scores on the test split only, per class, and trains the leaky and the leak-free version side by side to measure the inflation |
| Alerts relied on hand-typed table coordinates that matched no real video | `configs/table_config.json` held six made-up boxes | `tablewatch.tracker` discovers tables from the detections and keeps stable IDs |
| Table state flickered frame to frame; timers used the wall clock | A slow machine made tables look dirty for "longer" | Vote-window smoothing with hysteresis; all timing in video time; unit-tested |
| Roboflow API key committed in three files | | Key is read from Colab Secrets; nothing secret is in this repo |

## How it works

```
video frame -> YOLOv8 detector -> per-frame boxes with a state
                                        |
                                 TableTracker
             - matches boxes to known tables (IoU), registers new tables after 5 hits
             - each table keeps 3 s of votes; switches state only on a 60% majority
             - holds the last state while a table is hidden (waiter in front of it)
             - starts a timer on "dirty"; fires one alert at the threshold
                                        |
          annotated.mp4   +   events.json   +   summary.json (time-to-clean per table)
```

## Run it (Google Colab)

1. `notebooks/01_data_audit.ipynb` - audit the dataset, review dirty labels, build the leak-free split (CPU)
2. `notebooks/02_train_eval.ipynb` - train run A and run B, evaluate on held-out test (T4 GPU)
3. `notebooks/03_video_demo.ipynb` - run the full pipeline on a video

Each notebook clones this repo and imports `tablewatch`, so the logic lives in tested code,
not in notebook cells. Each one ends by downloading a zip of its evidence; unzip it into
`report/` and commit.

The dataset is downloaded from Roboflow inside the notebook. Add your key as a Colab secret
named `ROBOFLOW_API_KEY` (key icon in the left sidebar).

## Run the tests (no GPU, no dataset)

```bash
pip install -e ".[dev]"
pytest -q
```

19 tests cover leakage detection, the group-aware split, and the tracker (table discovery,
flicker rejection, alert timing, occlusion, frame-rate independence).

## Layout

```
src/tablewatch/
  data_audit.py   class balance, source-group leakage, near-duplicate check
  split.py        group-aware train/valid/test split
  tracker.py      detections -> stable table states, timers, alerts (pure Python)
  pipeline.py     detector + tracker over a video (CLI: python -m tablewatch.pipeline)
notebooks/        Colab notebooks 01-03
tests/            pytest suite
report/           evidence from each notebook run (metrics JSON, plots, contact sheets)
```

## Known limitations

- Assumes a fixed camera. A moving camera breaks table matching.
- "Dirty" is learned from very few examples; until more dirty-table data from more venues
  is added, treat dirty-table scores as preliminary and read them with the test-set count.
- Not validated on a live camera feed; all evaluation is on recorded video and images.

## Credits

Original 2025 project and dataset collection: Ashish Meena, Shashwat Mishra, Vaidehi Patel,
Vanshika Gautam, Varun Kaushal. This repository is a ground-up rewrite of the code and the
evaluation by Varun Kaushal.
