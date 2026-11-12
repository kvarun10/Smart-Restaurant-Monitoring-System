# TableWatch - find the dirty tables

In a busy cafe, guests leave and the table sits dirty until someone happens to notice.
TableWatch watches a fixed camera, tracks every table as **occupied**, **clean** or **dirty**,
and raises an alert when a table has been dirty for too long.

**It needs no labelled "dirty table" data and no training.**

> **Status:** code complete and unit-tested on synthetic scenes. Not yet evaluated on real
> footage - the results table below stays empty until it is. Every number in this README
> comes from a run whose evidence is saved under `report/`.

## The idea

Training a detector to recognise "dirty" needs thousands of labelled dirty tables from many
restaurants, and "dirty" does not even look the same everywhere: in some restaurants a
*clean* table is bare, in others it is pre-set with bowls and condiments. (An earlier
attempt at this project went the labelled-data route; auditing that dataset showed most of
its "dirty" labels were on clean, pre-set tables - see `report/01_data_audit/`.)

A fixed camera makes a simpler question available: **does this empty table still look the
way it did when it was clean?**

```
frame --> COCO detector (pretrained) --> people boxes
                                              |
for each table zone:                          v
    someone sitting at it?  ------------>  occupied
    nobody near: compare with the table's
    own clean reference (grid NCC)  ----->  clean  /  dirty
    someone walking in front  ---------->  no vote this frame
                                              |
                                              v
        TableTracker: 3 s vote window, 60 % majority to switch state,
        dirty timer in video time, one alert per dirty episode
                                              |
                                              v
        annotated.mp4   events.json   summary.json   evaluation.json
```

- **Tables** are found once per camera by clustering "dining table" detections across the
  video; the zones can be corrected by hand in a small JSON file.
- **Occupied** uses a pretrained person detector, so it works in any restaurant.
- **Clean vs dirty** compares each table with its own clean reference, cell by cell, using
  normalised cross-correlation: a change in lighting is ignored, a plate that was not there
  before is not. A table's standard place setting is part of its reference, so pre-set
  tables are not mistaken for dirty ones.
- **Time** is what turns flickering per-frame guesses into something staff can act on.

## Results

| Video | Tables | Annotated time | State accuracy | Dirty episodes found | Mean delay | False dirty flags |
|---|---|---|---|---|---|---|
| *pending - needs real fixed-camera footage with ground truth* | | | | | | |

How the numbers are produced: someone watches the raw video and writes down each table's true
state over time (`data/annotations_example.csv` shows the format) *before* looking at the
system's output. `tablewatch.evaluate` then reports, for all annotated table-seconds, how
often the state was right, and for each true dirty episode whether it was flagged, how many
seconds late, and how many dirty flags were false.

## Run it (Google Colab)

Open `notebooks/01_run_on_video.ipynb` in Colab, upload a video from a fixed camera, check
the detected table zones, run. Optional last step: upload ground truth and get the scores.

Recording your own test video:

- Fix the camera (tripod, shelf, tape). A hand-held video will not work.
- Start with the tables clean and empty; that first view is each table's reference.
- Capture full cycles: people sit, eat, leave their dishes, someone clears the table.
- 10-20 minutes with 3-6 tables in view gives a dozen or more dirty episodes.

Locally:

```bash
pip install -e ".[vision]"
python -m tablewatch.pipeline --source cafe.mp4 --out runs/demo
```

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

The suite runs without a GPU or any video. It simulates a camera view (guests arrive, eat,
leave dishes, a waiter walks past, the lights dim, the table is cleared) and checks the
state sequence, alert timing, and the evaluation metrics, alongside unit tests for each part.

## Layout

```
src/tablewatch/
  zones.py        locate tables once per camera
  cleanliness.py  clean-reference comparison (grid normalised cross-correlation)
  monitor.py      per-frame decision for every table
  tracker.py      vote smoothing, dirty timers, alerts (pure Python)
  evaluate.py     scoring against hand-written ground truth
  pipeline.py     everything over a video (CLI)
  data_audit.py, split.py   dataset audit tools used in report/01_data_audit
notebooks/        01_run_on_video (main), legacy_dataset_audit
report/           evidence: audit of the earlier dataset, and one folder per evaluated video
```

## Limitations (known before any evaluation)

- **Fixed camera only.** If the camera moves, the references are invalid.
- **Tables must be clean when first seen empty.** A table that is already dirty at start-up
  is learned as "clean". A production version would let staff confirm or reset a reference.
- **Anything that changes the table's look counts**: a moved chair inside the zone, a strong
  moving shadow, a forgotten bag. Evaluation will show how often this happens.
- **A table re-set differently after cleaning** (setting in a new position) can stay flagged.
- Table discovery relies on COCO's "dining table" class and may miss tables at steep angles;
  zones can be entered by hand.

## Background

This began as a 2025 college team project (Ashish Meena, Shashwat Mishra, Vaidehi Patel,
Vanshika Gautam, Varun Kaushal) that trained a 3-class YOLOv8 detector. This repository is a
new design and implementation by Varun Kaushal; none of the original code is used.
