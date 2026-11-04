# Offline test of the feature-scoring plan

Scripts only. They do not change the app or the plan.

The phone export is read from `/Users/nadavnissim/Downloads/photocleaner-export` (691 labeled photos). Labels stay keep = 0, delete = 1.

## Setup

```bash
conda activate photo-cleaner
cd experiments/ai-models-changes
python extract_features.py
python evaluate_models.py
```

`run_all.py` does both. Cached arrays and the written summary land in `output/`, which is gitignored because it is derived from a personal photo library. The write-up of this run is `FINDINGS.md`.

Two swipe sessions, with the first session's rows discarded before the second:

```bash
python incremental_study.py
```

That uses the cached features. It does not need the photo export. The write-up is the "Two sessions" section of `FINDINGS.md`.

The same 400 labels split as 200 then 200, and as 20, 20, 50, 30, 70, 50, 160:

```bash
python fractured_study.py
```

The write-up is the "Uneven sessions" section of `FINDINGS.md`. LightGBM on macOS needs OpenMP in this env (`conda install -n photo-cleaner -c conda-forge llvm-openmp`). The script pins OpenMP to one thread so it can load next to PyTorch.

## What is simulated

Ultralytics YOLO11n replaces YOLOX-Nano. Both are COCO detectors, and the box aggregation follows the plan (confidence 0.25, at most 50 boxes, 11 categories). A second pass uses a 416 input, which is the plan's YOLOX letterbox size. YOLO11n's native size is 640.

Faces use OpenCV YuNet, not Apple Vision. Text, Live Photo, portrait, panorama, HDR, and burst stay 0 in this export. Screenshot, WhatsApp, aspect ratio, and an EXIF GPS flag are filled when the file or the catalog has them.

## Evaluation

Ten stratified 80/20 splits. Logistic regression and gradient-boosted trees match the planned settings. The MLPs match the planned widths, Adam settings, class-balanced loss, and early stopping. "Plan" runs pick the saved epoch on the test split, which is what the app would report. "Clean" runs pick it on an inner slice of the training split, then refit. The clean number is the one to trust.
