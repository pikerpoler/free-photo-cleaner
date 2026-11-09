# AI Module

Fully on-device photo scoring from a 53-number feature vector. No network calls.

## Overview

```
Swipe keep/delete → Extract features → Save vector + label in SQLite → Train a fresh model on the full dataset → Pick active model → AI sort + score badge
```

## Features (53)

Photometric stats (16), YOLOX geometry + 13 category pairs (32; dog and cat have their own categories, other animals share one), Vision faces + text (6), screenshot / WhatsApp / aspect (3). See `ios/FreePhotoCleaner/FeatureSchema.swift` and `experiments/ai-models-changes/photo_features.py`.

YOLOX-Nano Core ML model: `PYTHONPATH=/path/to/YOLOX python scripts/export_yolox_nano.py --download-weights --verify` writes `ios/FreePhotoCleaner/Models/YOLOXNano.mlpackage` (~2 MB raw head `[1,3549,85]`). Xcode includes it in **Copy Bundle Resources**. On device, Swift decodes boxes + NMS (same math as YOLOX `decode_outputs` + `torchvision.ops.batched_nms`). If the model is missing or fails to load, the Train AI screen shows a warning and feature caches are not persisted for degraded runs; Vision + photometric features still run.

## Models

Each photo swipe stores a feature vector and keep/delete label in SQLite (`training_examples`). Those rows stay after the photo is deleted. Train rebuilds **only the selected model** from every saved row: a blank scorer fits a stratified 80% and reports accuracy and F1 on the held-out 20%. The live weights stay until that run finishes, then the new file replaces them. Stopping a run discards the replacement. Settings → AI sort chooses which model fills `scoreById`; Settings can still delete a checkpoint. The Train screen deletes the saved dataset, not the model.

| id | Description |
| --- | --- |
| `logistic-l2` | L2 logistic regression + Hessian memory (default) |
| `catboost` | Gradient boosted trees (depth 3, continued) |
| `gaussian-nb` | Gaussian naive Bayes partial fit |
| `mlp-small` | small MLP, 53→16→1 |
| `mlp-big` | big MLP, 53→128→64→32→16→8→1 |

## Storage

```
aiphotocleaner.db / training_examples   # feature vector + label per swipe (survives photo deletion)
Documents/ai_feature_cache/v3/          # temporary decode cache (version tracks FeatureSchema.version)
Documents/ai_feature_models/v3/         # one folder per model
```

Legacy CNN checkpoints under `Documents/ai_models/` without `featureSchemaVersion` are removed on trainer startup.

## Training UI

Train AI: model picker. Gaussian naive Bayes has no extra knobs. Logistic regression exposes **C** (default 1) and **max iterations** (default 500). CatBoost exposes **learning rate** (0.1), **depth** (3), **min photos per leaf** (8), and **new trees per train** (40). MLPs also show **train epochs** (default 80), batch size, learning rate (default 0.001), scheduler, and a loss/accuracy graph. The graph's test curve is the same held-out 20%, and the MLP keeps the epoch with the lowest test loss. Every model shows that held-out accuracy and F1 for the last update.

A train needs at least 20 saved photos with both keeps and deletes. The screen shows how much storage the saved vectors use. Delete dataset clears `training_examples` only.

Reset AI Queue clears scores only. Reset trained model clears all five models.
