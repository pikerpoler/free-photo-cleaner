# AI Module

Fully on-device photo scoring from a 53-number feature vector. No network calls.

## Overview

```
Swipe keep/delete → SQLite labels → Extract features (cached) → Train the selected model → Pick active model → AI sort + score badge
```

## Features (53)

Photometric stats (16), YOLOX geometry + 11 category pairs (28), Vision faces + text (6), screenshot / WhatsApp / aspect (3). See `ios/FreePhotoCleaner/FeatureSchema.swift` and `experiments/ai-models-changes/photo_features.py`.

YOLOX-Nano Core ML model: run `python scripts/export_yolox_nano.py` and add `YOLOXNano.mlpackage` to the app bundle. Without it, YOLO slots stay zero; Vision and photometric features still run.

## Models

Train updates **only the selected model**, and only with photos that model has not seen yet. Those marks are a bit per photo (`trained.bits`), with the photo ids stored once in `asset_slots.bin`. Feature vectors are cached, so another model can train on the same photos without decoding them again. Delete a model on the Train screen to fit it from scratch. Settings → AI sort chooses which model fills `scoreById`.

| id | Description |
| --- | --- |
| `logistic-l2` | L2 logistic regression + Hessian memory (default) |
| `catboost` | Gradient boosted trees (depth 3, continued) |
| `gaussian-nb` | Gaussian naive Bayes partial fit |
| `mlp-small` | small MLP, 53→16→1 |
| `mlp-big` | big MLP, 53→128→64→32→16→8→1 |

## Storage

```
Documents/ai_feature_cache/v1/          # cached 53-vectors
Documents/ai_feature_models/v1/         # one folder per model
Documents/ai_feature_models/v1/asset_slots.bin
Documents/ai_feature_models/v1/<id>/trained.bits
```

Legacy CNN checkpoints under `Documents/ai_models/` without `featureSchemaVersion` are removed on trainer startup.

## Training UI

Train AI: model picker. Gaussian naive Bayes has no extra knobs. Logistic regression exposes **C** (default 1) and **max iterations** (default 500). CatBoost exposes **learning rate** (0.1), **depth** (3), **min photos per leaf** (8), and **new trees per train** (40). MLPs also show **train epochs** (default 80), batch size, learning rate (default 0.001), scheduler, and a loss/accuracy graph. Logistic, CatBoost, and naive Bayes show accuracy and F1 for the last update.

A new model needs at least 20 labeled photos with both keeps and deletes. Later trains only add photos that model has not learned.

Reset AI Queue clears scores only. Reset trained model clears all five models and incorporated ids.
