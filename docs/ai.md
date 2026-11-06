# AI Module

Fully on-device photo scoring from a 53-number feature vector. No network calls.

## Overview

```
Swipe keep/delete → SQLite labels → Extract features (cached) → Train the selected model → Pick active model → AI sort + score badge
```

## Features (53)

Photometric stats (16), YOLOX geometry + 13 category pairs (32; dog and cat have their own categories, other animals share one), Vision faces + text (6), screenshot / WhatsApp / aspect (3). See `ios/FreePhotoCleaner/FeatureSchema.swift` and `experiments/ai-models-changes/photo_features.py`.

YOLOX-Nano Core ML model: `PYTHONPATH=/path/to/YOLOX python scripts/export_yolox_nano.py --download-weights --verify` writes `ios/FreePhotoCleaner/Models/YOLOXNano.mlpackage` (~2 MB raw head `[1,3549,85]`). Xcode includes it in **Copy Bundle Resources**. On device, Swift decodes boxes + NMS (same math as YOLOX `decode_outputs` + `torchvision.ops.batched_nms`). If the model is missing or fails to load, the Train AI screen shows a warning and feature caches are not persisted for degraded runs; Vision + photometric features still run.

## Models

Train updates **only the selected model**, and only with photos that model has not seen yet. Each session fits on a stratified 80% of those photos and reports accuracy and F1 on the held-out 20%. Test photos are not marked learned, so a later session can train on them. Learned photos are a bit per photo (`trained.bits`), with the photo ids stored once in `asset_slots.bin`. Feature vectors are cached, so another model can train on the same photos without decoding them again. Delete a model on the Train screen to fit it from scratch. Settings → AI sort chooses which model fills `scoreById`.

| id | Description |
| --- | --- |
| `logistic-l2` | L2 logistic regression + Hessian memory (default) |
| `catboost` | Gradient boosted trees (depth 3, continued) |
| `gaussian-nb` | Gaussian naive Bayes partial fit |
| `mlp-small` | small MLP, 53→16→1 |
| `mlp-big` | big MLP, 53→128→64→32→16→8→1 |

## Storage

```
Documents/ai_feature_cache/v2/          # cached 53-vectors (version tracks `FeatureSchema.version`)
Documents/ai_feature_models/v2/         # one folder per model
Documents/ai_feature_models/v2/asset_slots.bin
Documents/ai_feature_models/v2/<id>/trained.bits
```

Legacy CNN checkpoints under `Documents/ai_models/` without `featureSchemaVersion` are removed on trainer startup.

## Training UI

Train AI: model picker. Gaussian naive Bayes has no extra knobs. Logistic regression exposes **C** (default 1) and **max iterations** (default 500). CatBoost exposes **learning rate** (0.1), **depth** (3), **min photos per leaf** (8), and **new trees per train** (40). MLPs also show **train epochs** (default 80), batch size, learning rate (default 0.001), scheduler, and a loss/accuracy graph. The graph's test curve is the same held-out 20%, and the MLP keeps the epoch with the lowest test loss. Every model shows that held-out accuracy and F1 for the last update.

A new model needs at least 20 labeled photos with both keeps and deletes. Later trains only add photos that model has not learned.

Reset AI Queue clears scores only. Reset trained model clears all five models and incorporated ids.
