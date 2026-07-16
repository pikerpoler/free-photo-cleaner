# AI Module

Fully on-device machine learning for photo delete/keep prediction. No network calls. No pretrained weights in the IPA.

## Overview

```
Swipe keep/delete → SQLite labels → Train AI screen → Native CNN trainer → Best checkpoint → AI sort scoring
```

- **Labels**: kept (`0`) and pending-delete (`1`) photos in SQLite. Unseen photos are never used for training.
- **Training**: manual via Train AI screen (epoch SGD, 80/20 split, live loss graph, Stop, keep best by test loss).
- **Models**: from-scratch MLP / CNN / ResNet-18 on resized RGB pixels (`trainResize` 32–512).
- **Scoring**: native `predict` using the active checkpoint for AI sort.

## Key files

| Path | Role |
|------|------|
| `src/screens/TrainAIScreen.tsx` | Train UI: model, resize, batch, LR, epochs, graph |
| `src/services/cnnTrainer.ts` | JS bridge to `CNNTrainerModule` |
| `src/ai/cnnTypes.ts` | Model IDs and training types |
| `src/stores/aiStore.ts` | Active model status + `scoreAssets` |
| `ios/.../CNNTrainerModule.swift` | Decode PHAsset → train loop → events |
| `ios/.../NeuralNetCore.swift` | Layers, CNN/MLP/ResNet-18, serialize |

## Models

- `mlp-tiny` / `mlp-small` / `mlp-medium` — flatten pixels → FC
- `cnn-nano` / `cnn-tiny` / `cnn-small` — small conv nets
- `resnet-18` — from-scratch ResNet-18 (no ImageNet weights)

## Persistence

- Labels: `kept_assets`, `pending_deletes`
- Active model: `DocumentDirectory/ai_models/active_weights.bin` + `active_meta.json`
- Library index: `asset_catalog` (speeds discovery / date filter)

Deletion remains manual via **Delete N**; training never deletes gallery photos.
