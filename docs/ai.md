# AI Module

Fully on-device machine learning for photo delete/keep prediction. No network calls.

## Overview

```
Swipe keep/delete → SQLite labels → Train AI → Native trainer → Per-model checkpoint → AI sort + score badge
```

## Models

Defined in `ios/FreePhotoCleaner/NeuralNetCore.swift` (`ModelFactory.create`):

- Scratch: `mlp-*`, `cnn-micro`, `cnn-nano`, `cnn-tiny`, `cnn-small`
- `mobilenet-v3-head`: frozen feature backbone + trainable linear head
- ResNet-18 removed

## Training UX

- Accuracy + loss graphs, power-of-2 resize (32–512)
- Augmentations (normalize, crop, flip, rotate, noise, jitter, grayscale)
- LR schedulers: constant, cosine, step, exponential
- Best checkpoint by lowest test loss (stores test acc too)

## Checkpoints

```
Documents/ai_models/{modelId}/weights.bin
Documents/ai_models/{modelId}/meta.json
Documents/ai_models/active_model_id.txt
```

Settings → Photos Sort = AI shows checkpoint list (loss, acc, test size, date) with select/delete. Switching model or **Reset AI Queue** clears scores and rebuilds the swipe queue without touching keep/delete labels.
