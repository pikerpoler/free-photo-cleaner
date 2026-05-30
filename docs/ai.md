# AI Module

Fully on-device machine learning pipeline for photo delete/keep prediction. No network calls.

## Overview

```
Swipe → Embedding → Training → Weights → Scoring → Sort
```

- **Embedding**: 576-dim vector from asset URI (hash-based placeholder for MobileNet ONNX)
- **Classifier**: 2-layer MLP with ReLU hidden + sigmoid output
- **Training**: Online SGD with configurable batch size and learning rate
- **Scoring**: P(delete) per asset, used for AI sort mode

## Files

### types.ts (`src/ai/types.ts`)

```ts
EMBEDDING_DIM = 576;

type TrainingEntry = { assetId: string; label: number }; // 0=keep, 1=delete

type ModelWeights = {
  w1: number[][];   // hidden layer weights
  b1: number[];     // hidden layer biases
  w2: number[][];   // output layer weights
  b2: number[];     // output layer biases
};

type AIModelSize = 'small' | 'medium' | 'large';

MODEL_ARCHITECTURES = {
  small:  { hiddenSize: 64 },
  medium: { hiddenSize: 128 },
  large:  { hiddenSize: 256 },
};
```

### classifier.ts (`src/ai/classifier.ts`)

Pure math, no side effects. MLP with:
- Input: 576-dim embedding
- Hidden: configurable (64/128/256 neurons), ReLU activation
- Output: 1 neuron, sigmoid activation → P(delete)

| Export | Purpose |
|--------|---------|
| `createWeights(inputDim, hiddenSize)` | Xavier-initialized random weights |
| `forward(weights, input)` | Forward pass → {output, hidden} |
| `backward(weights, input, hidden, output, target)` | BCE loss gradients |
| `sgdStep(weights, gradients, lr)` | In-place weight update |
| `predict(weights, input)` | Forward pass → scalar P(delete) |

### embeddingModel.ts (`src/ai/embeddingModel.ts`)

Currently a **placeholder** using hash-based embeddings from the asset URI string. Designed to be replaced with MobileNet ONNX inference.

| Export | Purpose |
|--------|---------|
| `loadModel()` | No-op today (would load ONNX model) |
| `computeEmbedding(assetId)` | Hash URI → 576-dim float array |
| `isModelLoaded()` | Always true today |

### embeddingCache.ts (`src/ai/embeddingCache.ts`)

SQLite-backed cache for computed embeddings (avoids recomputation).

| Export | Purpose |
|--------|---------|
| `getCachedEmbedding(assetId)` | Read from `embedding_cache` table |
| `setCachedEmbedding(assetId, embedding)` | Write base64 Float32Array blob |
| `hasCachedEmbedding(assetId)` | Existence check |
| `clearEmbeddingCache()` | Truncate cache |

### trainer.ts (`src/ai/trainer.ts`)

Batch SGD trainer that runs via `InteractionManager.runAfterInteractions` to avoid blocking the UI.

| Export | Purpose |
|--------|---------|
| `configureTrainer(config)` | Set batch size, learning rate, model size |
| `enqueueTraining(entry)` | Add sample to queue |
| `flushTraining()` | Process all remaining samples |
| `flushPartialBatch()` | Process current partial batch |
| `getQueueSize()` | Number of pending training samples |

Training flow per batch:
1. For each sample: get embedding (cache or compute), forward pass, backward pass, accumulate gradients
2. After `batchSize` samples: average gradients, SGD step, save weights to disk

### weights.ts (`src/ai/weights.ts`)

Filesystem persistence for model weights using react-native-fs.

| Export | Purpose |
|--------|---------|
| `saveWeights(name, weights)` | Serialize to JSON, write to `DocumentDirectory/ai_models/` |
| `loadWeights(name)` | Read JSON file, deserialize |
| `deleteWeights(name)` | Remove weight file |
| `weightsExist(name)` | Check if file exists |

## AI Sort Flow

1. User selects "AI" sort in settings
2. `queueStore.loadQueue` or `applySort` calls `aiStore.scoreAssets(assets)`
3. `scoreAssets` computes embedding + `predict()` for each asset
4. Assets sorted by P(delete) descending (most likely to delete shown first)
5. Queue updated with AI-sorted order

## Future Work

`assets/models/README.md` describes planned MobileNet ONNX integration:
- Replace `computeEmbedding` with real image feature extraction
- Bundle `.onnx` file in `assets/models/`
- Use `onnxruntime-react-native` for inference
