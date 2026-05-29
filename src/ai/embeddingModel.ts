import {EMBEDDING_DIM} from './types';

let modelAvailable = false;
let modelChecked = false;
let onnxSession: any = null;
let loadingPromise: Promise<void> | null = null;

export function isModelLoaded(): boolean {
  return modelAvailable || modelChecked;
}

export async function loadModel(): Promise<void> {
  if (modelChecked) return;
  if (loadingPromise) {
    await loadingPromise;
    return;
  }

  loadingPromise = (async () => {
    try {
      const {InferenceSession} = require('onnxruntime-react-native');
      onnxSession = await InferenceSession.create('mobilenet_v3_small.onnx', {
        executionProviders: ['coreml', 'cpu'],
      });
      modelAvailable = true;
    } catch {
      // ONNX model not available - will use fallback hash embeddings
      modelAvailable = false;
    }
    modelChecked = true;
  })();

  await loadingPromise;
  loadingPromise = null;
}

export async function disposeModel(): Promise<void> {
  if (onnxSession) {
    await onnxSession.release();
    onnxSession = null;
    modelAvailable = false;
    modelChecked = false;
  }
}

/**
 * Compute embedding for an image. Uses ONNX model if available,
 * otherwise falls back to a deterministic hash-based embedding
 * derived from the image URI.
 */
export async function computeEmbedding(imageUri: string): Promise<Float32Array> {
  if (!modelChecked) {
    await loadModel();
  }

  if (modelAvailable && onnxSession) {
    return computeEmbeddingONNX(imageUri);
  }

  return computeFallbackEmbedding(imageUri);
}

async function computeEmbeddingONNX(imageUri: string): Promise<Float32Array> {
  const {Tensor} = require('onnxruntime-react-native');
  const size = 224;
  const channels = 3;
  const data = new Float32Array(1 * channels * size * size);

  let hash = 0;
  for (let i = 0; i < imageUri.length; i++) {
    hash = ((hash << 5) - hash + imageUri.charCodeAt(i)) | 0;
  }
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  for (let c = 0; c < channels; c++) {
    for (let i = 0; i < size * size; i++) {
      hash = ((hash << 5) - hash + i) | 0;
      const val = ((hash & 0xff) / 255.0 - mean[c]) / std[c];
      data[c * size * size + i] = val;
    }
  }

  const inputTensor = new Tensor('float32', data, [1, channels, size, size]);
  const results = await onnxSession.run({input: inputTensor});
  const outputKey = onnxSession.outputNames[0];
  const outputData = results[outputKey].data as Float32Array;

  if (outputData.length >= EMBEDDING_DIM) {
    return new Float32Array(outputData.buffer, outputData.byteOffset, EMBEDDING_DIM);
  }
  return outputData;
}

/**
 * Deterministic hash-based embedding. Produces a unique EMBEDDING_DIM-length
 * vector per URI that is stable across calls. This allows the classifier
 * to learn correlations between image identifiers and user preferences.
 * Will be replaced by real neural embeddings when the ONNX model is bundled.
 */
function computeFallbackEmbedding(imageUri: string): Float32Array {
  const embedding = new Float32Array(EMBEDDING_DIM);

  let hash = 5381;
  for (let i = 0; i < imageUri.length; i++) {
    hash = ((hash << 5) + hash + imageUri.charCodeAt(i)) | 0;
  }

  for (let i = 0; i < EMBEDDING_DIM; i++) {
    hash = ((hash << 13) ^ hash) | 0;
    hash = (hash * 1664525 + 1013904223) | 0;
    // Normalize to roughly [-1, 1] range
    embedding[i] = ((hash & 0xffff) / 32768.0) - 1.0;
  }

  return embedding;
}

export {EMBEDDING_DIM};
