import {EMBEDDING_DIM} from './types';

export function isModelLoaded(): boolean {
  return true;
}

export async function loadModel(): Promise<void> {
  // No-op: using hash-based embeddings until ONNX model is bundled
}

export async function disposeModel(): Promise<void> {
  // No-op
}

/**
 * Compute embedding for an image URI.
 * Uses a deterministic hash-based embedding that produces a unique
 * EMBEDDING_DIM-length vector per URI. This allows the classifier
 * to learn correlations between images and user preferences.
 * Will be replaced by real neural embeddings when ONNX model is bundled.
 */
export async function computeEmbedding(imageUri: string): Promise<Float32Array> {
  return computeEmbeddingSync(imageUri);
}

export function computeEmbeddingSync(imageUri: string): Float32Array {
  const embedding = new Float32Array(EMBEDDING_DIM);

  let hash = 5381;
  for (let i = 0; i < imageUri.length; i++) {
    hash = ((hash << 5) + hash + imageUri.charCodeAt(i)) | 0;
  }

  for (let i = 0; i < EMBEDDING_DIM; i++) {
    hash = ((hash << 13) ^ hash) | 0;
    hash = (hash * 1664525 + 1013904223) | 0;
    embedding[i] = ((hash & 0xffff) / 32768.0) - 1.0;
  }

  return embedding;
}

export {EMBEDDING_DIM};
