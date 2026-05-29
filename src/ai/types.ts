import {AIModelSize} from '../types/media';

export interface TrainingEntry {
  assetId: string;
  label: 0 | 1; // 0 = keep, 1 = delete
}

export interface ClassifierWeights {
  layers: LayerWeights[];
}

export interface LayerWeights {
  weight: Float32Array; // shape: [outFeatures, inFeatures]
  bias: Float32Array; // shape: [outFeatures]
  outFeatures: number;
  inFeatures: number;
}

export interface ClassifierGradients {
  layers: LayerGradients[];
  count: number;
}

export interface LayerGradients {
  weightGrad: Float32Array;
  biasGrad: Float32Array;
}

export type ModelStatus = 'uninitialized' | 'ready';

export const EMBEDDING_DIM = 576;

export const MODEL_ARCHITECTURES: Record<AIModelSize, number[]> = {
  tiny: [EMBEDDING_DIM, 1],
  small: [EMBEDDING_DIM, 32, 1],
  medium: [EMBEDDING_DIM, 128, 64, 1],
};
