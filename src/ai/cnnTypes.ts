export type TrainableModelId =
  | 'mlp-tiny'
  | 'mlp-small'
  | 'mlp-medium'
  | 'cnn-nano'
  | 'cnn-tiny'
  | 'cnn-small'
  | 'resnet-18';

export const TRAINABLE_MODELS: {id: TrainableModelId; label: string}[] = [
  {id: 'mlp-tiny', label: 'MLP Tiny'},
  {id: 'mlp-small', label: 'MLP Small'},
  {id: 'mlp-medium', label: 'MLP Medium'},
  {id: 'cnn-nano', label: 'CNN Nano'},
  {id: 'cnn-tiny', label: 'CNN Tiny'},
  {id: 'cnn-small', label: 'CNN Small'},
  {id: 'resnet-18', label: 'ResNet-18'},
];

export const MIN_TRAIN_RESIZE = 32;
export const MAX_TRAIN_RESIZE = 512;
export const DEFAULT_TRAIN_RESIZE = 64;
export const MIN_LABELED_FOR_TRAIN = 20;
export const TRAIN_TEST_RATIO = 0.8;

export interface TrainingConfig {
  uris: string[];
  labels: number[];
  modelId: TrainableModelId;
  batchSize: number;
  learningRate: number;
  epochs: number;
  trainResize: number;
  trainRatio?: number;
}

export interface TrainingProgress {
  epoch: number;
  trainLoss: number;
  testLoss: number;
  bestTestLoss: number;
  trainSize: number;
  testSize: number;
  done: boolean;
}

export interface TrainingResult {
  epochsRan: number;
  bestTestLoss: number | null;
  cancelled: boolean;
  trainSize: number;
  testSize: number;
  modelId: string;
  trainResize: number;
}

export interface ActiveModelInfo {
  modelId: string;
  trainResize: number;
  bestTestLoss: number;
  savedAt: number;
}

export interface LabeledSample {
  id: string;
  uri: string;
  label: 0 | 1;
}
