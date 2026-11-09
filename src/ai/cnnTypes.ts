export type TrainableModelId =
  | 'logistic-l2'
  | 'catboost'
  | 'mlp-small'
  | 'mlp-big'
  | 'gaussian-nb';

export const TRAINABLE_MODELS: {id: TrainableModelId; label: string}[] = [
  {id: 'logistic-l2', label: 'Logistic Regression'},
  {id: 'catboost', label: 'CatBoost'},
  {id: 'gaussian-nb', label: 'Gaussian Naive Bayes'},
  {id: 'mlp-small', label: 'small MLP'},
  {id: 'mlp-big', label: 'big MLP'},
];

export const DEFAULT_TRAINABLE_MODEL: TrainableModelId = 'catboost';
export const MIN_LABELED_FOR_TRAIN = 20;
export const FEATURE_DIM = 57;
/** Bump in Swift `FeatureSchema.version` — JS can show retrain messaging. */
export const FEATURE_SCHEMA_VERSION = 3;

export type YOLOStatus = 'missing' | 'loaded' | 'failed';

export interface YOLODetectorStatus {
  status: YOLOStatus;
  loadError: string;
  lastRunError: string;
  featureSchemaVersion: number;
  expectedAnchors: number;
}

export function isMlpModel(id: string): boolean {
  return id === 'mlp-small' || id === 'mlp-big';
}

/** Used by the MLP trainers. Other models ignore these. */
export type LrSchedulerId = 'constant' | 'cosine' | 'step' | 'exponential';

export const LR_SCHEDULERS: {id: LrSchedulerId; label: string}[] = [
  {id: 'constant', label: 'Constant'},
  {id: 'cosine', label: 'Cosine'},
  {id: 'step', label: 'Step'},
  {id: 'exponential', label: 'Exponential'},
];

export interface AugmentationConfig {
  normalize: boolean;
  normalizeMean: [number, number, number];
  normalizeStd: [number, number, number];
  randomCrop: boolean;
  cropProbability: number;
  cropFraction: number;
  randomFlip: boolean;
  flipProbability: number;
  randomRotation: boolean;
  rotationProbability: number;
  rotationDegrees: number;
  gaussianNoise: boolean;
  noiseStd: number;
  colorJitter: boolean;
  jitterDelta: number;
  randomGrayscale: boolean;
  grayscaleProbability: number;
}

export const DEFAULT_AUGMENTATIONS: AugmentationConfig = {
  normalize: true,
  normalizeMean: [0.485, 0.456, 0.406],
  normalizeStd: [0.229, 0.224, 0.225],
  randomCrop: false,
  cropProbability: 0.5,
  cropFraction: 0.85,
  randomFlip: true,
  flipProbability: 0.5,
  randomRotation: false,
  rotationProbability: 0.3,
  rotationDegrees: 15,
  gaussianNoise: false,
  noiseStd: 0.02,
  colorJitter: false,
  jitterDelta: 0.1,
  randomGrayscale: false,
  grayscaleProbability: 0.1,
};

/** @deprecated CNN-era resize; feature models ignore this */
export const TRAIN_RESIZE_OPTIONS = [64] as const;
export type TrainResize = 64;
export const DEFAULT_TRAIN_RESIZE: TrainResize = 64;
export const TRAIN_TEST_RATIO = 1;

export interface TrainingConfig {
  /** Saved feature rows from `training_examples`. */
  features: number[][];
  labels: number[];
  /** @deprecated Training reads saved feature rows instead of photo URIs. */
  uris?: string[];
  modelId: TrainableModelId;
  batchSize?: number;
  learningRate?: number;
  maxEpochs?: number;
  scheduler?: LrSchedulerId;
  stepGamma?: number;
  stepEvery?: number;
  expDecay?: number;
  logisticC?: number;
  logisticMaxIter?: number;
  catboostLearningRate?: number;
  catboostDepth?: number;
  catboostMinLeaf?: number;
  catboostMaxNewTrees?: number;
}

export interface ModelStatus {
  modelId: string;
  hasModel: boolean;
  trainedAt: number;
  accuracy: number;
  f1: number;
  trainSize: number;
  /** Held-out photos for this update. 0 on checkpoints saved before the split. */
  testSize: number;
  learnedCount: number;
}

export interface TrainingProgress {
  epoch: number;
  trainLoss: number;
  testLoss: number;
  trainAcc: number;
  testAcc: number;
  bestTestLoss: number;
  bestTestAcc: number;
  trainSize: number;
  testSize: number;
  learningRate: number;
  done: boolean;
  phase?: 'features' | 'fit';
  index?: number;
  total?: number;
}

export interface TrainingResult {
  epochsRan: number;
  bestTestLoss: number | null;
  bestTestAcc: number | null;
  accuracy?: number | null;
  f1?: number | null;
  cancelled: boolean;
  trainSize: number;
  testSize: number;
  /** Labeled photos skipped because they could not be read. */
  skippedUnreadable?: number;
  modelId: string;
  trainResize: number;
}

export interface ModelCheckpointInfo {
  modelId: string;
  featureSchemaVersion?: number;
  label?: string;
  trainResize?: number;
  bestTestLoss?: number;
  bestTestAcc?: number;
  accuracy?: number;
  f1?: number;
  testSize?: number;
  trainSize?: number;
  learnedCount?: number;
  trainedAt: number;
  savedAt?: number;
}

export interface LabeledSample {
  id: string;
  uri: string;
  label: 0 | 1;
}

export function clampTrainResize(_n: number): TrainResize {
  return 64;
}
