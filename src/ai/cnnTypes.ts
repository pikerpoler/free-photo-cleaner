export type TrainableModelId =
  | 'mlp-tiny'
  | 'mlp-small'
  | 'mlp-medium'
  | 'cnn-micro'
  | 'cnn-nano'
  | 'cnn-tiny'
  | 'cnn-small'
  | 'mobilenet-v3-head';

export const TRAINABLE_MODELS: {id: TrainableModelId; label: string}[] = [
  {id: 'mlp-tiny', label: 'MLP Tiny'},
  {id: 'mlp-small', label: 'MLP Small'},
  {id: 'mlp-medium', label: 'MLP Medium'},
  {id: 'cnn-micro', label: 'CNN Micro'},
  {id: 'cnn-nano', label: 'CNN Nano'},
  {id: 'cnn-tiny', label: 'CNN Tiny'},
  {id: 'cnn-small', label: 'CNN Small (heavy)'},
  {id: 'mobilenet-v3-head', label: 'MobileNetV3 Head'},
];

export const TRAIN_RESIZE_OPTIONS = [32, 64, 128, 256, 512] as const;
export type TrainResize = (typeof TRAIN_RESIZE_OPTIONS)[number];
export const DEFAULT_TRAIN_RESIZE: TrainResize = 64;
export const MIN_LABELED_FOR_TRAIN = 20;
export const TRAIN_TEST_RATIO = 0.8;

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

export interface TrainingConfig {
  uris: string[];
  labels: number[];
  modelId: TrainableModelId;
  batchSize: number;
  learningRate: number;
  epochs: number;
  trainResize: number;
  trainRatio?: number;
  lrScheduler?: LrSchedulerId;
  stepGamma?: number;
  stepSize?: number;
  expDecay?: number;
  augmentations?: AugmentationConfig;
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
}

export interface TrainingResult {
  epochsRan: number;
  bestTestLoss: number | null;
  bestTestAcc: number | null;
  cancelled: boolean;
  trainSize: number;
  testSize: number;
  modelId: string;
  trainResize: number;
}

export interface ModelCheckpointInfo {
  modelId: string;
  trainResize: number;
  bestTestLoss: number;
  bestTestAcc: number;
  testSize: number;
  trainSize?: number;
  trainedAt: number;
  savedAt?: number;
}

/** @deprecated use ModelCheckpointInfo */
export type ActiveModelInfo = ModelCheckpointInfo & {savedAt?: number};

export interface LabeledSample {
  id: string;
  uri: string;
  label: 0 | 1;
}

export function clampTrainResize(n: number): TrainResize {
  const allowed = TRAIN_RESIZE_OPTIONS as readonly number[];
  let best: TrainResize = DEFAULT_TRAIN_RESIZE;
  let bestDist = Infinity;
  for (const v of allowed) {
    const d = Math.abs(v - n);
    if (d < bestDist) {
      bestDist = d;
      best = v as TrainResize;
    }
  }
  return best;
}
