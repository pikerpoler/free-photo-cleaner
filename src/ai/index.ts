export {EMBEDDING_DIM, MODEL_ARCHITECTURES} from './types';
export type {
  TrainingEntry,
  ClassifierWeights,
  ClassifierGradients,
  LayerWeights,
  LayerGradients,
  ModelStatus,
} from './types';
export {
  createWeights,
  createGradients,
  zeroGradients,
  forward,
  backward,
  sgdStep,
  predict,
} from './classifier';
export {
  loadModel,
  disposeModel,
  computeEmbedding,
  isModelLoaded,
} from './embeddingModel';
export {
  getCachedEmbedding,
  setCachedEmbedding,
  hasCachedEmbedding,
  clearEmbeddingCache,
} from './embeddingCache';
export {
  saveWeights,
  loadWeights,
  deleteWeights,
  weightsExist,
  serializeWeights,
  deserializeWeights,
} from './weights';
export {
  configureTrainer,
  enqueueTraining,
  clearTrainingQueue,
  flushTraining,
  flushPartialBatch,
  getQueueSize,
  getTrainerState,
} from './trainer';
