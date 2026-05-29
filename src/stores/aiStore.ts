import {create} from 'zustand';
import {AIModelSize, MediaAsset} from '../types/media';
import {ClassifierWeights, ModelStatus} from '../ai/types';
import {createWeights, predict} from '../ai/classifier';
import {
  loadWeights,
  saveWeights,
  deleteWeights,
  weightsExist,
  initializeWeightsIfNeeded,
} from '../ai/weights';
import {
  loadModel as loadEmbeddingModel,
  isModelLoaded,
  computeEmbedding,
} from '../ai/embeddingModel';
import {getCachedEmbedding, setCachedEmbedding} from '../ai/embeddingCache';
import {
  configureTrainer,
  updateTrainerWeights,
  updateTrainerConfig,
  enqueueTraining,
  clearTrainingQueue,
  flushTraining,
  flushPartialBatch,
  getQueueSize,
} from '../ai/trainer';
import {useSettingsStore} from './settingsStore';

interface AIState {
  isEmbeddingModelLoaded: boolean;
  isLoadingEmbeddingModel: boolean;
  trainingQueueSize: number;
  modelStatus: Record<AIModelSize, ModelStatus>;
  weights: Record<AIModelSize, ClassifierWeights | null>;

  ensureEmbeddingModel: () => Promise<void>;
  ensureClassifier: (size: AIModelSize) => Promise<ClassifierWeights>;
  scoreAssets: (assets: MediaAsset[]) => Promise<Map<string, number>>;
  trainOnSwipe: (assetId: string, label: 0 | 1) => void;
  resetModel: (size: AIModelSize) => Promise<void>;
  flushAllTraining: () => Promise<void>;
  flushPartial: () => Promise<void>;
  getTrainingQueueSize: () => number;
  refreshModelStatuses: () => Promise<void>;
  syncTrainerConfig: () => void;
}

export const useAIStore = create<AIState>((set, get) => ({
  isEmbeddingModelLoaded: false,
  isLoadingEmbeddingModel: false,
  trainingQueueSize: 0,
  modelStatus: {
    tiny: 'uninitialized',
    small: 'uninitialized',
    medium: 'uninitialized',
  },
  weights: {
    tiny: null,
    small: null,
    medium: null,
  },

  ensureEmbeddingModel: async () => {
    if (get().isEmbeddingModelLoaded || get().isLoadingEmbeddingModel) return;
    set({isLoadingEmbeddingModel: true});
    try {
      await loadEmbeddingModel();
      set({isEmbeddingModelLoaded: true, isLoadingEmbeddingModel: false});
    } catch {
      set({isLoadingEmbeddingModel: false});
    }
  },

  ensureClassifier: async (size: AIModelSize) => {
    const current = get().weights[size];
    if (current) return current;

    // Try loading from disk
    let weights = await loadWeights(size);
    if (!weights) {
      // Initialize fresh random weights
      weights = createWeights(size);
      await saveWeights(size, weights);
    }

    set(state => ({
      weights: {...state.weights, [size]: weights},
      modelStatus: {...state.modelStatus, [size]: 'ready' as ModelStatus},
    }));

    return weights;
  },

  scoreAssets: async (assets: MediaAsset[]) => {
    const settings = useSettingsStore.getState();
    const size = settings.aiModel;
    const scores = new Map<string, number>();

    // Ensure the classifier is ready
    const weights = await get().ensureClassifier(size);

    // Ensure embedding model is loaded
    await get().ensureEmbeddingModel();

    for (const asset of assets) {
      let embedding = getCachedEmbedding(asset.id);
      if (!embedding) {
        try {
          embedding = await computeEmbedding(asset.uri);
          setCachedEmbedding(asset.id, embedding);
        } catch {
          // Assign neutral score if embedding fails
          scores.set(asset.id, 0.5);
          continue;
        }
      }
      const score = predict(weights, embedding);
      scores.set(asset.id, score);
    }

    return scores;
  },

  trainOnSwipe: (assetId: string, label: 0 | 1) => {
    const settings = useSettingsStore.getState();
    if (!settings.trainAI) return;

    const size = settings.aiModel;
    const weights = get().weights[size];

    // Lazy init: if weights don't exist yet, create them now
    if (!weights) {
      const newWeights = createWeights(size);
      set(state => ({
        weights: {...state.weights, [size]: newWeights},
        modelStatus: {...state.modelStatus, [size]: 'ready' as ModelStatus},
      }));

      // Configure trainer with new weights
      configureTrainer({
        weights: newWeights,
        modelSize: size,
        batch: settings.aiBatchSize,
        lr: settings.aiStepSize,
        onChange: () => {
          set({trainingQueueSize: getQueueSize()});
        },
      });

      // Also persist the weights
      saveWeights(size, newWeights);
    } else {
      // Ensure trainer is configured with current weights
      configureTrainer({
        weights,
        modelSize: size,
        batch: settings.aiBatchSize,
        lr: settings.aiStepSize,
        onChange: () => {
          set({trainingQueueSize: getQueueSize()});
        },
      });
    }

    // Ensure embedding model will be loaded (non-blocking)
    get().ensureEmbeddingModel();

    enqueueTraining({assetId, label});
    set({trainingQueueSize: getQueueSize()});
  },

  resetModel: async (size: AIModelSize) => {
    await deleteWeights(size);
    set(state => ({
      weights: {...state.weights, [size]: null},
      modelStatus: {...state.modelStatus, [size]: 'uninitialized' as ModelStatus},
    }));
  },

  flushAllTraining: async () => {
    await flushTraining();
    set({trainingQueueSize: 0});
  },

  flushPartial: async () => {
    await flushPartialBatch();
    clearTrainingQueue();
    set({trainingQueueSize: 0});
  },

  getTrainingQueueSize: () => {
    return getQueueSize();
  },

  refreshModelStatuses: async () => {
    const statuses: Record<AIModelSize, ModelStatus> = {
      tiny: (await weightsExist('tiny')) ? 'ready' : 'uninitialized',
      small: (await weightsExist('small')) ? 'ready' : 'uninitialized',
      medium: (await weightsExist('medium')) ? 'ready' : 'uninitialized',
    };
    set({modelStatus: statuses});
  },

  syncTrainerConfig: () => {
    const settings = useSettingsStore.getState();
    updateTrainerConfig(settings.aiBatchSize, settings.aiStepSize);
    const weights = get().weights[settings.aiModel];
    if (weights) {
      updateTrainerWeights(weights);
    }
  },
}));
