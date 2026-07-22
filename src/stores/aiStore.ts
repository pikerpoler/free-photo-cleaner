import {create} from 'zustand';
import {MediaAsset} from '../types/media';
import {ModelCheckpointInfo, TrainableModelId} from '../ai/cnnTypes';
import {
  deleteModel,
  getActiveModelInfo,
  hasActiveModel,
  listModelCheckpoints,
  loadModel,
  predictScores,
  resetActiveModel,
} from '../services/cnnTrainer';

interface AIState {
  hasModel: boolean;
  activeModelInfo: ModelCheckpointInfo | null;
  checkpoints: ModelCheckpointInfo[];
  isScoring: boolean;
  scoreById: Record<string, number>;

  refreshModelStatus: () => Promise<void>;
  refreshCheckpoints: () => Promise<void>;
  scoreAssets: (assets: MediaAsset[]) => Promise<Map<string, number>>;
  clearScores: () => void;
  selectSortModel: (modelId: TrainableModelId) => Promise<void>;
  deleteCheckpoint: (modelId: string) => Promise<void>;
  resetModel: () => Promise<void>;
}

export const useAIStore = create<AIState>((set, get) => ({
  hasModel: false,
  activeModelInfo: null,
  checkpoints: [],
  isScoring: false,
  scoreById: {},

  refreshModelStatus: async () => {
    const exists = await hasActiveModel();
    const info = exists ? await getActiveModelInfo() : null;
    set({hasModel: exists, activeModelInfo: info});
  },

  refreshCheckpoints: async () => {
    const checkpoints = await listModelCheckpoints();
    set({checkpoints});
    await get().refreshModelStatus();
  },

  scoreAssets: async (assets: MediaAsset[]) => {
    const scores = new Map<string, number>();
    if (assets.length === 0) return scores;

    const exists = await hasActiveModel();
    if (!exists) {
      for (const a of assets) scores.set(a.id, 0.5);
      set({scoreById: Object.fromEntries(scores)});
      return scores;
    }

    set({isScoring: true});
    try {
      const CHUNK = 64;
      for (let i = 0; i < assets.length; i += CHUNK) {
        const chunk = assets.slice(i, i + CHUNK);
        const chunkScores = await predictScores(chunk.map(a => a.uri));
        chunk.forEach((asset, idx) => {
          scores.set(asset.id, chunkScores[idx] ?? 0.5);
        });
      }
      set(state => ({
        scoreById: {...state.scoreById, ...Object.fromEntries(scores)},
      }));
    } finally {
      set({isScoring: false});
    }
    return scores;
  },

  clearScores: () => set({scoreById: {}}),

  selectSortModel: async (modelId: TrainableModelId) => {
    await loadModel(modelId);
    get().clearScores();
    await get().refreshModelStatus();
  },

  deleteCheckpoint: async (modelId: string) => {
    await deleteModel(modelId);
    get().clearScores();
    await get().refreshCheckpoints();
  },

  resetModel: async () => {
    await resetActiveModel();
    set({hasModel: false, activeModelInfo: null, scoreById: {}, checkpoints: []});
  },
}));
