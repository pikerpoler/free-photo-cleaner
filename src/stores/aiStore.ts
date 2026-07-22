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

/** Serialize scoreAssets so concurrent callers never re-predict the same ids. */
let scoreChain: Promise<unknown> = Promise.resolve();

interface AIState {
  hasModel: boolean;
  activeModelInfo: ModelCheckpointInfo | null;
  checkpoints: ModelCheckpointInfo[];
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

  scoreAssets: (assets: MediaAsset[]) => {
    const run = async (): Promise<Map<string, number>> => {
      const scores = new Map<string, number>();
      if (assets.length === 0) return scores;

      // Reuse cached scores — never re-predict the same asset.
      const need: MediaAsset[] = [];
      for (const a of assets) {
        const existing = get().scoreById[a.id];
        if (existing !== undefined) {
          scores.set(a.id, existing);
        } else {
          need.push(a);
        }
      }
      if (need.length === 0) return scores;

      const exists = await hasActiveModel();
      if (!exists) {
        const defaults: Record<string, number> = {};
        for (const a of need) {
          defaults[a.id] = 0.5;
          scores.set(a.id, 0.5);
        }
        set(state => ({scoreById: {...state.scoreById, ...defaults}}));
        return scores;
      }

      const CHUNK = 64;
      for (let i = 0; i < need.length; i += CHUNK) {
        const chunk = need.slice(i, i + CHUNK);
        const chunkScores = await predictScores(chunk.map(a => a.uri));
        const published: Record<string, number> = {};
        chunk.forEach((asset, idx) => {
          const value = chunkScores[idx] ?? 0.5;
          published[asset.id] = value;
          scores.set(asset.id, value);
        });
        set(state => ({
          scoreById: {...state.scoreById, ...published},
        }));
      }
      return scores;
    };

    const result = scoreChain.then(run, run);
    scoreChain = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
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
