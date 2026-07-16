import {create} from 'zustand';
import {MediaAsset} from '../types/media';
import {ActiveModelInfo} from '../ai/cnnTypes';
import {
  getActiveModelInfo,
  hasActiveModel,
  predictScores,
  resetActiveModel,
} from '../services/cnnTrainer';

interface AIState {
  hasModel: boolean;
  activeModelInfo: ActiveModelInfo | null;
  isScoring: boolean;

  refreshModelStatus: () => Promise<void>;
  scoreAssets: (assets: MediaAsset[]) => Promise<Map<string, number>>;
  resetModel: () => Promise<void>;
}

export const useAIStore = create<AIState>((set, get) => ({
  hasModel: false,
  activeModelInfo: null,
  isScoring: false,

  refreshModelStatus: async () => {
    const exists = await hasActiveModel();
    const info = exists ? await getActiveModelInfo() : null;
    set({hasModel: exists, activeModelInfo: info});
  },

  scoreAssets: async (assets: MediaAsset[]) => {
    const scores = new Map<string, number>();
    if (assets.length === 0) return scores;

    const exists = await hasActiveModel();
    if (!exists) {
      for (const a of assets) scores.set(a.id, 0.5);
      return scores;
    }

    set({isScoring: true});
    try {
      // Score in chunks to avoid huge native payloads
      const CHUNK = 64;
      for (let i = 0; i < assets.length; i += CHUNK) {
        const chunk = assets.slice(i, i + CHUNK);
        const chunkScores = await predictScores(chunk.map(a => a.uri));
        chunk.forEach((asset, idx) => {
          scores.set(asset.id, chunkScores[idx] ?? 0.5);
        });
      }
    } finally {
      set({isScoring: false});
    }
    return scores;
  },

  resetModel: async () => {
    await resetActiveModel();
    set({hasModel: false, activeModelInfo: null});
  },
}));
