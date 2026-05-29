import {create} from 'zustand';
import {
  AIModelSize,
  DateFilter,
  PhotoFilters,
  SortMode,
  VideoFilters,
} from '../types/media';
import {getSetting, setSetting, KEYS} from '../services/storage';

interface SettingsState {
  photoSort: SortMode;
  videoSort: SortMode;
  photoFilters: PhotoFilters;
  videoFilters: VideoFilters;
  dateFilter: DateFilter;
  theme: 'system' | 'light' | 'dark';
  trainAI: boolean;
  aiModel: AIModelSize;
  aiBatchSize: number;
  aiStepSize: number;

  setPhotoSort: (sort: SortMode) => void;
  setVideoSort: (sort: SortMode) => void;
  setPhotoFilters: (filters: Partial<PhotoFilters>) => void;
  setVideoFilters: (filters: Partial<VideoFilters>) => void;
  setDateFilter: (filter: Partial<DateFilter>) => void;
  setTheme: (theme: 'system' | 'light' | 'dark') => void;
  setTrainAI: (enabled: boolean) => void;
  setAIModel: (model: AIModelSize) => void;
  setAIBatchSize: (size: number) => void;
  setAIStepSize: (rate: number) => void;
  loadSettings: () => void;
}

const DEFAULT_PHOTO_FILTERS: PhotoFilters = {
  mode: 'all',
  screenshots: true,
  whatsapp: true,
  noMetadata: true,
};

const DEFAULT_VIDEO_FILTERS: VideoFilters = {
  minDuration: 0,
  maxDuration: Infinity,
  minSize: 0,
  maxSize: Infinity,
};

function getDefaultDateFilter(): DateFilter {
  const now = Date.now();
  const thirtyDaysAgo = now - 30 * 24 * 60 * 60 * 1000;
  return {enabled: false, from: thirtyDaysAgo, to: now};
}

export const useSettingsStore = create<SettingsState>((set, get) => ({
  photoSort: 'newest_first',
  videoSort: 'largest_first',
  photoFilters: DEFAULT_PHOTO_FILTERS,
  videoFilters: DEFAULT_VIDEO_FILTERS,
  dateFilter: getDefaultDateFilter(),
  theme: 'system',
  trainAI: true,
  aiModel: 'tiny',
  aiBatchSize: 10,
  aiStepSize: 0.01,

  setPhotoSort: (sort: SortMode) => {
    set({photoSort: sort});
    setSetting(KEYS.PHOTO_SORT, sort);
  },

  setVideoSort: (sort: SortMode) => {
    set({videoSort: sort});
    setSetting(KEYS.VIDEO_SORT, sort);
  },

  setPhotoFilters: (filters: Partial<PhotoFilters>) => {
    const current = get().photoFilters;
    const updated = {...current, ...filters};
    set({photoFilters: updated});
    setSetting(KEYS.PHOTO_FILTERS, updated);
  },

  setVideoFilters: (filters: Partial<VideoFilters>) => {
    const current = get().videoFilters;
    const updated = {...current, ...filters};
    set({videoFilters: updated});
    setSetting(KEYS.VIDEO_FILTERS, updated);
  },

  setDateFilter: (filter: Partial<DateFilter>) => {
    const current = get().dateFilter;
    const updated = {...current, ...filter};
    if (updated.from > updated.to) {
      updated.to = updated.from;
    }
    set({dateFilter: updated});
    setSetting(KEYS.DATE_FILTER, updated);
  },

  setTheme: (theme: 'system' | 'light' | 'dark') => {
    set({theme});
    setSetting(KEYS.THEME, theme);
  },

  setTrainAI: (enabled: boolean) => {
    set({trainAI: enabled});
    setSetting(KEYS.TRAIN_AI, enabled);
  },

  setAIModel: (model: AIModelSize) => {
    set({aiModel: model});
    setSetting(KEYS.AI_MODEL, model);
  },

  setAIBatchSize: (size: number) => {
    set({aiBatchSize: size});
    setSetting(KEYS.AI_BATCH_SIZE, size);
  },

  setAIStepSize: (rate: number) => {
    set({aiStepSize: rate});
    setSetting(KEYS.AI_STEP_SIZE, rate);
  },

  loadSettings: () => {
    set({
      photoSort: getSetting(KEYS.PHOTO_SORT, 'newest_first' as SortMode),
      videoSort: getSetting(KEYS.VIDEO_SORT, 'largest_first' as SortMode),
      photoFilters: getSetting(KEYS.PHOTO_FILTERS, DEFAULT_PHOTO_FILTERS),
      videoFilters: getSetting(KEYS.VIDEO_FILTERS, DEFAULT_VIDEO_FILTERS),
      dateFilter: getSetting(KEYS.DATE_FILTER, getDefaultDateFilter()),
      theme: getSetting(KEYS.THEME, 'system' as const),
      trainAI: getSetting(KEYS.TRAIN_AI, true),
      aiModel: getSetting(KEYS.AI_MODEL, 'tiny' as AIModelSize),
      aiBatchSize: getSetting(KEYS.AI_BATCH_SIZE, 10),
      aiStepSize: getSetting(KEYS.AI_STEP_SIZE, 0.01),
    });
  },
}));
