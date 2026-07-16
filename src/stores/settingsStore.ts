import {create} from 'zustand';
import {
  DateFilter,
  PhotoFilters,
  SortMode,
  VideoFilters,
} from '../types/media';
import {TrainableModelId} from '../ai/cnnTypes';
import {
  DEFAULT_TRAIN_RESIZE,
  MAX_TRAIN_RESIZE,
  MIN_TRAIN_RESIZE,
} from '../ai/cnnTypes';
import {getSetting, setSetting, KEYS} from '../services/storage';

function migrateModelId(raw: string): TrainableModelId {
  const map: Record<string, TrainableModelId> = {
    tiny: 'mlp-tiny',
    small: 'mlp-small',
    medium: 'mlp-medium',
  };
  if (map[raw]) return map[raw];
  const valid: TrainableModelId[] = [
    'mlp-tiny',
    'mlp-small',
    'mlp-medium',
    'cnn-nano',
    'cnn-tiny',
    'cnn-small',
    'resnet-18',
  ];
  return valid.includes(raw as TrainableModelId)
    ? (raw as TrainableModelId)
    : 'cnn-nano';
}

interface SettingsState {
  photoSort: SortMode;
  videoSort: SortMode;
  photoFilters: PhotoFilters;
  videoFilters: VideoFilters;
  dateFilter: DateFilter;
  theme: 'system' | 'light' | 'dark';
  aiModel: TrainableModelId;
  aiBatchSize: number;
  aiStepSize: number;
  aiEpochs: number;
  aiTrainResize: number;

  setPhotoSort: (sort: SortMode) => void;
  setVideoSort: (sort: SortMode) => void;
  setPhotoFilters: (filters: Partial<PhotoFilters>) => void;
  setVideoFilters: (filters: Partial<VideoFilters>) => void;
  setDateFilter: (filter: Partial<DateFilter>) => void;
  setTheme: (theme: 'system' | 'light' | 'dark') => void;
  setAIModel: (model: TrainableModelId | string) => void;
  setAIBatchSize: (size: number) => void;
  setAIStepSize: (rate: number) => void;
  setAIEpochs: (epochs: number) => void;
  setAITrainResize: (resize: number) => void;
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
  aiModel: 'cnn-nano',
  aiBatchSize: 8,
  aiStepSize: 0.01,
  aiEpochs: 20,
  aiTrainResize: DEFAULT_TRAIN_RESIZE,

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

  setAIModel: (model: TrainableModelId | string) => {
    const migrated = migrateModelId(String(model));
    set({aiModel: migrated});
    setSetting(KEYS.AI_MODEL, migrated);
  },

  setAIBatchSize: (size: number) => {
    set({aiBatchSize: size});
    setSetting(KEYS.AI_BATCH_SIZE, size);
  },

  setAIStepSize: (rate: number) => {
    set({aiStepSize: rate});
    setSetting(KEYS.AI_STEP_SIZE, rate);
  },

  setAIEpochs: (epochs: number) => {
    set({aiEpochs: epochs});
    setSetting(KEYS.AI_EPOCHS, epochs);
  },

  setAITrainResize: (resize: number) => {
    const clamped = Math.min(
      MAX_TRAIN_RESIZE,
      Math.max(MIN_TRAIN_RESIZE, Math.round(resize)),
    );
    set({aiTrainResize: clamped});
    setSetting(KEYS.AI_TRAIN_RESIZE, clamped);
  },

  loadSettings: () => {
    const rawModel = getSetting(KEYS.AI_MODEL, 'cnn-nano' as string);
    set({
      photoSort: getSetting(KEYS.PHOTO_SORT, 'newest_first' as SortMode),
      videoSort: getSetting(KEYS.VIDEO_SORT, 'largest_first' as SortMode),
      photoFilters: getSetting(KEYS.PHOTO_FILTERS, DEFAULT_PHOTO_FILTERS),
      videoFilters: getSetting(KEYS.VIDEO_FILTERS, DEFAULT_VIDEO_FILTERS),
      dateFilter: getSetting(KEYS.DATE_FILTER, getDefaultDateFilter()),
      theme: getSetting(KEYS.THEME, 'system' as const),
      aiModel: migrateModelId(rawModel),
      aiBatchSize: getSetting(KEYS.AI_BATCH_SIZE, 8),
      aiStepSize: getSetting(KEYS.AI_STEP_SIZE, 0.01),
      aiEpochs: getSetting(KEYS.AI_EPOCHS, 20),
      aiTrainResize: getSetting(KEYS.AI_TRAIN_RESIZE, DEFAULT_TRAIN_RESIZE),
    });
  },
}));
