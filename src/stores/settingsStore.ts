import {create} from 'zustand';
import {
  DateFilter,
  PhotoFilters,
  SortMode,
  VideoFilters,
} from '../types/media';
import {
  AugmentationConfig,
  clampTrainResize,
  DEFAULT_AUGMENTATIONS,
  DEFAULT_TRAIN_RESIZE,
  LrSchedulerId,
  TrainableModelId,
  TRAINABLE_MODELS,
} from '../ai/cnnTypes';
import {getSetting, setSetting, KEYS} from '../services/storage';

function migrateModelId(raw: string): TrainableModelId {
  const map: Record<string, TrainableModelId> = {
    tiny: 'mlp-tiny',
    small: 'mlp-small',
    medium: 'mlp-medium',
    'resnet-18': 'cnn-nano',
  };
  if (map[raw]) return map[raw];
  const valid = TRAINABLE_MODELS.map(m => m.id);
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
  aiSortModel: TrainableModelId | null;
  aiBatchSize: number;
  aiStepSize: number;
  aiEpochs: number;
  aiTrainResize: number;
  aiLrScheduler: LrSchedulerId;
  aiStepGamma: number;
  aiStepSizeEpochs: number;
  aiExpDecay: number;
  aiAugmentations: AugmentationConfig;

  setPhotoSort: (sort: SortMode) => void;
  setVideoSort: (sort: SortMode) => void;
  setPhotoFilters: (filters: Partial<PhotoFilters>) => void;
  setVideoFilters: (filters: Partial<VideoFilters>) => void;
  setDateFilter: (filter: Partial<DateFilter>) => void;
  setTheme: (theme: 'system' | 'light' | 'dark') => void;
  setAIModel: (model: TrainableModelId | string) => void;
  setAISortModel: (model: TrainableModelId | null) => void;
  setAIBatchSize: (size: number) => void;
  setAIStepSize: (rate: number) => void;
  setAIEpochs: (epochs: number) => void;
  setAITrainResize: (resize: number) => void;
  setAILrScheduler: (s: LrSchedulerId) => void;
  setAIStepGamma: (v: number) => void;
  setAIStepSizeEpochs: (v: number) => void;
  setAIExpDecay: (v: number) => void;
  setAIAugmentations: (partial: Partial<AugmentationConfig>) => void;
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
  aiSortModel: null,
  aiBatchSize: 8,
  aiStepSize: 0.01,
  aiEpochs: 20,
  aiTrainResize: DEFAULT_TRAIN_RESIZE,
  aiLrScheduler: 'constant',
  aiStepGamma: 0.5,
  aiStepSizeEpochs: 5,
  aiExpDecay: 0.95,
  aiAugmentations: DEFAULT_AUGMENTATIONS,

  setPhotoSort: (sort: SortMode) => {
    set({photoSort: sort});
    setSetting(KEYS.PHOTO_SORT, sort);
  },

  setVideoSort: (sort: SortMode) => {
    set({videoSort: sort});
    setSetting(KEYS.VIDEO_SORT, sort);
  },

  setPhotoFilters: (filters: Partial<PhotoFilters>) => {
    const updated = {...get().photoFilters, ...filters};
    set({photoFilters: updated});
    setSetting(KEYS.PHOTO_FILTERS, updated);
  },

  setVideoFilters: (filters: Partial<VideoFilters>) => {
    const updated = {...get().videoFilters, ...filters};
    set({videoFilters: updated});
    setSetting(KEYS.VIDEO_FILTERS, updated);
  },

  setDateFilter: (filter: Partial<DateFilter>) => {
    const updated = {...get().dateFilter, ...filter};
    if (updated.from > updated.to) updated.to = updated.from;
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

  setAISortModel: (model: TrainableModelId | null) => {
    set({aiSortModel: model});
    setSetting(KEYS.AI_SORT_MODEL, model);
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
    const clamped = clampTrainResize(resize);
    set({aiTrainResize: clamped});
    setSetting(KEYS.AI_TRAIN_RESIZE, clamped);
  },

  setAILrScheduler: (s: LrSchedulerId) => {
    set({aiLrScheduler: s});
    setSetting(KEYS.AI_LR_SCHEDULER, s);
  },

  setAIStepGamma: (v: number) => {
    set({aiStepGamma: v});
    setSetting(KEYS.AI_STEP_GAMMA, v);
  },

  setAIStepSizeEpochs: (v: number) => {
    set({aiStepSizeEpochs: v});
    setSetting(KEYS.AI_STEP_SIZE_EPOCHS, v);
  },

  setAIExpDecay: (v: number) => {
    set({aiExpDecay: v});
    setSetting(KEYS.AI_EXP_DECAY, v);
  },

  setAIAugmentations: (partial: Partial<AugmentationConfig>) => {
    const updated = {...get().aiAugmentations, ...partial};
    if (updated.noiseStd > 0.05) updated.noiseStd = 0.05;
    set({aiAugmentations: updated});
    setSetting(KEYS.AI_AUGMENTATIONS, updated);
  },

  loadSettings: () => {
    const rawModel = getSetting(KEYS.AI_MODEL, 'cnn-nano' as string);
    const rawSort = getSetting<string | null>(KEYS.AI_SORT_MODEL, null);
    set({
      photoSort: getSetting(KEYS.PHOTO_SORT, 'newest_first' as SortMode),
      videoSort: getSetting(KEYS.VIDEO_SORT, 'largest_first' as SortMode),
      photoFilters: getSetting(KEYS.PHOTO_FILTERS, DEFAULT_PHOTO_FILTERS),
      videoFilters: getSetting(KEYS.VIDEO_FILTERS, DEFAULT_VIDEO_FILTERS),
      dateFilter: getSetting(KEYS.DATE_FILTER, getDefaultDateFilter()),
      theme: getSetting(KEYS.THEME, 'system' as const),
      aiModel: migrateModelId(rawModel),
      aiSortModel: rawSort ? migrateModelId(rawSort) : null,
      aiBatchSize: getSetting(KEYS.AI_BATCH_SIZE, 8),
      aiStepSize: getSetting(KEYS.AI_STEP_SIZE, 0.01),
      aiEpochs: getSetting(KEYS.AI_EPOCHS, 20),
      aiTrainResize: clampTrainResize(
        getSetting(KEYS.AI_TRAIN_RESIZE, DEFAULT_TRAIN_RESIZE),
      ),
      aiLrScheduler: getSetting(KEYS.AI_LR_SCHEDULER, 'constant' as LrSchedulerId),
      aiStepGamma: getSetting(KEYS.AI_STEP_GAMMA, 0.5),
      aiStepSizeEpochs: getSetting(KEYS.AI_STEP_SIZE_EPOCHS, 5),
      aiExpDecay: getSetting(KEYS.AI_EXP_DECAY, 0.95),
      aiAugmentations: {
        ...DEFAULT_AUGMENTATIONS,
        ...getSetting(KEYS.AI_AUGMENTATIONS, DEFAULT_AUGMENTATIONS),
      },
    });
  },
}));
