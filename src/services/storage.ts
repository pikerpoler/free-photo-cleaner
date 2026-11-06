import {createMMKV, type MMKV} from 'react-native-mmkv';

export const storage: MMKV = createMMKV({id: 'aiphotocleaner'});

const KEYS = {
  PHOTO_SORT: 'settings.photo.sort',
  VIDEO_SORT: 'settings.video.sort',
  PHOTO_FILTERS: 'settings.photo.filters',
  VIDEO_FILTERS: 'settings.video.filters',
  DATE_FILTER: 'settings.date.filter',
  THEME: 'settings.theme',
  AI_MODEL: 'settings.ai.model',
  AI_BATCH_SIZE: 'settings.ai.batchSize',
  AI_STEP_SIZE: 'settings.ai.stepSize',
  AI_EPOCHS: 'settings.ai.epochs',
  AI_TRAIN_RESIZE: 'settings.ai.trainResize',
  AI_LR_SCHEDULER: 'settings.ai.lrScheduler',
  AI_STEP_GAMMA: 'settings.ai.stepGamma',
  AI_STEP_SIZE_EPOCHS: 'settings.ai.stepSizeEpochs',
  AI_EXP_DECAY: 'settings.ai.expDecay',
  AI_LOGISTIC_C: 'settings.ai.logisticC',
  AI_LOGISTIC_MAX_ITER: 'settings.ai.logisticMaxIter',
  AI_CATBOOST_LR: 'settings.ai.catboostLearningRate',
  AI_CATBOOST_DEPTH: 'settings.ai.catboostDepth',
  AI_CATBOOST_MIN_LEAF: 'settings.ai.catboostMinLeaf',
  AI_CATBOOST_MAX_NEW_TREES: 'settings.ai.catboostMaxNewTrees',
  AI_AUGMENTATIONS: 'settings.ai.augmentations',
  AI_SORT_MODEL: 'settings.ai.sortModel',
  LIBRARY_PHOTO_COUNT: 'library.photoCount',
  LIBRARY_VIDEO_COUNT: 'library.videoCount',
  LIBRARY_FREE_SPACE: 'library.freeSpace',
  LIBRARY_TOTAL_SPACE: 'library.totalSpace',
  LIBRARY_PHOTOS_SIZE: 'library.photosSize',
  LIBRARY_VIDEOS_SIZE: 'library.videosSize',
  /** One-time backfill of swipe_decisions from kept_assets + pending_deletes */
  SWIPE_DECISIONS_BACKFILL_DONE: 'migrate.swipeDecisionsBackfillDone',
} as const;

export function aiTrainingHistoryKey(modelId: string): string {
  return `ai.trainingHistory.${modelId}`;
}

export function getSetting<T>(key: string, defaultValue: T): T {
  const raw = storage.getString(key);
  if (raw === undefined) return defaultValue;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return defaultValue;
  }
}

export function setSetting<T>(key: string, value: T): void {
  storage.set(key, JSON.stringify(value));
}

export {KEYS};
