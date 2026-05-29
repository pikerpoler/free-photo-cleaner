import {createMMKV, type MMKV} from 'react-native-mmkv';

export const storage: MMKV = createMMKV({id: 'aiphotocleaner'});

const KEYS = {
  PHOTO_SORT: 'settings.photo.sort',
  VIDEO_SORT: 'settings.video.sort',
  PHOTO_FILTERS: 'settings.photo.filters',
  VIDEO_FILTERS: 'settings.video.filters',
  DATE_FILTER: 'settings.date.filter',
  THEME: 'settings.theme',
  TRAIN_AI: 'settings.ai.train',
  AI_MODEL: 'settings.ai.model',
  AI_BATCH_SIZE: 'settings.ai.batchSize',
  AI_STEP_SIZE: 'settings.ai.stepSize',
} as const;

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
