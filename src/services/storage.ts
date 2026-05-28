import {createMMKV, type MMKV} from 'react-native-mmkv';

export const storage: MMKV = createMMKV({id: 'freephotocleaner'});

const KEYS = {
  PHOTO_SORT: 'settings.photo.sort',
  VIDEO_SORT: 'settings.video.sort',
  PHOTO_FILTERS: 'settings.photo.filters',
  VIDEO_FILTERS: 'settings.video.filters',
  THEME: 'settings.theme',
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
