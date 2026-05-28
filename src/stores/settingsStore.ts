import {create} from 'zustand';
import {PhotoFilters, SortMode, VideoFilters} from '../types/media';
import {getSetting, setSetting, KEYS} from '../services/storage';

interface SettingsState {
  photoSort: SortMode;
  videoSort: SortMode;
  photoFilters: PhotoFilters;
  videoFilters: VideoFilters;
  theme: 'system' | 'light' | 'dark';

  setPhotoSort: (sort: SortMode) => void;
  setVideoSort: (sort: SortMode) => void;
  setPhotoFilters: (filters: Partial<PhotoFilters>) => void;
  setVideoFilters: (filters: Partial<VideoFilters>) => void;
  setTheme: (theme: 'system' | 'light' | 'dark') => void;
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

export const useSettingsStore = create<SettingsState>((set, get) => ({
  photoSort: 'newest_first',
  videoSort: 'largest_first',
  photoFilters: DEFAULT_PHOTO_FILTERS,
  videoFilters: DEFAULT_VIDEO_FILTERS,
  theme: 'system',

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

  setTheme: (theme: 'system' | 'light' | 'dark') => {
    set({theme});
    setSetting(KEYS.THEME, theme);
  },

  loadSettings: () => {
    set({
      photoSort: getSetting(KEYS.PHOTO_SORT, 'newest_first' as SortMode),
      videoSort: getSetting(KEYS.VIDEO_SORT, 'largest_first' as SortMode),
      photoFilters: getSetting(KEYS.PHOTO_FILTERS, DEFAULT_PHOTO_FILTERS),
      videoFilters: getSetting(KEYS.VIDEO_FILTERS, DEFAULT_VIDEO_FILTERS),
      theme: getSetting(KEYS.THEME, 'system' as const),
    });
  },
}));
