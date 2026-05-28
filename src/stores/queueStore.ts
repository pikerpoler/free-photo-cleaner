import {create} from 'zustand';
import {Alert, Platform} from 'react-native';
import {
  MediaAsset,
  MediaType,
  PhotoFilters,
  SortMode,
  VideoFilters,
} from '../types/media';
import {fetchAllMediaAssets, deleteAsset} from '../services/media';
import {batchDeleteAssets} from '../services/nativeStorage';
import {
  getKeptAssetIds,
  markAssetKept,
  removeAssetFromKept,
  getPendingDeletesByType,
  addPendingDelete,
  removePendingDelete,
  clearPendingDeletesByType,
  getPendingDeleteIds,
} from '../services/database';

type UndoEntry = {type: 'keep' | 'delete'; asset: MediaAsset; queue: MediaType};

interface QueueState {
  photoQueue: MediaAsset[];
  videoQueue: MediaAsset[];
  photoIndex: number;
  videoIndex: number;
  isLoadingPhotos: boolean;
  isLoadingVideos: boolean;
  photoPendingCount: number;
  videoPendingCount: number;
  photoUndoStack: UndoEntry[];
  videoUndoStack: UndoEntry[];

  loadQueue: (
    type: MediaType,
    sort: SortMode,
    photoFilters?: PhotoFilters,
    videoFilters?: VideoFilters,
  ) => Promise<void>;
  swipeLeft: (type: MediaType) => void;
  swipeRight: (type: MediaType) => void;
  undo: (type: MediaType) => void;
  flushDeletes: (type: MediaType) => Promise<void>;
  getCurrentAsset: (type: MediaType) => MediaAsset | null;
  getNextAsset: (type: MediaType) => MediaAsset | null;
}

function sortAssets(assets: MediaAsset[], mode: SortMode): MediaAsset[] {
  const sorted = [...assets];
  switch (mode) {
    case 'random':
      for (let i = sorted.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [sorted[i], sorted[j]] = [sorted[j], sorted[i]];
      }
      return sorted;
    case 'largest_first':
      return sorted.sort((a, b) => b.fileSize - a.fileSize);
    case 'smallest_first':
      return sorted.sort((a, b) => a.fileSize - b.fileSize);
    case 'oldest_first':
      return sorted.sort((a, b) => a.creationDate - b.creationDate);
    case 'newest_first':
      return sorted.sort((a, b) => b.creationDate - a.creationDate);
  }
}

function applyPhotoFilters(
  assets: MediaAsset[],
  filters: PhotoFilters,
): MediaAsset[] {
  if (filters.mode === 'all') return assets;

  return assets.filter(asset => {
    if (filters.screenshots && asset.isScreenshot) return true;
    if (filters.whatsapp && asset.isWhatsApp) return true;
    if (filters.noMetadata && !asset.hasExif) return true;
    return false;
  });
}

function applyVideoFilters(
  assets: MediaAsset[],
  filters: VideoFilters,
): MediaAsset[] {
  return assets.filter(asset => {
    const duration = asset.duration || 0;
    const size = asset.fileSize;

    if (duration < filters.minDuration) return false;
    if (isFinite(filters.maxDuration) && duration > filters.maxDuration)
      return false;
    if (size < filters.minSize) return false;
    if (isFinite(filters.maxSize) && size > filters.maxSize) return false;

    return true;
  });
}

export const useQueueStore = create<QueueState>((set, get) => ({
  photoQueue: [],
  videoQueue: [],
  photoIndex: 0,
  videoIndex: 0,
  isLoadingPhotos: false,
  isLoadingVideos: false,
  photoPendingCount: getPendingDeletesByType('photo').length,
  videoPendingCount: getPendingDeletesByType('video').length,
  photoUndoStack: [],
  videoUndoStack: [],

  loadQueue: async (type, sort, photoFilters, videoFilters) => {
    const loadingKey =
      type === 'photo' ? 'isLoadingPhotos' : 'isLoadingVideos';
    set({[loadingKey]: true});

    try {
      let assets = await fetchAllMediaAssets(type);
      const keptIds = getKeptAssetIds();
      const pendingIds = getPendingDeleteIds();

      assets = assets.filter(a => !keptIds.has(a.id) && !pendingIds.has(a.id));

      if (type === 'photo' && photoFilters) {
        assets = applyPhotoFilters(assets, photoFilters);
      } else if (type === 'video' && videoFilters) {
        assets = applyVideoFilters(assets, videoFilters);
      }

      assets = sortAssets(assets, sort);

      const queueKey = type === 'photo' ? 'photoQueue' : 'videoQueue';
      const indexKey = type === 'photo' ? 'photoIndex' : 'videoIndex';

      set({[queueKey]: assets, [indexKey]: 0, [loadingKey]: false});
    } catch {
      set({[loadingKey]: false});
    }
  },

  swipeLeft: (type: MediaType) => {
    const state = get();
    const queue = type === 'photo' ? state.photoQueue : state.videoQueue;
    const index = type === 'photo' ? state.photoIndex : state.videoIndex;
    const asset = queue[index];

    if (!asset) return;

    const indexKey = type === 'photo' ? 'photoIndex' : 'videoIndex';
    const countKey = type === 'photo' ? 'photoPendingCount' : 'videoPendingCount';
    const stackKey = type === 'photo' ? 'photoUndoStack' : 'videoUndoStack';
    const currentStack = type === 'photo' ? state.photoUndoStack : state.videoUndoStack;
    const currentCount = type === 'photo' ? state.photoPendingCount : state.videoPendingCount;

    addPendingDelete(asset.id, asset.uri, asset.filename, asset.fileSize, asset.type);

    set({
      [indexKey]: index + 1,
      [countKey]: currentCount + 1,
      [stackKey]: [...currentStack, {type: 'delete', asset, queue: type}],
    });
  },

  swipeRight: (type: MediaType) => {
    const state = get();
    const queue = type === 'photo' ? state.photoQueue : state.videoQueue;
    const index = type === 'photo' ? state.photoIndex : state.videoIndex;
    const asset = queue[index];

    if (!asset) return;

    const indexKey = type === 'photo' ? 'photoIndex' : 'videoIndex';
    const stackKey = type === 'photo' ? 'photoUndoStack' : 'videoUndoStack';
    const currentStack = type === 'photo' ? state.photoUndoStack : state.videoUndoStack;

    markAssetKept(asset.id);

    set({
      [indexKey]: index + 1,
      [stackKey]: [...currentStack, {type: 'keep', asset, queue: type}],
    });
  },

  flushDeletes: async (type: MediaType) => {
    const pending = getPendingDeletesByType(type);

    if (pending.length === 0) {
      Alert.alert('Nothing to delete', `No pending ${type} deletions found in database.`);
      return;
    }

    const uris = pending.map(p => p.uri);

    if (Platform.OS === 'ios') {
      const result = await batchDeleteAssets(uris);
      if (result.success) {
        clearPendingDeletesByType(type);
        const countKey = type === 'photo' ? 'photoPendingCount' : 'videoPendingCount';
        const stackKey = type === 'photo' ? 'photoUndoStack' : 'videoUndoStack';
        set({[countKey]: 0, [stackKey]: []});
      }
    } else {
      for (const uri of uris) {
        await deleteAsset(uri);
      }
      clearPendingDeletesByType(type);
      const countKey = type === 'photo' ? 'photoPendingCount' : 'videoPendingCount';
      const stackKey = type === 'photo' ? 'photoUndoStack' : 'videoUndoStack';
      set({[countKey]: 0, [stackKey]: []});
    }
  },

  undo: (type: MediaType) => {
    const state = get();
    const stackKey = type === 'photo' ? 'photoUndoStack' : 'videoUndoStack';
    const stack = type === 'photo' ? state.photoUndoStack : state.videoUndoStack;

    if (stack.length === 0) return;

    const lastEntry = stack[stack.length - 1];
    const newStack = stack.slice(0, -1);

    const indexKey = type === 'photo' ? 'photoIndex' : 'videoIndex';
    const currentIndex = get()[indexKey] as number;

    if (currentIndex > 0) {
      set({[indexKey]: currentIndex - 1});
    }

    if (lastEntry.type === 'keep') {
      removeAssetFromKept(lastEntry.asset.id);
    } else {
      removePendingDelete(lastEntry.asset.id);
      const countKey = type === 'photo' ? 'photoPendingCount' : 'videoPendingCount';
      const currentCount = get()[countKey] as number;
      set({[countKey]: Math.max(0, currentCount - 1)});
    }

    set({[stackKey]: newStack});
  },

  getCurrentAsset: (type: MediaType) => {
    const state = get();
    const queue = type === 'photo' ? state.photoQueue : state.videoQueue;
    const index = type === 'photo' ? state.photoIndex : state.videoIndex;
    return queue[index] || null;
  },

  getNextAsset: (type: MediaType) => {
    const state = get();
    const queue = type === 'photo' ? state.photoQueue : state.videoQueue;
    const index = type === 'photo' ? state.photoIndex : state.videoIndex;
    return queue[index + 1] || null;
  },
}));
