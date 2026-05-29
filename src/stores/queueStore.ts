import {create} from 'zustand';
import {Alert, Platform} from 'react-native';
import {
  DateFilter,
  MediaAsset,
  MediaType,
  PhotoFilters,
  SortMode,
  VideoFilters,
} from '../types/media';
import {
  loadMediaProgressively,
  deleteAsset,
  ProgressiveLoadController,
} from '../services/media';
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
import {useAIStore} from './aiStore';
import {useSettingsStore} from './settingsStore';
import {clearTrainingQueue} from '../ai/trainer';

type UndoEntry = {type: 'keep' | 'delete'; asset: MediaAsset; queue: MediaType};

interface QueueState {
  photoQueue: MediaAsset[];
  videoQueue: MediaAsset[];
  photoIndex: number;
  videoIndex: number;
  isLoadingPhotos: boolean;
  isLoadingVideos: boolean;
  isLoadingPhotosMore: boolean;
  isLoadingVideosMore: boolean;
  photoPendingCount: number;
  videoPendingCount: number;
  photoUndoStack: UndoEntry[];
  videoUndoStack: UndoEntry[];

  loadQueue: (
    type: MediaType,
    sort: SortMode,
    photoFilters?: PhotoFilters,
    videoFilters?: VideoFilters,
    dateFilter?: DateFilter,
  ) => void;
  applySort: (type: MediaType, sort: SortMode) => void;
  swipeLeft: (type: MediaType) => void;
  swipeRight: (type: MediaType) => void;
  undo: (type: MediaType) => void;
  flushDeletes: (type: MediaType) => Promise<void>;
  getCurrentAsset: (type: MediaType) => MediaAsset | null;
  getNextAsset: (type: MediaType) => MediaAsset | null;
}

let photoLoadController: ProgressiveLoadController | null = null;
let videoLoadController: ProgressiveLoadController | null = null;

let photoRawCache: MediaAsset[] = [];
let videoRawCache: MediaAsset[] = [];

let currentPhotoSort: SortMode = 'newest_first';
let currentVideoSort: SortMode = 'largest_first';
let currentPhotoFilters: PhotoFilters | undefined;
let currentVideoFilters: VideoFilters | undefined;
let currentDateFilter: DateFilter | undefined;

function sortAssets(assets: MediaAsset[], mode: SortMode): MediaAsset[] {
  if (mode === 'ai') {
    // AI sort is handled asynchronously via sortAssetsAI; fallback to random here
    const sorted = [...assets];
    for (let i = sorted.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [sorted[i], sorted[j]] = [sorted[j], sorted[i]];
    }
    return sorted;
  }
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

async function sortAssetsAI(assets: MediaAsset[]): Promise<MediaAsset[]> {
  const aiStore = useAIStore.getState();
  const scores = await aiStore.scoreAssets(assets);
  const sorted = [...assets];
  sorted.sort((a, b) => {
    const scoreA = scores.get(a.id) ?? 0.5;
    const scoreB = scores.get(b.id) ?? 0.5;
    return scoreB - scoreA; // Highest P(delete) first
  });
  return sorted;
}

function applyDateFilter(
  assets: MediaAsset[],
  dateFilter: DateFilter,
): MediaAsset[] {
  if (!dateFilter.enabled) return assets;
  const startOfFrom = new Date(dateFilter.from);
  startOfFrom.setHours(0, 0, 0, 0);
  const endOfTo = new Date(dateFilter.to);
  endOfTo.setHours(23, 59, 59, 999);
  return assets.filter(
    a => a.creationDate >= startOfFrom.getTime() && a.creationDate <= endOfTo.getTime(),
  );
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

function filterAndSort(
  raw: MediaAsset[],
  type: MediaType,
  sort: SortMode,
  photoFilters?: PhotoFilters,
  videoFilters?: VideoFilters,
  dateFilter?: DateFilter,
): MediaAsset[] {
  const keptIds = getKeptAssetIds();
  const pendingIds = getPendingDeleteIds();

  let assets = raw.filter(a => !keptIds.has(a.id) && !pendingIds.has(a.id));

  if (dateFilter) {
    assets = applyDateFilter(assets, dateFilter);
  }

  if (type === 'photo' && photoFilters) {
    assets = applyPhotoFilters(assets, photoFilters);
  } else if (type === 'video' && videoFilters) {
    assets = applyVideoFilters(assets, videoFilters);
  }

  return sortAssets(assets, sort);
}

function showDeleteTrainingGuard(
  queueSize: number,
  batchSize: number,
  aiStore: ReturnType<typeof useAIStore.getState>,
): Promise<boolean> {
  return new Promise(resolve => {
    if (queueSize > 0 && queueSize < batchSize) {
      // Case B: partial batch
      Alert.alert(
        `${queueSize} photo${queueSize > 1 ? 's' : ''} haven't trained yet`,
        'Quick-train before deleting?',
        [
          {
            text: 'Train & Delete',
            onPress: async () => {
              await aiStore.flushPartial();
              resolve(true);
            },
          },
          {
            text: 'Skip & Delete',
            style: 'destructive',
            onPress: () => {
              clearTrainingQueue();
              resolve(true);
            },
          },
        ],
        {cancelable: false},
      );
    } else {
      // Case C: large queue (>= batchSize)
      Alert.alert(
        'AI is still learning',
        `Your recent choices (${queueSize} remaining) are still being learned from. Wait or delete now?`,
        [
          {
            text: 'Wait',
            style: 'cancel',
            onPress: () => resolve(false),
          },
          {
            text: 'Delete Now',
            style: 'destructive',
            onPress: () => {
              clearTrainingQueue();
              resolve(true);
            },
          },
        ],
        {cancelable: false},
      );
    }
  });
}

export const useQueueStore = create<QueueState>((set, get) => ({
  photoQueue: [],
  videoQueue: [],
  photoIndex: 0,
  videoIndex: 0,
  isLoadingPhotos: false,
  isLoadingVideos: false,
  isLoadingPhotosMore: false,
  isLoadingVideosMore: false,
  photoPendingCount: getPendingDeletesByType('photo').length,
  videoPendingCount: getPendingDeletesByType('video').length,
  photoUndoStack: [],
  videoUndoStack: [],

  loadQueue: (type, sort, photoFilters, videoFilters, dateFilter) => {
    const isPhoto = type === 'photo';

    if (isPhoto) {
      photoLoadController?.cancel();
      currentPhotoSort = sort;
      currentPhotoFilters = photoFilters;
      photoRawCache = [];
    } else {
      videoLoadController?.cancel();
      currentVideoSort = sort;
      currentVideoFilters = videoFilters;
      videoRawCache = [];
    }
    currentDateFilter = dateFilter;

    const loadingKey = isPhoto ? 'isLoadingPhotos' : 'isLoadingVideos';
    const loadingMoreKey = isPhoto ? 'isLoadingPhotosMore' : 'isLoadingVideosMore';
    set({[loadingKey]: true, [loadingMoreKey]: false});

    let isFirstBatch = true;

    const controller = loadMediaProgressively(type, (batch, done) => {
      if (isPhoto) {
        photoRawCache = [...photoRawCache, ...batch];
      } else {
        videoRawCache = [...videoRawCache, ...batch];
      }

      const rawCache = isPhoto ? photoRawCache : videoRawCache;
      const currentSort = isPhoto ? currentPhotoSort : currentVideoSort;
      const pf = isPhoto ? currentPhotoFilters : undefined;
      const vf = isPhoto ? undefined : currentVideoFilters;
      const queueKey = isPhoto ? 'photoQueue' : 'videoQueue';

      if (isFirstBatch) {
        isFirstBatch = false;
        const queue = filterAndSort(rawCache, type, currentSort, pf, vf, currentDateFilter);
        const indexKey = isPhoto ? 'photoIndex' : 'videoIndex';
        set({
          [queueKey]: queue,
          [indexKey]: 0,
          [loadingKey]: false,
          [loadingMoreKey]: !done,
        });
        // Async AI re-sort after initial load
        if (currentSort === 'ai' && isPhoto) {
          sortAssetsAI(queue).then(aiSorted => {
            if (currentPhotoSort === 'ai') {
              set({[queueKey]: aiSorted, [indexKey]: 0});
            }
          });
        }
      } else if (done) {
        // All batches loaded -- merge remainder behind the user's current position
        const currentQueue = isPhoto ? get().photoQueue : get().videoQueue;
        const currentIndex = isPhoto ? get().photoIndex : get().videoIndex;
        const frozenPrefix = currentQueue.slice(0, currentIndex + 1);
        const frozenIds = new Set(frozenPrefix.map(a => a.id));
        const unfrozenRaw = rawCache.filter(a => !frozenIds.has(a.id));
        const remainder = filterAndSort(unfrozenRaw, type, currentSort, pf, vf, currentDateFilter);
        const merged = [...frozenPrefix, ...remainder];
        set({[queueKey]: merged, [loadingMoreKey]: false});

        // AI re-sort the unfrozen portion
        if (currentSort === 'ai' && isPhoto) {
          sortAssetsAI(remainder).then(aiSorted => {
            if (currentPhotoSort === 'ai') {
              set({[queueKey]: [...frozenPrefix, ...aiSorted]});
            }
          });
        }
      }
      // Intermediate batches: silently accumulate in rawCache, no queue update
    });

    if (isPhoto) {
      photoLoadController = controller;
    } else {
      videoLoadController = controller;
    }
  },

  applySort: (type: MediaType, sort: SortMode) => {
    const isPhoto = type === 'photo';
    const rawCache = isPhoto ? photoRawCache : videoRawCache;

    if (isPhoto) {
      currentPhotoSort = sort;
    } else {
      currentVideoSort = sort;
    }

    if (rawCache.length === 0) return;

    const pf = isPhoto ? currentPhotoFilters : undefined;
    const vf = isPhoto ? undefined : currentVideoFilters;
    const queue = filterAndSort(rawCache, type, sort, pf, vf, currentDateFilter);

    const queueKey = isPhoto ? 'photoQueue' : 'videoQueue';
    const indexKey = isPhoto ? 'photoIndex' : 'videoIndex';
    set({[queueKey]: queue, [indexKey]: 0});

    // If AI sort mode and photo type, async re-sort with scores
    if (sort === 'ai' && isPhoto) {
      sortAssetsAI(queue).then(aiSorted => {
        // Only apply if sort mode hasn't changed since we started
        if (currentPhotoSort === 'ai') {
          set({[queueKey]: aiSorted, [indexKey]: 0});
        }
      });
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

    // Train AI on this swipe (photos only, non-blocking)
    if (type === 'photo') {
      useAIStore.getState().trainOnSwipe(asset.id, 1);
    }

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

    // Train AI on this swipe (photos only, non-blocking)
    if (type === 'photo') {
      useAIStore.getState().trainOnSwipe(asset.id, 0);
    }

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

    // Check AI training queue before deleting (photos only)
    if (type === 'photo') {
      const aiStore = useAIStore.getState();
      const queueSize = aiStore.getTrainingQueueSize();
      const {aiBatchSize, trainAI} = useSettingsStore.getState();

      if (trainAI && queueSize > 0) {
        const proceed = await showDeleteTrainingGuard(queueSize, aiBatchSize, aiStore);
        if (!proceed) return;
      }
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
