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
  upsertCatalogAssets,
  getCatalogAssets,
  pruneCatalogToIds,
} from '../services/database';
import {useAIStore} from './aiStore';

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

function dedupeById(assets: MediaAsset[]): MediaAsset[] {
  const seen = new Set<string>();
  const out: MediaAsset[] = [];
  for (const a of assets) {
    if (seen.has(a.id)) continue;
    seen.add(a.id);
    out.push(a);
  }
  return out;
}

function mergeUnique(existing: MediaAsset[], batch: MediaAsset[]): MediaAsset[] {
  const seen = new Set(existing.map(a => a.id));
  const merged = [...existing];
  for (const a of batch) {
    if (!seen.has(a.id)) {
      seen.add(a.id);
      merged.push(a);
    }
  }
  return merged;
}

function sortAssets(assets: MediaAsset[], mode: SortMode): MediaAsset[] {
  if (mode === 'ai') {
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
    return scoreB - scoreA;
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
    a =>
      a.creationDate >= startOfFrom.getTime() &&
      a.creationDate <= endOfTo.getTime(),
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

function catalogDateBounds(dateFilter?: DateFilter): {
  from?: number;
  to?: number;
} {
  if (!dateFilter?.enabled) return {};
  const startOfFrom = new Date(dateFilter.from);
  startOfFrom.setHours(0, 0, 0, 0);
  const endOfTo = new Date(dateFilter.to);
  endOfTo.setHours(23, 59, 59, 999);
  return {from: startOfFrom.getTime(), to: endOfTo.getTime()};
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
    const loadingMoreKey = isPhoto
      ? 'isLoadingPhotosMore'
      : 'isLoadingVideosMore';
    const queueKey = isPhoto ? 'photoQueue' : 'videoQueue';
    const indexKey = isPhoto ? 'photoIndex' : 'videoIndex';

    // Hydrate instantly from catalog when available
    const bounds = catalogDateBounds(dateFilter);
    const catalog = getCatalogAssets(type, bounds.from, bounds.to);
    const pf = isPhoto ? photoFilters : undefined;
    const vf = isPhoto ? undefined : videoFilters;

    if (catalog.length > 0) {
      if (isPhoto) {
        photoRawCache = catalog;
      } else {
        videoRawCache = catalog;
      }
      const queue = filterAndSort(
        catalog,
        type,
        sort,
        pf,
        vf,
        dateFilter,
      );
      set({
        [queueKey]: queue,
        [indexKey]: 0,
        [loadingKey]: false,
        [loadingMoreKey]: true,
      });
      if (sort === 'ai' && isPhoto) {
        sortAssetsAI(queue).then(aiSorted => {
          if (currentPhotoSort === 'ai') {
            set({[queueKey]: aiSorted, [indexKey]: 0});
          }
        });
      }
    } else {
      set({[loadingKey]: true, [loadingMoreKey]: false});
    }

    let isFirstBatch = catalog.length === 0;
    const seenIds = new Set<string>();

    const controller = loadMediaProgressively(
      type,
      (batch, done) => {
        const uniqueBatch = dedupeById(batch);
        for (const a of uniqueBatch) seenIds.add(a.id);

        upsertCatalogAssets(uniqueBatch);

        if (isPhoto) {
          photoRawCache = mergeUnique(photoRawCache, uniqueBatch);
        } else {
          videoRawCache = mergeUnique(videoRawCache, uniqueBatch);
        }

        const rawCache = isPhoto ? photoRawCache : videoRawCache;
        const currentSort = isPhoto ? currentPhotoSort : currentVideoSort;
        const curPf = isPhoto ? currentPhotoFilters : undefined;
        const curVf = isPhoto ? undefined : currentVideoFilters;

        if (isFirstBatch) {
          isFirstBatch = false;
          const queue = filterAndSort(
            rawCache,
            type,
            currentSort,
            curPf,
            curVf,
            currentDateFilter,
          );
          set({
            [queueKey]: queue,
            [indexKey]: 0,
            [loadingKey]: false,
            [loadingMoreKey]: !done,
          });
          if (currentSort === 'ai' && isPhoto) {
            sortAssetsAI(queue).then(aiSorted => {
              if (currentPhotoSort === 'ai') {
                set({[queueKey]: aiSorted, [indexKey]: 0});
              }
            });
          }
        } else if (done) {
          // Full library sync without date filter on CameraRoll when catalog warm
          // — still prune using all seen IDs from this sync pass
          if (!currentDateFilter?.enabled) {
            pruneCatalogToIds(type, seenIds);
            // Refresh raw cache from catalog after prune
            const refreshed = getCatalogAssets(type);
            if (isPhoto) {
              photoRawCache = refreshed;
            } else {
              videoRawCache = refreshed;
            }
          }

          const finalRaw = isPhoto ? photoRawCache : videoRawCache;
          const currentQueue = isPhoto ? get().photoQueue : get().videoQueue;
          const currentIndex = isPhoto ? get().photoIndex : get().videoIndex;
          const frozenPrefix = currentQueue.slice(0, currentIndex + 1);
          const frozenIds = new Set(frozenPrefix.map(a => a.id));
          const unfrozenRaw = finalRaw.filter(a => !frozenIds.has(a.id));
          const remainder = filterAndSort(
            unfrozenRaw,
            type,
            currentSort,
            curPf,
            curVf,
            currentDateFilter,
          );
          const merged = [...frozenPrefix, ...remainder];
          set({[queueKey]: merged, [loadingMoreKey]: false, [loadingKey]: false});

          if (currentSort === 'ai' && isPhoto) {
            sortAssetsAI(remainder).then(aiSorted => {
              if (currentPhotoSort === 'ai') {
                set({[queueKey]: [...frozenPrefix, ...aiSorted]});
              }
            });
          }
        } else {
          // Intermediate: update remaining count from accumulating cache without resetting index
          const currentQueue = isPhoto ? get().photoQueue : get().videoQueue;
          const currentIndex = isPhoto ? get().photoIndex : get().videoIndex;
          const frozenPrefix = currentQueue.slice(0, Math.min(currentIndex + 1, currentQueue.length));
          const frozenIds = new Set(frozenPrefix.map(a => a.id));
          const unfrozenRaw = rawCache.filter(a => !frozenIds.has(a.id));
          const remainder = filterAndSort(
            unfrozenRaw,
            type,
            currentSort,
            curPf,
            curVf,
            currentDateFilter,
          );
          set({
            [queueKey]: [...frozenPrefix, ...remainder],
            [loadingMoreKey]: true,
            [loadingKey]: false,
          });
        }
      },
      dateFilter,
    );

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
    const queue = filterAndSort(
      rawCache,
      type,
      sort,
      pf,
      vf,
      currentDateFilter,
    );

    const queueKey = isPhoto ? 'photoQueue' : 'videoQueue';
    const indexKey = isPhoto ? 'photoIndex' : 'videoIndex';
    set({[queueKey]: queue, [indexKey]: 0});

    if (sort === 'ai' && isPhoto) {
      sortAssetsAI(queue).then(aiSorted => {
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
    const countKey =
      type === 'photo' ? 'photoPendingCount' : 'videoPendingCount';
    const stackKey = type === 'photo' ? 'photoUndoStack' : 'videoUndoStack';
    const currentStack =
      type === 'photo' ? state.photoUndoStack : state.videoUndoStack;
    const currentCount =
      type === 'photo' ? state.photoPendingCount : state.videoPendingCount;

    addPendingDelete(
      asset.id,
      asset.uri,
      asset.filename,
      asset.fileSize,
      asset.type,
    );

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
    const currentStack =
      type === 'photo' ? state.photoUndoStack : state.videoUndoStack;

    markAssetKept(asset.id);

    set({
      [indexKey]: index + 1,
      [stackKey]: [...currentStack, {type: 'keep', asset, queue: type}],
    });
  },

  flushDeletes: async (type: MediaType) => {
    const pending = getPendingDeletesByType(type);

    if (pending.length === 0) {
      Alert.alert(
        'Nothing to delete',
        `No pending ${type} deletions found in database.`,
      );
      return;
    }

    const uris = pending.map(p => p.uri);

    if (Platform.OS === 'ios') {
      const result = await batchDeleteAssets(uris);
      if (result.success) {
        clearPendingDeletesByType(type);
        const countKey =
          type === 'photo' ? 'photoPendingCount' : 'videoPendingCount';
        const stackKey =
          type === 'photo' ? 'photoUndoStack' : 'videoUndoStack';
        set({[countKey]: 0, [stackKey]: []});
      }
    } else {
      for (const uri of uris) {
        await deleteAsset(uri);
      }
      clearPendingDeletesByType(type);
      const countKey =
        type === 'photo' ? 'photoPendingCount' : 'videoPendingCount';
      const stackKey = type === 'photo' ? 'photoUndoStack' : 'videoUndoStack';
      set({[countKey]: 0, [stackKey]: []});
    }
  },

  undo: (type: MediaType) => {
    const state = get();
    const stackKey = type === 'photo' ? 'photoUndoStack' : 'videoUndoStack';
    const stack =
      type === 'photo' ? state.photoUndoStack : state.videoUndoStack;

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
      const countKey =
        type === 'photo' ? 'photoPendingCount' : 'videoPendingCount';
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
