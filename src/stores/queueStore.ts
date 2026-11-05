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
  /** Photos still in the library that pass the current filters. */
  photoEligibleCount: number;
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
  resetAIQueue: () => Promise<void>;
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

/** Max scored photos ahead of (and including) the current card for AI sort. */
const AI_QUEUE_CAP = 300;
/** Unscored candidates scored per refill pass. */
const AI_SCORE_BATCH = 32;
/** Kick another refill when the scored window drops below this. */
const AI_REFILL_THRESHOLD = 250;

let aiFillInFlight = false;

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
  const sorted = [...assets];
  switch (mode) {
    case 'ai':
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

function sampleRandom<T>(arr: T[], n: number): T[] {
  if (arr.length <= n) return [...arr];
  const copy = [...arr];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy.slice(0, n);
}

function filterEligible(
  raw: MediaAsset[],
  type: MediaType,
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

  return assets;
}

function countPhotoEligible(): number {
  return filterEligible(
    photoRawCache,
    'photo',
    currentPhotoFilters,
    undefined,
    currentDateFilter,
  ).length;
}

function buildScoredRemainder(
  eligible: MediaAsset[],
  frozenIds: Set<string>,
  scoreById: Record<string, number>,
  cap: number,
): MediaAsset[] {
  const scored = eligible.filter(
    a => !frozenIds.has(a.id) && scoreById[a.id] !== undefined,
  );
  scored.sort((a, b) => (scoreById[b.id] ?? 0) - (scoreById[a.id] ?? 0));
  return scored.slice(0, Math.max(0, cap));
}

/** Step the cursor back onto an undone card, reinserting it if a refill removed it. */
function restoreUndoneAsset(
  queue: MediaAsset[],
  index: number,
  asset: MediaAsset,
): {queue: MediaAsset[]; index: number} {
  if (index > 0 && queue[index - 1]?.id === asset.id) {
    return {queue, index: index - 1};
  }
  if (queue[index]?.id === asset.id) {
    return {queue, index};
  }

  const removedBefore = queue
    .slice(0, index)
    .reduce((count, item) => count + (item.id === asset.id ? 1 : 0), 0);
  const without = queue.filter(item => item.id !== asset.id);
  const insertAt = Math.max(
    0,
    Math.min(index - removedBefore, without.length),
  );
  return {
    queue: [...without.slice(0, insertAt), asset, ...without.slice(insertAt)],
    index: insertAt,
  };
}

function rebuildAIQueueFromScores(get: () => QueueState): {
  queue: MediaAsset[];
  index: number;
  remaining: number;
  unscoredCount: number;
  eligibleCount: number;
} {
  const state = get();
  const scoreById = useAIStore.getState().scoreById;
  const eligible = filterEligible(
    photoRawCache,
    'photo',
    currentPhotoFilters,
    undefined,
    currentDateFilter,
  );
  const unscoredCount = eligible.filter(
    a => scoreById[a.id] === undefined,
  ).length;

  const index = state.photoIndex;
  const past = state.photoQueue.slice(0, index);
  const current = state.photoQueue[index];
  const frozenIds = new Set(past.map(a => a.id));
  if (current) frozenIds.add(current.id);

  // Drop past/current cards that are no longer eligible, but keep anything
  // still on the undo stack. Those cards sit just behind the cursor, and
  // removing them makes undo pop the button without bringing the photo back.
  const eligibleIds = new Set(eligible.map(a => a.id));
  const undoIds = new Set(state.photoUndoStack.map(entry => entry.asset.id));
  const pastKept = past.filter(
    a => eligibleIds.has(a.id) || undoIds.has(a.id),
  );
  const currentKept =
    current && eligibleIds.has(current.id) ? current : undefined;

  const aheadCap = currentKept ? AI_QUEUE_CAP - 1 : AI_QUEUE_CAP;
  const frozenForAhead = new Set(pastKept.map(a => a.id));
  if (currentKept) frozenForAhead.add(currentKept.id);

  const ahead = buildScoredRemainder(
    eligible,
    frozenForAhead,
    scoreById,
    aheadCap,
  );

  const queue = currentKept
    ? [...pastKept, currentKept, ...ahead]
    : ahead;
  const nextIndex = currentKept ? pastKept.length : 0;
  const remaining = queue.length - nextIndex;

  return {
    queue,
    index: nextIndex,
    remaining,
    unscoredCount,
    eligibleCount: eligible.length,
  };
}

async function ensureAIQueueFilled(
  set: (
    partial:
      | Partial<QueueState>
      | ((state: QueueState) => Partial<QueueState>),
  ) => void,
  get: () => QueueState,
): Promise<void> {
  if (currentPhotoSort !== 'ai') return;
  if (aiFillInFlight) return;
  aiFillInFlight = true;

  try {
    while (currentPhotoSort === 'ai') {
      const built = rebuildAIQueueFromScores(get);
      const stillNeedScores =
        built.remaining < AI_QUEUE_CAP && built.unscoredCount > 0;

      set({
        photoQueue: built.queue,
        photoIndex: built.index,
        photoEligibleCount: built.eligibleCount,
        isLoadingPhotos:
          built.queue.length === 0 &&
          (stillNeedScores || photoRawCache.length === 0),
      });

      if (!stillNeedScores) {
        if (built.queue.length > 0 || built.unscoredCount === 0) {
          set({isLoadingPhotos: false});
        }
        break;
      }

      const scoreById = useAIStore.getState().scoreById;
      const eligible = filterEligible(
        photoRawCache,
        'photo',
        currentPhotoFilters,
        undefined,
        currentDateFilter,
      );
      const unscored = eligible.filter(a => scoreById[a.id] === undefined);
      if (unscored.length === 0) {
        set({isLoadingPhotos: false});
        break;
      }

      const batch = sampleRandom(unscored, AI_SCORE_BATCH);
      await useAIStore.getState().scoreAssets(batch);
    }
  } finally {
    aiFillInFlight = false;
  }
}

function maybeRefillAIQueue(
  set: (
    partial:
      | Partial<QueueState>
      | ((state: QueueState) => Partial<QueueState>),
  ) => void,
  get: () => QueueState,
): void {
  if (currentPhotoSort !== 'ai') return;
  const state = get();
  const remaining = state.photoQueue.length - state.photoIndex;
  if (remaining < AI_REFILL_THRESHOLD) {
    void ensureAIQueueFilled(set, get);
  }
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
  return sortAssets(
    filterEligible(raw, type, photoFilters, videoFilters, dateFilter),
    sort,
  );
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
  photoEligibleCount: 0,
  photoUndoStack: [],
  videoUndoStack: [],

  loadQueue: (type, sort, photoFilters, videoFilters, dateFilter) => {
    const isPhoto = type === 'photo';
    const isAI = isPhoto && sort === 'ai';

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

      if (isAI) {
        // Never paint unscored/random cards — fill from scores only.
        set({
          photoQueue: [],
          photoIndex: 0,
          photoEligibleCount: countPhotoEligible(),
          isLoadingPhotos: true,
          isLoadingPhotosMore: true,
        });
        void ensureAIQueueFilled(set, get);
      } else {
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
          ...(isPhoto ? {photoEligibleCount: countPhotoEligible()} : {}),
        });
      }
    } else {
      set({
        [queueKey]: [],
        [indexKey]: 0,
        [loadingKey]: true,
        [loadingMoreKey]: false,
        ...(isPhoto ? {photoEligibleCount: countPhotoEligible()} : {}),
      });
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
        const aiMode = isPhoto && currentSort === 'ai';

        if (aiMode) {
          // Catalog/rawCache only — never shuffle unscored into photoQueue.
          if (done) {
            if (!currentDateFilter?.enabled) {
              pruneCatalogToIds(type, seenIds);
              photoRawCache = getCatalogAssets(type);
            }
            set({
              isLoadingPhotosMore: false,
              photoEligibleCount: countPhotoEligible(),
            });
          } else {
            set({
              isLoadingPhotosMore: true,
              photoEligibleCount: countPhotoEligible(),
            });
          }
          void ensureAIQueueFilled(set, get);
          return;
        }

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
            ...(isPhoto ? {photoEligibleCount: countPhotoEligible()} : {}),
          });
        } else if (done) {
          // Full library sync without date filter on CameraRoll when catalog warm
          // — still prune using all seen IDs from this sync pass
          if (!currentDateFilter?.enabled) {
            pruneCatalogToIds(type, seenIds);
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
          set({
            [queueKey]: [...frozenPrefix, ...remainder],
            [loadingMoreKey]: false,
            [loadingKey]: false,
            ...(isPhoto ? {photoEligibleCount: countPhotoEligible()} : {}),
          });
        } else {
          // Intermediate: update remaining without resetting index
          const currentQueue = isPhoto ? get().photoQueue : get().videoQueue;
          const currentIndex = isPhoto ? get().photoIndex : get().videoIndex;
          const frozenPrefix = currentQueue.slice(
            0,
            Math.min(currentIndex + 1, currentQueue.length),
          );
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
            ...(isPhoto ? {photoEligibleCount: countPhotoEligible()} : {}),
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

    const queueKey = isPhoto ? 'photoQueue' : 'videoQueue';
    const indexKey = isPhoto ? 'photoIndex' : 'videoIndex';

    if (isPhoto && sort === 'ai') {
      set({
        photoQueue: [],
        photoIndex: 0,
        photoEligibleCount: countPhotoEligible(),
        isLoadingPhotos: true,
      });
      void ensureAIQueueFilled(set, get);
      return;
    }

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

    set({
      [queueKey]: queue,
      [indexKey]: 0,
      isLoadingPhotos: false,
      ...(isPhoto ? {photoEligibleCount: countPhotoEligible()} : {}),
    });
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
      ...(type === 'photo' ? {photoEligibleCount: countPhotoEligible()} : {}),
    });

    if (type === 'photo') {
      maybeRefillAIQueue(set, get);
    }
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
      ...(type === 'photo' ? {photoEligibleCount: countPhotoEligible()} : {}),
    });

    if (type === 'photo') {
      maybeRefillAIQueue(set, get);
    }
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
    const isPhoto = type === 'photo';
    const stackKey = isPhoto ? 'photoUndoStack' : 'videoUndoStack';
    const stack = isPhoto ? state.photoUndoStack : state.videoUndoStack;

    if (stack.length === 0) return;

    const lastEntry = stack[stack.length - 1];
    const newStack = stack.slice(0, -1);

    if (lastEntry.type === 'keep') {
      removeAssetFromKept(lastEntry.asset.id);
    } else {
      removePendingDelete(lastEntry.asset.id);
    }

    const queue = isPhoto ? state.photoQueue : state.videoQueue;
    const index = isPhoto ? state.photoIndex : state.videoIndex;
    const restored = restoreUndoneAsset(queue, index, lastEntry.asset);

    const countKey = isPhoto ? 'photoPendingCount' : 'videoPendingCount';
    const currentCount = state[countKey];

    set({
      [isPhoto ? 'photoQueue' : 'videoQueue']: restored.queue,
      [isPhoto ? 'photoIndex' : 'videoIndex']: restored.index,
      [stackKey]: newStack,
      ...(lastEntry.type === 'delete'
        ? {[countKey]: Math.max(0, currentCount - 1)}
        : {}),
      ...(isPhoto ? {photoEligibleCount: countPhotoEligible()} : {}),
    });
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

  resetAIQueue: async () => {
    useAIStore.getState().clearScores();
    if (photoRawCache.length === 0) return;

    if (currentPhotoSort === 'ai') {
      set({
        photoQueue: [],
        photoIndex: 0,
        photoEligibleCount: countPhotoEligible(),
        isLoadingPhotos: true,
      });
      await ensureAIQueueFilled(set, get);
      return;
    }

    const queue = filterAndSort(
      photoRawCache,
      'photo',
      currentPhotoSort,
      currentPhotoFilters,
      undefined,
      currentDateFilter,
    );
    set({
      photoQueue: queue,
      photoIndex: 0,
      photoEligibleCount: countPhotoEligible(),
    });
  },
}));
