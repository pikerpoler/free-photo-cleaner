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
  getPersistedUndoStack,
  getPendingDeleteAggregate,
  getKeptAggregate,
} from '../services/database';
import {useAIStore} from './aiStore';
import {deleteCachedFeatures, pruneFeatureCacheFromUris} from '../services/cnnTrainer';
import {
  cancelTrainingExample,
  captureTrainingExample,
  waitForTrainingCaptures,
} from '../services/trainingCapture';

type UndoEntry = {type: 'keep' | 'delete'; asset: MediaAsset; queue: MediaType};

export type ReviewAggregateStats = {
  toReviewCount: number;
  toReviewBytes: number;
  pendingDeleteCount: number;
  pendingDeleteBytes: number;
  keptCount: number;
  keptBytes: number;
};

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
  photoReviewStats: ReviewAggregateStats;
  videoReviewStats: ReviewAggregateStats;
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
  applyFilters: (
    type: MediaType,
    sort: SortMode,
    photoFilters?: PhotoFilters,
    videoFilters?: VideoFilters,
  ) => void;
  swipeLeft: (type: MediaType) => void;
  swipeRight: (type: MediaType) => void;
  undo: (type: MediaType) => void;
  flushDeletes: (type: MediaType) => Promise<void>;
  getCurrentAsset: (type: MediaType) => MediaAsset | null;
  getNextAsset: (type: MediaType) => MediaAsset | null;
  resetAIQueue: () => Promise<void>;
  /** Re-sync undo stacks from DB after settings cleared keep history */
  onKeepHistoryReset: () => void;
}

let photoLoadController: ProgressiveLoadController | null = null;
let videoLoadController: ProgressiveLoadController | null = null;

let photoRawCache: MediaAsset[] = [];
let photoRawIds = new Set<string>();
let videoRawCache: MediaAsset[] = [];
let videoRawIds = new Set<string>();

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

function setRawCache(type: MediaType, assets: MediaAsset[]): void {
  if (type === 'photo') {
    photoRawCache = assets;
    photoRawIds = new Set(assets.map(a => a.id));
  } else {
    videoRawCache = assets;
    videoRawIds = new Set(assets.map(a => a.id));
  }
}

/** Append unseen assets. Returns how many were added. */
function appendNewRawAssets(type: MediaType, batch: MediaAsset[]): number {
  const ids = type === 'photo' ? photoRawIds : videoRawIds;
  const added: MediaAsset[] = [];
  for (const a of batch) {
    if (ids.has(a.id)) continue;
    ids.add(a.id);
    added.push(a);
  }
  if (added.length === 0) return 0;
  if (type === 'photo') {
    photoRawCache = [...photoRawCache, ...added];
  } else {
    videoRawCache = [...videoRawCache, ...added];
  }
  return added.length;
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

type MatchedCache = {
  raw: MediaAsset[];
  photoFilters?: PhotoFilters;
  videoFilters?: VideoFilters;
  dateFilter?: DateFilter;
  result: MediaAsset[];
};

let photoMatchedCache: MatchedCache | null = null;
let videoMatchedCache: MatchedCache | null = null;

function filterMatched(
  raw: MediaAsset[],
  type: MediaType,
  photoFilters?: PhotoFilters,
  videoFilters?: VideoFilters,
  dateFilter?: DateFilter,
): MediaAsset[] {
  const cache = type === 'photo' ? photoMatchedCache : videoMatchedCache;
  if (
    cache &&
    cache.raw === raw &&
    cache.photoFilters === photoFilters &&
    cache.videoFilters === videoFilters &&
    cache.dateFilter === dateFilter
  ) {
    return cache.result;
  }

  let assets = raw;
  if (dateFilter) {
    assets = applyDateFilter(assets, dateFilter);
  }
  if (type === 'photo' && photoFilters) {
    assets = applyPhotoFilters(assets, photoFilters);
  } else if (type === 'video' && videoFilters) {
    assets = applyVideoFilters(assets, videoFilters);
  }

  const entry: MatchedCache = {
    raw,
    photoFilters,
    videoFilters,
    dateFilter,
    result: assets,
  };
  if (type === 'photo') {
    photoMatchedCache = entry;
  } else {
    videoMatchedCache = entry;
  }
  return assets;
}

function filterEligible(
  raw: MediaAsset[],
  type: MediaType,
  photoFilters?: PhotoFilters,
  videoFilters?: VideoFilters,
  dateFilter?: DateFilter,
): MediaAsset[] {
  const matched = filterMatched(
    raw,
    type,
    photoFilters,
    videoFilters,
    dateFilter,
  );
  const keptIds = getKeptAssetIds();
  const pendingIds = getPendingDeleteIds();
  return matched.filter(a => !keptIds.has(a.id) && !pendingIds.has(a.id));
}

function sumFileSizes(assets: MediaAsset[]): number {
  return assets.reduce((sum, a) => sum + a.fileSize, 0);
}

function getPhotoToReviewStats(): {count: number; totalBytes: number} {
  const eligible = filterEligible(
    photoRawCache,
    'photo',
    currentPhotoFilters,
    undefined,
    currentDateFilter,
  );
  return {
    count: eligible.length,
    totalBytes: sumFileSizes(eligible),
  };
}

function getVideoToReviewStats(): {count: number; totalBytes: number} {
  const eligible = filterEligible(
    videoRawCache,
    'video',
    undefined,
    currentVideoFilters,
    currentDateFilter,
  );
  return {
    count: eligible.length,
    totalBytes: sumFileSizes(eligible),
  };
}

function buildReviewAggregates(): {
  photoEligibleCount: number;
  photoReviewStats: ReviewAggregateStats;
  videoReviewStats: ReviewAggregateStats;
} {
  const photoTR = getPhotoToReviewStats();
  const videoTR = getVideoToReviewStats();
  const photoPend = getPendingDeleteAggregate('photo');
  const videoPend = getPendingDeleteAggregate('video');
  const photoKept = getKeptAggregate('photo');
  const videoKept = getKeptAggregate('video');
  return {
    photoEligibleCount: photoTR.count,
    photoReviewStats: {
      toReviewCount: photoTR.count,
      toReviewBytes: photoTR.totalBytes,
      pendingDeleteCount: photoPend.count,
      pendingDeleteBytes: photoPend.totalBytes,
      keptCount: photoKept.count,
      keptBytes: photoKept.totalBytes,
    },
    videoReviewStats: {
      toReviewCount: videoTR.count,
      toReviewBytes: videoTR.totalBytes,
      pendingDeleteCount: videoPend.count,
      pendingDeleteBytes: videoPend.totalBytes,
      keptCount: videoKept.count,
      keptBytes: videoKept.totalBytes,
    },
  };
}

function reviewStatsPartial(type?: MediaType): Partial<
  Pick<
    QueueState,
    'photoEligibleCount' | 'photoReviewStats' | 'videoReviewStats'
  >
> {
  if (type === 'photo') {
    const photoTR = getPhotoToReviewStats();
    const photoPend = getPendingDeleteAggregate('photo');
    const photoKept = getKeptAggregate('photo');
    return {
      photoEligibleCount: photoTR.count,
      photoReviewStats: {
        toReviewCount: photoTR.count,
        toReviewBytes: photoTR.totalBytes,
        pendingDeleteCount: photoPend.count,
        pendingDeleteBytes: photoPend.totalBytes,
        keptCount: photoKept.count,
        keptBytes: photoKept.totalBytes,
      },
    };
  }
  if (type === 'video') {
    const videoTR = getVideoToReviewStats();
    const videoPend = getPendingDeleteAggregate('video');
    const videoKept = getKeptAggregate('video');
    return {
      videoReviewStats: {
        toReviewCount: videoTR.count,
        toReviewBytes: videoTR.totalBytes,
        pendingDeleteCount: videoPend.count,
        pendingDeleteBytes: videoPend.totalBytes,
        keptCount: videoKept.count,
        keptBytes: videoKept.totalBytes,
      },
    };
  }
  return buildReviewAggregates();
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
        isLoadingPhotos:
          built.queue.length === 0 &&
          (stillNeedScores || photoRawCache.length === 0),
        ...reviewStatsPartial(),
      });

      if (!stillNeedScores) {
        if (built.queue.length > 0 || built.unscoredCount === 0) {
          set({isLoadingPhotos: false, ...reviewStatsPartial()});
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
        set({isLoadingPhotos: false, ...reviewStatsPartial()});
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
    void ensureAIQueueFilled(set, get); // eslint-disable-line no-void
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

/**
 * Rebuild the queue after a progressive page without moving the current card.
 * Random order keeps every card already placed and only shuffles assets that
 * were not in the queue yet, appended after them. Other modes re-sort the tail.
 */
function rebuildLoadedQueue(
  currentQueue: MediaAsset[],
  currentIndex: number,
  raw: MediaAsset[],
  type: MediaType,
  sort: SortMode,
  photoFilters?: PhotoFilters,
  videoFilters?: VideoFilters,
  dateFilter?: DateFilter,
): MediaAsset[] {
  const frozenCount = Math.min(currentIndex + 1, currentQueue.length);
  const frozenPrefix = currentQueue.slice(0, frozenCount);

  if (sort === 'random') {
    const frozenIds = new Set(frozenPrefix.map(a => a.id));
    const eligible = filterEligible(
      raw,
      type,
      photoFilters,
      videoFilters,
      dateFilter,
    );
    const eligibleById = new Map(eligible.map(a => [a.id, a]));
    const placed = new Set(frozenIds);
    const keptRemainder: MediaAsset[] = [];
    for (const asset of currentQueue.slice(frozenCount)) {
      const fresh = eligibleById.get(asset.id);
      if (!fresh || placed.has(asset.id)) continue;
      placed.add(asset.id);
      keptRemainder.push(fresh);
    }
    const newcomers = eligible.filter(a => !placed.has(a.id));
    if (
      newcomers.length === 0 &&
      keptRemainder.length === currentQueue.length - frozenCount
    ) {
      return currentQueue;
    }
    const shuffledNew =
      newcomers.length > 0 ? sortAssets(newcomers, 'random') : newcomers;
    return [...frozenPrefix, ...keptRemainder, ...shuffledNew];
  }

  const frozenIds = new Set(frozenPrefix.map(a => a.id));
  const unfrozenRaw = raw.filter(a => !frozenIds.has(a.id));
  const remainder = filterAndSort(
    unfrozenRaw,
    type,
    sort,
    photoFilters,
    videoFilters,
    dateFilter,
  );
  return [...frozenPrefix, ...remainder];
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

const initialReview = buildReviewAggregates();

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
  photoEligibleCount: initialReview.photoEligibleCount,
  photoReviewStats: initialReview.photoReviewStats,
  videoReviewStats: initialReview.videoReviewStats,
  photoUndoStack: getPersistedUndoStack('photo') as UndoEntry[],
  videoUndoStack: getPersistedUndoStack('video') as UndoEntry[],

  loadQueue: (type, sort, photoFilters, videoFilters, dateFilter) => {
    const isPhoto = type === 'photo';
    const isAI = isPhoto && sort === 'ai';

    if (isPhoto) {
      photoLoadController?.cancel();
      currentPhotoSort = sort;
      currentPhotoFilters = photoFilters;
      setRawCache('photo', []);
    } else {
      videoLoadController?.cancel();
      currentVideoSort = sort;
      currentVideoFilters = videoFilters;
      setRawCache('video', []);
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
      setRawCache(type, catalog);

      if (isAI) {
        // Never paint unscored/random cards — fill from scores only.
        set({
          photoQueue: [],
          photoIndex: 0,
          isLoadingPhotos: true,
          isLoadingPhotosMore: true,
          ...reviewStatsPartial(),
        });
        void ensureAIQueueFilled(set, get); // eslint-disable-line no-void
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
          ...reviewStatsPartial(),
        });
      }
    } else {
      set({
        [queueKey]: [],
        [indexKey]: 0,
        [loadingKey]: true,
        [loadingMoreKey]: false,
        ...reviewStatsPartial(),
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

        const addedCount = appendNewRawAssets(type, uniqueBatch);

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
              if (type === 'photo') {
                pruneFeatureCacheFromUris(Array.from(seenIds)).catch(() => {});
              }
              setRawCache('photo', getCatalogAssets(type));
            }
            set({
              isLoadingPhotosMore: false,
              ...reviewStatsPartial(),
            });
          } else if (addedCount === 0) {
            return;
          } else {
            set({
              isLoadingPhotosMore: true,
              ...reviewStatsPartial(),
            });
          }
          void ensureAIQueueFilled(set, get); // eslint-disable-line no-void
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
            ...reviewStatsPartial(),
          });
        } else if (done) {
          // Full library sync without date filter on CameraRoll when catalog warm
          // — still prune using all seen IDs from this sync pass
          if (!currentDateFilter?.enabled) {
            pruneCatalogToIds(type, seenIds);
            if (type === 'photo') {
              pruneFeatureCacheFromUris(Array.from(seenIds)).catch(() => {});
            }
            setRawCache(type, getCatalogAssets(type));
          }

          const finalRaw = isPhoto ? photoRawCache : videoRawCache;
          const currentQueue = isPhoto ? get().photoQueue : get().videoQueue;
          const currentIndex = isPhoto ? get().photoIndex : get().videoIndex;
          const queue = rebuildLoadedQueue(
            currentQueue,
            currentIndex,
            finalRaw,
            type,
            currentSort,
            curPf,
            curVf,
            currentDateFilter,
          );
          set({
            [queueKey]: queue,
            [loadingMoreKey]: false,
            [loadingKey]: false,
            ...reviewStatsPartial(),
          });
        } else if (addedCount === 0) {
          return;
        } else {
          // Intermediate: keep the current card and, for random, the placed tail.
          const currentQueue = isPhoto ? get().photoQueue : get().videoQueue;
          const currentIndex = isPhoto ? get().photoIndex : get().videoIndex;
          const queue = rebuildLoadedQueue(
            currentQueue,
            currentIndex,
            rawCache,
            type,
            currentSort,
            curPf,
            curVf,
            currentDateFilter,
          );
          set({
            [queueKey]: queue,
            [loadingMoreKey]: true,
            [loadingKey]: false,
            ...reviewStatsPartial(),
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

  applyFilters: (type, sort, photoFilters, videoFilters) => {
    if (type === 'photo') {
      currentPhotoFilters = photoFilters;
    } else {
      currentVideoFilters = videoFilters;
    }
    get().applySort(type, sort);
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
    const loadingKey = isPhoto ? 'isLoadingPhotos' : 'isLoadingVideos';

    if (isPhoto && sort === 'ai') {
      set({
        photoQueue: [],
        photoIndex: 0,
        isLoadingPhotos: true,
        ...reviewStatsPartial(),
      });
      void ensureAIQueueFilled(set, get); // eslint-disable-line no-void
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
      [loadingKey]: false,
      ...reviewStatsPartial(),
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
      ...reviewStatsPartial(type),
    });

    if (type === 'photo') {
      captureTrainingExample(asset.id, asset.uri, 1);
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

    markAssetKept(asset.id, type);

    set({
      [indexKey]: index + 1,
      [stackKey]: [...currentStack, {type: 'keep', asset, queue: type}],
      ...reviewStatsPartial(type),
    });

    if (type === 'photo') {
      captureTrainingExample(asset.id, asset.uri, 0);
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

    if (type === 'photo') {
      await waitForTrainingCaptures(pending.map(p => p.id));
    }

    const uris = pending.map(p => p.uri);

    const countKey =
      type === 'photo' ? 'photoPendingCount' : 'videoPendingCount';
    const stackKey = type === 'photo' ? 'photoUndoStack' : 'videoUndoStack';

    if (Platform.OS === 'ios') {
      const result = await batchDeleteAssets(uris);
      if (result.success) {
        if (type === 'photo') {
          deleteCachedFeatures(uris).catch(() => {});
        }
        clearPendingDeletesByType(type);
        set({
          [countKey]: 0,
          [stackKey]: [],
          ...reviewStatsPartial(type),
        });
      } else {
        set({
          [countKey]: getPendingDeletesByType(type).length,
          ...reviewStatsPartial(type),
        });
      }
    } else {
      const deletedUris: string[] = [];
      for (const uri of uris) {
        if (await deleteAsset(uri)) {
          deletedUris.push(uri);
        }
      }
      if (type === 'photo' && deletedUris.length > 0) {
        deleteCachedFeatures(deletedUris).catch(() => {});
      }
      if (deletedUris.length === uris.length) {
        clearPendingDeletesByType(type);
        set({
          [countKey]: 0,
          [stackKey]: [],
          ...reviewStatsPartial(type),
        });
      } else {
        set({
          [countKey]: getPendingDeletesByType(type).length,
          ...reviewStatsPartial(type),
        });
      }
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
    if (isPhoto) {
      cancelTrainingExample(lastEntry.asset.id);
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
      ...reviewStatsPartial(type),
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
        isLoadingPhotos: true,
        ...reviewStatsPartial(),
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
      ...reviewStatsPartial(),
    });
  },

  onKeepHistoryReset: () => {
    set({
      photoUndoStack: getPersistedUndoStack('photo') as UndoEntry[],
      videoUndoStack: getPersistedUndoStack('video') as UndoEntry[],
      ...reviewStatsPartial(),
    });
  },
}));
