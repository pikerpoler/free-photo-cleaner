import {MediaAsset, MediaType} from '../src/types/media';
import type {OnBatchLoaded} from '../src/services/media';

const mockLoadMediaProgressively = jest.fn();
const mockGetCatalogAssets = jest.fn();
const mockAddPendingDelete = jest.fn();
const mockMarkAssetKept = jest.fn();

jest.mock('../src/services/media', () => ({
  loadMediaProgressively: (...args: unknown[]) =>
    mockLoadMediaProgressively(...args),
  deleteAsset: jest.fn(),
  stampLocalAvailability: jest.fn(async (assets: MediaAsset[]) => assets),
  applyCachedLocalFlags: jest.fn((assets: MediaAsset[]) => assets),
}));

jest.mock('../src/services/nativeStorage', () => ({
  batchDeleteAssets: jest.fn(),
}));

jest.mock('../src/services/database', () => ({
  getKeptAssetIds: jest.fn(() => new Set<string>()),
  markAssetKept: (...args: unknown[]) => mockMarkAssetKept(...args),
  removeAssetFromKept: jest.fn(),
  getPendingDeletesByType: jest.fn(() => []),
  addPendingDelete: (...args: unknown[]) => mockAddPendingDelete(...args),
  removePendingDelete: jest.fn(),
  clearPendingDeletesByType: jest.fn(),
  getPendingDeleteIds: jest.fn(() => new Set<string>()),
  upsertCatalogAssets: jest.fn(),
  getCatalogAssets: (...args: unknown[]) => mockGetCatalogAssets(...args),
  updateCatalogLocalFlags: jest.fn(),
  pruneCatalogToIds: jest.fn(),
  getPersistedUndoStack: jest.fn(() => []),
  getPendingDeleteAggregate: jest.fn(() => ({count: 0, totalBytes: 0})),
  getKeptAggregate: jest.fn(() => ({count: 0, totalBytes: 0})),
}));

jest.mock('../src/stores/aiStore', () => ({
  useAIStore: {
    getState: () => ({
      hasModel: false,
      activeModelInfo: null,
      scoreById: {},
      scoreAssets: jest.fn(async () => {}),
      clearScores: jest.fn(),
      clearScoresForIds: jest.fn(),
      refreshModelStatus: jest.fn(async () => {}),
    }),
  },
}));

jest.mock('../src/services/cnnTrainer', () => ({
  deleteCachedFeatures: jest.fn(async () => {}),
  pruneFeatureCacheFromUris: jest.fn(async () => {}),
}));

jest.mock('../src/services/trainingCapture', () => ({
  cancelTrainingExample: jest.fn(),
  captureTrainingExample: jest.fn(),
  waitForTrainingCaptures: jest.fn(async () => {}),
}));

jest.mock('../src/services/storage', () => ({
  addLifetimeDeletedStats: jest.fn(),
  getLifetimeDeletedStats: jest.fn(() => ({count: 0, bytes: 0})),
}));

import {useQueueStore} from '../src/stores/queueStore';

function makeAsset(id: string, type: MediaType): MediaAsset {
  return {
    id,
    uri: id,
    type,
    filename: `${id}.mp4`,
    width: 1920,
    height: 1080,
    fileSize: 10_000_000,
    creationDate: Date.now(),
    duration: 12,
    hasExif: true,
    isScreenshot: false,
    isWhatsApp: false,
  };
}

describe('queueStore video loading after photo load', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockGetCatalogAssets.mockReturnValue([]);
  });

  test('loads videos on first launch after photos have incremented photoLoadGeneration', () => {
    const callbacks = new Map<MediaType, OnBatchLoaded>();
    mockLoadMediaProgressively.mockImplementation(
      (type: MediaType, onBatch: OnBatchLoaded) => {
        callbacks.set(type, onBatch);
        return {cancel: jest.fn(), pause: jest.fn(), resume: jest.fn()};
      },
    );

    // 1. Photos tab mounts first on launch, incrementing photoLoadGeneration.
    useQueueStore.getState().loadQueue('photo', 'ai');
    expect(callbacks.has('photo')).toBe(true);

    // 2. User switches to Videos tab on first launch (empty catalog).
    useQueueStore.getState().loadQueue('video', 'largest_first');
    expect(useQueueStore.getState().isLoadingVideos).toBe(true);
    expect(callbacks.has('video')).toBe(true);

    // 3. First video batch arrives and clears the loading spinner.
    const video = makeAsset('ph://video-1', 'video');
    callbacks.get('video')?.([video], true);

    const state = useQueueStore.getState();
    expect(state.isLoadingVideos).toBe(false);
    expect(state.isLoadingVideosMore).toBe(false);
    expect(state.videoQueue).toHaveLength(1);
    expect(state.videoQueue[0].id).toBe('ph://video-1');
  });

  test('loading videos does not clear isScoringPhotos while photos are scoring', () => {
    const callbacks = new Map<MediaType, OnBatchLoaded>();
    mockLoadMediaProgressively.mockImplementation(
      (type: MediaType, onBatch: OnBatchLoaded) => {
        callbacks.set(type, onBatch);
        return {cancel: jest.fn(), pause: jest.fn(), resume: jest.fn()};
      },
    );

    useQueueStore.setState({isScoringPhotos: true});
    useQueueStore.getState().loadQueue('video', 'largest_first');
    expect(useQueueStore.getState().isScoringPhotos).toBe(true);

    callbacks.get('video')?.([makeAsset('ph://video-2', 'video')], true);
    expect(useQueueStore.getState().isScoringPhotos).toBe(true);
  });
});

describe('queueStore failed image handling', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    useQueueStore.setState({
      photoQueue: [],
      photoIndex: 0,
      photoPendingCount: 0,
      photoUndoStack: [],
    });
  });

  test('removes a failed current photo without keeping or deleting it', () => {
    const failed = makeAsset('ph://failed-current', 'photo');
    const next = makeAsset('ph://next-after-failure', 'photo');
    useQueueStore.setState({photoQueue: [failed, next], photoIndex: 0});

    useQueueStore.getState().skipFailedAsset('photo', failed.id);

    const state = useQueueStore.getState();
    expect(state.photoQueue.map(asset => asset.id)).toEqual([next.id]);
    expect(state.photoIndex).toBe(0);
    expect(state.photoUndoStack).toHaveLength(0);
    expect(state.photoPendingCount).toBe(0);
    expect(mockAddPendingDelete).not.toHaveBeenCalled();
    expect(mockMarkAssetKept).not.toHaveBeenCalled();
  });

  test('removes a failed preloaded photo without moving the current photo', () => {
    const current = makeAsset('ph://current-before-failure', 'photo');
    const failedNext = makeAsset('ph://failed-next', 'photo');
    const after = makeAsset('ph://after-failed-next', 'photo');
    useQueueStore.setState({
      photoQueue: [current, failedNext, after],
      photoIndex: 0,
    });

    useQueueStore.getState().skipFailedAsset('photo', failedNext.id);

    const state = useQueueStore.getState();
    expect(state.photoQueue.map(asset => asset.id)).toEqual([
      current.id,
      after.id,
    ]);
    expect(state.photoIndex).toBe(0);
    expect(state.getCurrentAsset('photo')?.id).toBe(current.id);
    expect(mockAddPendingDelete).not.toHaveBeenCalled();
    expect(mockMarkAssetKept).not.toHaveBeenCalled();
  });

  test('ignores a stale swipe callback for an image that was skipped', () => {
    const failed = makeAsset('ph://failed-during-gesture', 'photo');
    const next = makeAsset('ph://gesture-next', 'photo');
    useQueueStore.setState({photoQueue: [failed, next], photoIndex: 0});

    useQueueStore.getState().skipFailedAsset('photo', failed.id);
    useQueueStore.getState().swipeLeft('photo', failed.id);
    useQueueStore.getState().swipeRight('photo', failed.id);

    expect(useQueueStore.getState().getCurrentAsset('photo')?.id).toBe(next.id);
    expect(mockAddPendingDelete).not.toHaveBeenCalled();
    expect(mockMarkAssetKept).not.toHaveBeenCalled();
  });
});
