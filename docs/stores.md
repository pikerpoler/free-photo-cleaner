# Stores (Zustand)

Three stores, no Redux or Context. Cross-store access uses `getState()`.

## queueStore (`src/stores/queueStore.ts`)

The largest store (~509 lines). Manages photo and video queues, swipe actions, undo, and deletion.

### State Shape

```ts
{
  photoQueue: MediaAsset[];
  photoIndex: number;
  photoUndoStack: UndoEntry[];
  photoPendingCount: number;
  isLoadingPhotos: boolean;
  isLoadingPhotosMore: boolean;

  videoQueue: MediaAsset[];     // Same shape for videos
  videoIndex: number;
  videoUndoStack: UndoEntry[];
  videoPendingCount: number;
  isLoadingVideos: boolean;
  isLoadingVideosMore: boolean;
}
```

### Key Actions

| Action | What it does |
|--------|-------------|
| `loadQueue(type, sort, filters, limit?, dateFilter?)` | Progressive CameraRoll load → filter kept/pending → sort → set queue |
| `applySort(type, sort)` | Re-sort existing raw cache without reloading |
| `swipeLeft(type)` | Advance index, add to pending deletes (DB), train AI if photo |
| `swipeRight(type)` | Advance index, mark kept (DB), train AI if photo |
| `undo(type)` | Pop undo stack, revert index, remove from kept/pending |
| `flushDeletes(type)` | Delete assets via CameraRoll, clear pending, reset undo |

### Module-Level State (Outside Zustand)

```ts
let photoRawCache: MediaAsset[] = [];
let videoRawCache: MediaAsset[] = [];
let photoLoadController: ProgressiveLoadController | null;
let videoLoadController: ProgressiveLoadController | null;
```

These avoid re-renders during progressive loading. The raw cache holds ALL loaded assets before filtering; the queue in Zustand holds only the filtered/sorted subset.

### Progressive Loading Strategy

1. `loadMediaProgressively` paginates CameraRoll (100 assets/batch)
2. First batch: immediate queue update, sets `isLoading` false
3. Later batches: merge behind current index (frozen prefix)
4. Date/filter changes: cancel controller, full reload
5. Sort-only changes: re-sort from raw cache

---

## settingsStore (`src/stores/settingsStore.ts`)

Persists all user preferences to MMKV.

### State Shape

```ts
{
  photoSort: SortMode;          // 'dateNewest' | 'dateOldest' | 'sizeLargest' | 'sizeSmallest' | 'ai'
  videoSort: SortMode;          // Same minus 'ai'
  photoFilters: PhotoFilters;   // { screenshots: boolean, screenRecordings: boolean }
  videoFilters: VideoFilters;   // { screenRecordings: boolean }
  dateFilter: DateFilter;       // { startDate?: string, endDate?: string }
  theme: 'system' | 'light' | 'dark';  // Persisted but not applied (screens use useColorScheme)

  // AI settings
  trainAI: boolean;
  aiModelSize: AIModelSize;     // 'small' | 'medium' | 'large'
  aiBatchSize: number;          // 1–32
  aiLearningRate: number;       // 0.0001–0.01
}
```

### Persistence

Every setter calls `setSetting(key, value)` which writes JSON to MMKV. On app start, `loadSettings()` hydrates all fields from MMKV with defaults.

---

## aiStore (`src/stores/aiStore.ts`)

Manages the on-device AI classifier lifecycle.

### State Shape

```ts
{
  isModelLoaded: boolean;
  isClassifierLoaded: Record<AIModelSize, boolean>;
  trainingQueueSize: number;
}
```

### Key Actions

| Action | What it does |
|--------|-------------|
| `initialize()` | Load embedding model + classifier weights from disk |
| `trainOnSwipe(assetId, label)` | Enqueue training sample (0=keep, 1=delete) |
| `scoreAssets(assets)` | Predict P(delete) for each asset, return sorted |
| `resetModels()` | Delete all weight files, clear embedding cache |
| `flushTraining()` | Force-process remaining training queue |

### Cross-Store Access

- Reads `settingsStore.getState()` for `aiModelSize`, `aiBatchSize`, `aiLearningRate`, `trainAI`
- Called from `queueStore` on swipe actions
