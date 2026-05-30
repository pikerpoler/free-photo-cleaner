# Services

## media.ts (`src/services/media.ts`)

CameraRoll access layer. Handles pagination, progressive loading, and deletion.

### Exports

| Export | Signature | Purpose |
|--------|-----------|---------|
| `fetchMediaAssets` | `(type, limit?, after?, dateFilter?) → {assets, endCursor, hasMore}` | Single-page CameraRoll fetch (100 items default) |
| `loadMediaProgressively` | `(type, onBatch, dateFilter?) → ProgressiveLoadController` | Paginate all assets, deliver in batches |
| `deleteAsset` | `(uri) → boolean` | Delete single asset via CameraRoll (Android) |

### MediaAsset Mapping

CameraRoll `PhotoIdentifier` is mapped to `MediaAsset`:
```ts
{
  id: string;            // node.image.uri
  uri: string;           // node.image.uri (ph:// on iOS)
  filename: string;
  creationDate: string;  // ISO date
  width: number;
  height: number;
  fileSize: number;      // bytes
  duration?: number;     // seconds (videos)
  type: 'photo' | 'video';
  group_name: string[];  // album names
}
```

### Screenshot/Screen Recording Detection

`isScreenshot(asset)` checks: filename contains "screenshot", OR group_name includes "Screenshots", OR dimensions match known screen sizes (iPhone/iPad/Android).

`isScreenRecording(asset)` checks: group_name includes "Screen Recordings".

---

## database.ts (`src/services/database.ts`)

SQLite (op-sqlite) for operational data. Single database instance.

### Tables

| Table | Columns | Purpose |
|-------|---------|---------|
| `kept_assets` | `id TEXT PRIMARY KEY, kept_at TEXT` | Assets user swiped right on |
| `pending_deletes` | `id TEXT PRIMARY KEY, type TEXT, uri TEXT, added_at TEXT` | Assets pending batch deletion |
| `embedding_cache` | `asset_id TEXT PRIMARY KEY, embedding BLOB` | Cached AI embeddings |

### Exports

| Export | Purpose |
|--------|---------|
| `getDatabase()` | Initialize DB, create tables, return instance |
| `markAssetKept(id)` | Insert into kept_assets |
| `isAssetKept(id)` | Check if exists in kept_assets |
| `getKeptAssetIds()` | Return all kept IDs as Set |
| `addPendingDelete(id, type, uri)` | Insert into pending_deletes |
| `getPendingDeletesByType(type)` | Get all pending for type |
| `getPendingDeleteIds()` | Return all pending IDs as Set |
| `removePendingDelete(id)` | Remove single pending |
| `clearPendingDeletesByType(type)` | Clear all pending for type |
| `resetKeepHistory()` | Drop and recreate kept_assets |

---

## storage.ts (`src/services/storage.ts`)

MMKV wrapper for user settings.

### Exports

```ts
const storage: MMKV;                        // Singleton MMKV instance
function getSetting<T>(key: string): T | undefined;  // JSON.parse from MMKV
function setSetting(key: string, value: any): void;   // JSON.stringify to MMKV
const KEYS: Record<string, string>;          // Setting key constants
```

### Keys

`photoSort`, `videoSort`, `photoFilters`, `videoFilters`, `dateFilter`, `theme`, `trainAI`, `aiModelSize`, `aiBatchSize`, `aiLearningRate`

---

## nativeStorage.ts (`src/services/nativeStorage.ts`)

iOS native module wrapper for storage info and batch deletion.

### Exports

| Export | Platform | Purpose |
|--------|----------|---------|
| `getNativeStorageInfo()` | iOS only | Returns `StorageInfo` with totalSpace, freeSpace, photoSize, videoSize |
| `batchDeleteAssets(uris)` | iOS only | Calls `CameraRoll.deletePhotos(uris)` with system confirmation dialog |
| `invalidateStorageCache()` | iOS only | Clears cached storage info |

On Android, `getNativeStorageInfo()` returns null and `batchDeleteAssets` falls back to sequential deletion.
