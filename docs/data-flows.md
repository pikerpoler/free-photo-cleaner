# Data Flows

End-to-end traces of the major features.

## 1. Photo/Video Loading

```
PhotosScreen mounts
  → useEffect: loadQueue('photo', sort, filters, dateFilter)
    → queueStore.loadQueue()
      → media.loadMediaProgressively(type, onBatch, dateFilter)
        → CameraRoll.getPhotos(first: 100, after: cursor)
          → onBatch(assets, isDone)
            → database.getKeptAssetIds()
            → database.getPendingDeleteIds()
            → Filter: remove kept + pending + apply filters
            → Sort: by date/size/AI
            → Set photoQueue + photoIndex
        → (repeat for next pages until done)
      → If AI sort: aiStore.scoreAssets() → re-sort
```

**Timing**: First batch renders immediately (loading spinner dismissed). Later batches merge behind the current swipe index.

**Filter/sort changes**:
- Filter or date change → full reload (`loadQueue` again)
- Sort-only change → `applySort` re-sorts from module-level raw cache

## 2. Swipe (Keep / Delete)

### Right swipe (Keep)

```
User swipes right
  → SwipeCard: exit animation on UI thread
    → withTiming callback: flip topSlotSV, runOnJS(advanceIndex)
      → advanceIndex: reset exiting slot, call onSwipeRight(), flip topSlot
        → PhotosScreen.handleSwipeRight()
          → queueStore.swipeRight('photo')
            → database.markAssetKept(id)
            → Push to undoStack
            → Advance photoIndex
            → If trainAI: aiStore.trainOnSwipe(id, 0)  // 0 = keep
```

### Left swipe (Delete)

```
Same animation flow, then:
  → queueStore.swipeLeft('photo')
    → database.addPendingDelete(id, type, uri)
    → Increment photoPendingCount
    → Push to undoStack
    → Advance photoIndex
    → If trainAI: aiStore.trainOnSwipe(id, 1)  // 1 = delete
```

### Undo

```
User taps Undo
  → queueStore.undo('photo')
    → Pop undoStack
    → Decrement photoIndex
    → If was delete: database.removePendingDelete(id), decrement pendingCount
    → If was keep: remove from kept_assets
    → Restore queue to pre-swipe state
```

## 3. Batch Deletion

```
User taps "Delete N"
  → PhotosScreen.handleDelete()
    → If pendingCount >= 1024: show warning alert
    → queueStore.flushDeletes('photo')
      → If trainAI && trainingQueue > 0: show train/skip/wait alert
      → database.getPendingDeletesByType('photo') → get URIs
      → iOS: nativeStorage.batchDeleteAssets(uris)
             → CameraRoll.deletePhotos(uris) (system confirmation)
      → Android: sequential deleteAsset(uri) per URI
      → database.clearPendingDeletesByType('photo')
      → Reset pendingCount, undo stack
      → nativeStorage.invalidateStorageCache()
```

## 4. AI Training Pipeline

```
Swipe triggers trainOnSwipe(assetId, label)
  → aiStore.trainOnSwipe()
    → trainer.enqueueTraining({assetId, label})
      → Queue grows
      → When queue.length >= batchSize:
        → InteractionManager.runAfterInteractions:
          → For each sample in batch:
            → embeddingCache.getCachedEmbedding(id) || embeddingModel.computeEmbedding(id)
            → classifier.forward(weights, embedding)
            → classifier.backward(weights, embedding, hidden, output, label)
            → Accumulate gradients
          → Average gradients
          → classifier.sgdStep(weights, avgGradients, learningRate)
          → weights.saveWeights(modelName, weights)
```

## 5. Settings Change

```
User changes setting in SettingsScreen
  → settingsStore.setSomething(value)
    → storage.setSetting(key, value) → MMKV write
    → Zustand state update → subscribers re-render
  → PhotosScreen re-renders
    → useEffect detects change
      → If filter/date changed: loadQueue() (full reload)
      → If sort-only changed: applySort() (re-sort cached)
```
