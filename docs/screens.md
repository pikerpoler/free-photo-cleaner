# Screens

## PhotosScreen (`src/screens/PhotosScreen.tsx`)

Main screen. Shows a swipeable photo card stack with delete/undo controls.

### State Sources

- `useQueueStore`: `photoQueue`, `photoIndex`, loading flags, `swipeLeft/Right`, `undo`, `flushDeletes`, undo stack, pending count
- `useSettingsStore`: `photoSort`, `photoFilters`, `dateFilter`

### Lifecycle

1. On mount: calls `loadQueue('photo', sort, filters, dateFilter)`
2. On filter/date change: reloads queue
3. On sort-only change: calls `applySort('photo', sort)` (no reload)

### Renders

| State              | UI                                              |
|--------------------|-------------------------------------------------|
| Loading            | Spinner + "Loading photos..."                   |
| Loading more       | Spinner + "Loading more photos..."               |
| No current asset   | "All caught up!" + delete button if pending > 0 |
| Has current asset  | Header (remaining count, undo, delete) + SwipeCard + StorageBar |

### SwipeCard Usage

```tsx
<SwipeCard
  onSwipeLeft={handleSwipeLeft}    // → swipeLeft('photo')
  onSwipeRight={handleSwipeRight}  // → swipeRight('photo')
  cardKey={currentAsset.id}
  behindContent={nextAsset ? <MediaCard asset={nextAsset} /> : undefined}>
  <MediaCard asset={currentAsset} />
</SwipeCard>
```

### Batch Delete Warning

Shows an alert if `photoPendingCount >= 1024` warning about system timeouts.

---

## VideosScreen (`src/screens/VideosScreen.tsx`)

Same pattern as PhotosScreen but for videos.

### Differences from PhotosScreen

- Uses `videoQueue`, `videoIndex`, `videoSort`, `videoFilters`
- Uses `VideoCard` instead of `MediaCard` for current asset
- Uses `MediaCard` for the behind card (thumbnail only, no playback)
- No AI training on swipe
- `VideoCard` gets `isActive` prop based on navigation focus

---

## SettingsScreen (`src/screens/SettingsScreen.tsx`)

Configuration screen with sections for photos, videos, and AI.

### Sections

**Photo Settings**:
- Sort mode picker: Date (newest/oldest), Size (largest/smallest), AI
- Filters: Screenshots, screen recordings (toggles)
- Date range: start/end date pickers

**Video Settings**:
- Sort mode picker (no AI option)
- Filters: Screen recordings (toggle)
- Same date range

**AI Settings** (only visible when AI sort is selected for photos):
- Train on swipe toggle
- Model size: small/medium/large
- Batch size slider: 1–32
- Learning rate slider: 0.0001–0.01
- Reset keep history button
- Reset AI models button

### State

All settings go through `useSettingsStore` setters, which persist to MMKV immediately.
