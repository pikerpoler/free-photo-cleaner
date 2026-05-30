# Components

## SwipeCard (`src/components/SwipeCard.tsx`)

The core interactive component. Implements a **two-card deck pattern** where two permanent `Animated.View` slots alternate between "top" (interactive) and "behind" (preview) roles.

### Props

```ts
interface SwipeCardProps {
  children: React.ReactNode;       // Content for the current/top card
  behindContent?: React.ReactNode; // Content for the next/behind card
  onSwipeLeft: () => void;         // Called after left swipe completes
  onSwipeRight: () => void;        // Called after right swipe completes
  cardKey?: string;                // Asset ID (accepted but unused internally)
  enabled?: boolean;               // Disable gesture (default: true)
}
```

### Internal Architecture

**Two permanent slots** (Slot 0 and Slot 1), both always in the React tree:

```
topSlot === 0:  Slot0 = children (top),  Slot1 = behindContent (behind)
topSlot === 1:  Slot0 = behindContent,   Slot1 = children (top)
```

**Shared values** (two sets, one per slot):
- `translateX0/Y0`, `translateX1/Y1` — position when acting as the top card
- `topSlotSV` — shared value (UI thread) tracking which slot is top
- `topSlot` — React state mirroring `topSlotSV` for content mapping
- `isAnimatingExit` — prevents gesture during exit animation

**Gesture**: Single `Gesture.Pan()` on a wrapper `Animated.View`. Reads `topSlotSV.value` in worklet callbacks to modify the correct slot's shared values.

**Exit animation flow**:
1. `withTiming` animates top slot's translateX to `EXIT_DISTANCE` (1.3× screen width)
2. Completion callback (UI thread): flips `topSlotSV`, clears `isAnimatingExit`
3. `runOnJS(advanceIndex)`: resets exiting slot's position, calls `onSwipeLeft/Right`, flips `topSlot` state

**Why this eliminates flash**: The visible card's content never changes during the transition. The slot that was behind (already showing the next photo) becomes top without any Image reload. Content only changes on the now-hidden slot.

**Animated styles per slot**: Each slot's `useAnimatedStyle` checks `topSlotSV` to return either:
- **Top card style**: translateX/Y, rotation, border color/width, zIndex: 2
- **Behind card style**: scale (0.92→1), opacity (0.6→1) based on the OTHER slot's translateX, zIndex: 1

**Indicators**: DELETE/KEEP overlays rendered inside both slots. Hidden (opacity: 0) when the slot is behind.

### Constants

| Constant          | Value              | Purpose                    |
|-------------------|--------------------|----------------------------|
| `SWIPE_THRESHOLD` | 25% of screen width| Min drag to trigger swipe   |
| `EXIT_DISTANCE`   | 130% of screen width| Off-screen animation target|
| `EXIT_DURATION_MS` | 350ms             | Exit animation duration     |

---

## MediaCard (`src/components/MediaCard.tsx`)

Displays a photo with metadata. Wrapped in `React.memo` with `asset.id` comparison.

### Props

```ts
interface MediaCardProps {
  asset: MediaAsset;
}
```

### Renders

- `Image` with `key={asset.id}` (prevents native image caching across asset swaps)
- Metadata strip: date, file size, resolution, duration (if video)

---

## VideoCard (`src/components/VideoCard.tsx`)

Video player card with tap-to-mute. Not memoized.

### Props

```ts
interface VideoCardProps {
  asset: MediaAsset;
  isActive?: boolean; // Controls playback (default: true)
}
```

### Behavior

- Auto-plays, loops, starts muted
- Tap toggles mute (with mute icon overlay)
- Pauses when `isActive` is false
- Same metadata strip as MediaCard

---

## StorageBar (`src/components/StorageBar.tsx`)

Segmented bar showing device storage breakdown.

### Behavior

- Calls `getNativeStorageInfo()` on mount
- Shows: Photos (blue), Videos (purple), Other (gray), Free (green)
- Displays total and free space in GB
- Falls back to "Storage info unavailable" on Android
