# Architecture

## Layer Diagram

```
┌─────────────────────────────────────────────┐
│  Screens (PhotosScreen, VideosScreen,       │
│           SettingsScreen)                    │
├─────────────────────────────────────────────┤
│  Components (SwipeCard, MediaCard,          │
│              VideoCard, StorageBar)          │
├─────────────────────────────────────────────┤
│  Stores (queueStore, settingsStore, aiStore)│
├──────────────┬──────────────┬───────────────┤
│  Services    │  AI Module   │  Utils        │
│  media.ts    │  classifier  │  format.ts    │
│  database.ts │  trainer     │               │
│  storage.ts  │  embeddings  │               │
│  nativeStor. │  weights     │               │
├──────────────┴──────────────┴───────────────┤
│  Native: CameraRoll, op-sqlite, MMKV,      │
│          react-native-fs, StorageInfoModule │
└─────────────────────────────────────────────┘
```

## Tech Stack

| Layer            | Technology                                    |
|------------------|-----------------------------------------------|
| Framework        | React Native 0.85.3, React 19.2.3             |
| Language         | TypeScript (strict)                            |
| Navigation       | React Navigation 7 (bottom tabs, 3 screens)   |
| State            | Zustand 5 (3 stores)                           |
| Animation        | react-native-reanimated 4, gesture-handler     |
| Persistence      | MMKV (settings), op-sqlite (data), FS (AI)     |
| Media access     | @react-native-camera-roll/camera-roll          |
| Video playback   | react-native-video                             |
| AI               | Custom JS MLP + hash embeddings (no server)    |
| Min Node         | >= 22.11.0                                     |
| iOS architecture | New Architecture enabled (Fabric + TurboModules)|

## Navigation Structure

```
App (GestureHandlerRootView)
└── AppNavigator (NavigationContainer)
    └── BottomTabNavigator
        ├── "Photos"   → PhotosScreen
        ├── "Videos"   → VideosScreen
        └── "Settings" → SettingsScreen
```

No stack navigators. No deep linking. Tab bar uses labels only (no icons).

## Persistence Strategy

| Data type       | Storage     | Reason                          |
|-----------------|-------------|---------------------------------|
| User settings   | MMKV        | Fast sync reads on app start    |
| Kept asset IDs  | SQLite      | Queryable set, survives reinstall|
| Pending deletes | SQLite      | Transactional, batch operations |
| Embedding cache | SQLite      | Binary blobs, keyed by asset ID |
| AI weights      | Filesystem  | Large binary, save/load by name |

## Platform Differences

| Feature           | iOS                                    | Android                          |
|-------------------|----------------------------------------|----------------------------------|
| Batch delete      | `CameraRoll.deletePhotos` (system UI)  | Sequential `deleteAsset` per URI |
| Storage info      | Native `StorageInfoModule` (Swift)     | Fallback "unavailable" message   |
| Privacy manifest  | `PrivacyInfo.xcprivacy` included       | N/A                              |
| Permissions       | Photo Library (Info.plist)             | READ_MEDIA_IMAGES/VIDEO          |

## Key Design Decisions

1. **Two-card deck pattern** in SwipeCard: two permanent Animated.Views alternate top/behind roles. Content only changes on the hidden card. Eliminates flash-on-swipe by avoiding any content change on visible views.

2. **Progressive loading**: CameraRoll is paginated (100/batch). First batch renders immediately; later batches merge behind the current swipe position so the user's place isn't disrupted.

3. **Module-level caches** in queueStore: raw asset arrays and load controllers live outside Zustand state to avoid re-renders during background loading.

4. **AI is fully offline**: no network calls. Hash-based embeddings today (576-dim from URI string). Placeholder for MobileNet ONNX in `assets/models/`.
