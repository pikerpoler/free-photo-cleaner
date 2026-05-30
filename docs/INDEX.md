# AI Photo Cleaner — Codebase Documentation

> **For AI agents**: Read this file first. It tells you which doc to open based on your task. Each doc is self-contained — don't read them all, pick the one(s) relevant to your task.

## Quick Orientation

Swipe-based iOS/Android photo & video cleanup app. Users swipe left (delete) or right (keep). Deletions are batched until confirmed. An on-device AI learns preferences and re-sorts photos by likelihood of deletion.

**Tech**: React Native 0.85, TypeScript, Zustand, Reanimated 4, Gesture Handler, op-sqlite, MMKV, CameraRoll.

**24 source files**, ~3,900 lines total in `src/`.

## Doc Routing — Read Based on Your Task

| If you need to...                                  | Read                                      |
|----------------------------------------------------|-------------------------------------------|
| Understand overall architecture & patterns         | [architecture.md](architecture.md)        |
| Work on the swipe card UI or animations            | [components.md](components.md)            |
| Work on screens (Photos, Videos, Settings)         | [screens.md](screens.md)                  |
| Understand or modify state management              | [stores.md](stores.md)                    |
| Understand photo/video loading, deletion, storage  | [services.md](services.md)               |
| Work on the AI classifier, training, or embeddings | [ai.md](ai.md)                            |
| Build, deploy, or debug build issues               | [build.md](build.md)                      |
| Trace a full feature flow end-to-end               | [data-flows.md](data-flows.md)            |

## File Map

```
src/
├── ai/                     # On-device ML classifier
│   ├── index.ts            #   Barrel re-export
│   ├── types.ts            #   TrainingEntry, weights, EMBEDDING_DIM=576
│   ├── classifier.ts       #   MLP forward/backward/predict (BCE + SGD)
│   ├── embeddingModel.ts   #   Hash-based embeddings (ONNX placeholder)
│   ├── embeddingCache.ts   #   SQLite embedding blob cache
│   ├── trainer.ts          #   Batch SGD via InteractionManager
│   └── weights.ts          #   Save/load weights to filesystem
├── components/
│   ├── SwipeCard.tsx       #   Two-card deck swipe with Reanimated
│   ├── MediaCard.tsx       #   Photo image + metadata strip
│   ├── VideoCard.tsx       #   Video player + metadata strip
│   └── StorageBar.tsx      #   Device storage segmented bar
├── navigation/
│   └── AppNavigator.tsx    #   3-tab bottom navigator
├── screens/
│   ├── PhotosScreen.tsx    #   Photo swipe queue + delete/undo
│   ├── VideosScreen.tsx    #   Video swipe queue + delete/undo
│   └── SettingsScreen.tsx  #   Filters, sort, AI config, resets
├── services/
│   ├── media.ts            #   CameraRoll fetch, progressive load, delete
│   ├── database.ts         #   SQLite: kept_assets, pending_deletes, embedding_cache
│   ├── storage.ts          #   MMKV wrapper for settings
│   └── nativeStorage.ts    #   iOS native storage info + batch delete
├── stores/
│   ├── queueStore.ts       #   Photo/video queues, swipe, undo, flush
│   ├── settingsStore.ts    #   Sort, filters, AI config (MMKV-backed)
│   └── aiStore.ts          #   Classifier weights, scoring, training
├── types/
│   └── media.ts            #   MediaAsset, SortMode, filters, StorageInfo
└── utils/
    └── format.ts           #   formatFileSize, formatDate, etc.
```

## Key Conventions

- **Named exports** everywhere (no default exports except `App.tsx`)
- **Zustand** stores with `create()`, cross-store via `getState()`
- **Persistence**: MMKV for settings, op-sqlite for operational data, react-native-fs for AI weights
- **Dark mode**: `useColorScheme()` per component (no centralized theme)
- **Memoization**: `React.memo` with id-based comparators on media cards
- **Path alias**: `@/*` → `src/*` in tsconfig (not widely used in imports)
