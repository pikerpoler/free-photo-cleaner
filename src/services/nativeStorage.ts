import {NativeModules, Platform, Alert} from 'react-native';
import {CameraRoll} from '@react-native-camera-roll/camera-roll';
import {StorageInfo} from '../types/media';
import {getSetting, setSetting, KEYS} from './storage';

const {StorageInfoModule} = NativeModules;

let cachedStorageInfo: StorageInfo | null = null;
let cacheTimestamp = 0;
const CACHE_TTL_MS = 60_000;

function persistLibraryStats(info: StorageInfo): void {
  if (info.photoCount != null) {
    setSetting(KEYS.LIBRARY_PHOTO_COUNT, info.photoCount);
  }
  if (info.videoCount != null) {
    setSetting(KEYS.LIBRARY_VIDEO_COUNT, info.videoCount);
  }
  setSetting(KEYS.LIBRARY_FREE_SPACE, info.freeSpace);
  setSetting(KEYS.LIBRARY_TOTAL_SPACE, info.totalSpace);
  setSetting(KEYS.LIBRARY_PHOTOS_SIZE, info.photosSize);
  setSetting(KEYS.LIBRARY_VIDEOS_SIZE, info.videosSize);
}

export function getCachedLibraryStats(): StorageInfo | null {
  const freeSpace = getSetting<number | null>(KEYS.LIBRARY_FREE_SPACE, null);
  if (freeSpace == null) return null;
  return {
    totalSpace: getSetting(KEYS.LIBRARY_TOTAL_SPACE, 0),
    freeSpace,
    photosSize: getSetting(KEYS.LIBRARY_PHOTOS_SIZE, 0),
    videosSize: getSetting(KEYS.LIBRARY_VIDEOS_SIZE, 0),
    photoCount: getSetting(KEYS.LIBRARY_PHOTO_COUNT, 0),
    videoCount: getSetting(KEYS.LIBRARY_VIDEO_COUNT, 0),
    isFallback: false,
  };
}

export async function getNativeStorageInfo(): Promise<StorageInfo> {
  const now = Date.now();
  if (cachedStorageInfo && now - cacheTimestamp < CACHE_TTL_MS) {
    return cachedStorageInfo;
  }

  // Instant paint from last persisted values while refreshing
  const persisted = getCachedLibraryStats();

  if (Platform.OS === 'ios' && StorageInfoModule) {
    try {
      const result = await StorageInfoModule.getStorageInfo();
      const info: StorageInfo = {
        totalSpace: result.totalSpace,
        freeSpace: result.freeSpace,
        photosSize: result.photosSize,
        videosSize: result.videosSize,
        photoCount: result.photoCount,
        videoCount: result.videoCount,
        isFallback: false,
      };
      cachedStorageInfo = info;
      cacheTimestamp = now;
      persistLibraryStats(info);
      return info;
    } catch (error: unknown) {
      console.warn(
        '[FreePhotoCleaner] getNativeStorageInfo failed:',
        error instanceof Error ? error.message : String(error),
      );
      if (persisted) return persisted;
      return getFallbackStorageInfo();
    }
  }
  if (persisted) return persisted;
  return getFallbackStorageInfo();
}

export function invalidateStorageCache(): void {
  cachedStorageInfo = null;
  cacheTimestamp = 0;
}

export async function batchDeleteAssets(uris: string[]): Promise<{success: boolean; deletedCount: number}> {
  if (uris.length === 0) {
    return {success: true, deletedCount: 0};
  }

  try {
    await CameraRoll.deletePhotos(uris);
    invalidateStorageCache();
    return {success: true, deletedCount: uris.length};
  } catch (error: unknown) {
    const msg = error instanceof Error ? error.message : String(error);
    if (msg.includes('cancelled') || msg.includes('denied')) {
      Alert.alert('Deletion Cancelled', 'You declined the deletion prompt. Items remain in your pending queue.');
    } else {
      Alert.alert('Delete Error', `Failed to delete: ${msg}`);
    }
    return {success: false, deletedCount: 0};
  }
}

function getFallbackStorageInfo(): StorageInfo {
  console.warn('[FreePhotoCleaner] Using FALLBACK storage info — native module unavailable or errored');
  return {
    totalSpace: 0,
    freeSpace: 0,
    photosSize: 0,
    videosSize: 0,
    photoCount: 0,
    videoCount: 0,
    isFallback: true,
  };
}
