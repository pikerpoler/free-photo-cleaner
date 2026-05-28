import {NativeModules, Platform, Alert} from 'react-native';
import {CameraRoll} from '@react-native-camera-roll/camera-roll';
import {StorageInfo} from '../types/media';

const {StorageInfoModule} = NativeModules;

let cachedStorageInfo: StorageInfo | null = null;
let cacheTimestamp = 0;
const CACHE_TTL_MS = 60_000;

export async function getNativeStorageInfo(): Promise<StorageInfo> {
  const now = Date.now();
  if (cachedStorageInfo && now - cacheTimestamp < CACHE_TTL_MS) {
    return cachedStorageInfo;
  }

  if (Platform.OS === 'ios' && StorageInfoModule) {
    try {
      const result = await StorageInfoModule.getStorageInfo();
      const info: StorageInfo = {
        totalSpace: result.totalSpace,
        freeSpace: result.freeSpace,
        photosSize: result.photosSize,
        videosSize: result.videosSize,
      };
      cachedStorageInfo = info;
      cacheTimestamp = now;
      return info;
    } catch (error: unknown) {
      console.warn(
        '[FreePhotoCleaner] getNativeStorageInfo failed:',
        error instanceof Error ? error.message : String(error),
      );
      return getFallbackStorageInfo();
    }
  }
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
  return {
    totalSpace: 128 * 1024 * 1024 * 1024,
    freeSpace: 32 * 1024 * 1024 * 1024,
    photosSize: 0,
    videosSize: 0,
  };
}
