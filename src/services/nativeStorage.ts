import {NativeModules, Platform, Alert} from 'react-native';
import {CameraRoll} from '@react-native-camera-roll/camera-roll';
import {StorageInfo} from '../types/media';

const {StorageInfoModule} = NativeModules;

export async function getNativeStorageInfo(): Promise<StorageInfo> {
  if (Platform.OS === 'ios' && StorageInfoModule) {
    try {
      const result = await StorageInfoModule.getStorageInfo();
      return {
        totalSpace: result.totalSpace,
        freeSpace: result.freeSpace,
        photosSize: result.photosSize,
        videosSize: result.videosSize,
      };
    } catch {
      return getFallbackStorageInfo();
    }
  }
  return getFallbackStorageInfo();
}

export async function batchDeleteAssets(uris: string[]): Promise<{success: boolean; deletedCount: number}> {
  if (uris.length === 0) {
    return {success: true, deletedCount: 0};
  }

  try {
    await CameraRoll.deletePhotos(uris);
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
