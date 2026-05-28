import {Platform} from 'react-native';
import {
  CameraRoll,
  PhotoIdentifier,
} from '@react-native-camera-roll/camera-roll';
import {MediaAsset, MediaType} from '../types/media';

const BATCH_SIZE = 100;

function detectScreenshot(asset: PhotoIdentifier): boolean {
  const filename = asset.node.image.filename || '';
  const lower = filename.toLowerCase();

  if (Platform.OS === 'ios') {
    return lower.includes('screenshot') || /^img_\d+\.png$/i.test(filename);
  }
  return (
    lower.includes('screenshot') ||
    lower.startsWith('screen_') ||
    filename.includes('Screenshot_')
  );
}

function detectWhatsApp(asset: PhotoIdentifier): boolean {
  const filename = asset.node.image.filename || '';
  const lower = filename.toLowerCase();

  if (Platform.OS === 'android') {
    return lower.includes('whatsapp') || /^img-\d+-wa\d+/i.test(filename);
  }
  // iOS: WhatsApp-saved images often lack distinctive names,
  // but some saved from WhatsApp have specific patterns
  return lower.includes('whatsapp') || /^img-\d+-wa/i.test(lower);
}

function hasMetadata(asset: PhotoIdentifier): boolean {
  return asset.node.timestamp > 0;
}

function photoIdentifierToMediaAsset(
  asset: PhotoIdentifier,
  type: MediaType,
): MediaAsset {
  const node = asset.node;
  return {
    id: node.image.uri,
    uri: node.image.uri,
    type,
    filename: node.image.filename || 'unknown',
    width: node.image.width || 0,
    height: node.image.height || 0,
    fileSize: node.image.fileSize || 0,
    creationDate: node.timestamp * 1000,
    duration: node.image.playableDuration || undefined,
    hasExif: hasMetadata(asset),
    isScreenshot: detectScreenshot(asset),
    isWhatsApp: detectWhatsApp(asset),
  };
}

export async function fetchMediaAssets(
  type: MediaType,
  cursor?: string,
): Promise<{assets: MediaAsset[]; endCursor: string | undefined; hasMore: boolean}> {
  const assetType = type === 'photo' ? 'Photos' : 'Videos';

  const result = await CameraRoll.getPhotos({
    first: BATCH_SIZE,
    after: cursor,
    assetType,
    include: ['filename', 'fileSize', 'imageSize', 'playableDuration'],
  });

  const assets = result.edges.map(edge =>
    photoIdentifierToMediaAsset(edge, type),
  );

  return {
    assets,
    endCursor: result.page_info.end_cursor,
    hasMore: result.page_info.has_next_page,
  };
}

export async function fetchAllMediaAssets(
  type: MediaType,
): Promise<MediaAsset[]> {
  const allAssets: MediaAsset[] = [];
  let cursor: string | undefined;
  let hasMore = true;

  while (hasMore) {
    const result = await fetchMediaAssets(type, cursor);
    allAssets.push(...result.assets);
    cursor = result.endCursor;
    hasMore = result.hasMore;
  }

  return allAssets;
}

export async function deleteAsset(uri: string): Promise<boolean> {
  try {
    await CameraRoll.deletePhotos([uri]);
    return true;
  } catch {
    return false;
  }
}

