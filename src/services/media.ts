import {Platform} from 'react-native';
import {
  CameraRoll,
  PhotoIdentifier,
} from '@react-native-camera-roll/camera-roll';
import {DateFilter, MediaAsset, MediaType} from '../types/media';

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

export function dateFilterToCameraRollTimes(dateFilter?: DateFilter): {
  fromTime?: number;
  toTime?: number;
} {
  if (!dateFilter?.enabled) return {};
  const startOfFrom = new Date(dateFilter.from);
  startOfFrom.setHours(0, 0, 0, 0);
  const endOfTo = new Date(dateFilter.to);
  endOfTo.setHours(23, 59, 59, 999);
  // CameraRoll fromTime is exclusive — subtract 1ms for inclusive start-of-day
  return {
    fromTime: startOfFrom.getTime() - 1,
    toTime: endOfTo.getTime(),
  };
}

export async function fetchMediaAssets(
  type: MediaType,
  cursor?: string,
  dateFilter?: DateFilter,
): Promise<{assets: MediaAsset[]; endCursor: string | undefined; hasMore: boolean}> {
  const assetType = type === 'photo' ? 'Photos' : 'Videos';
  const {fromTime, toTime} = dateFilterToCameraRollTimes(dateFilter);

  const result = await CameraRoll.getPhotos({
    first: BATCH_SIZE,
    after: cursor,
    assetType,
    include: ['filename', 'fileSize', 'imageSize', 'playableDuration'],
    ...(fromTime != null ? {fromTime} : {}),
    ...(toTime != null ? {toTime} : {}),
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

export interface ProgressiveLoadController {
  cancel: () => void;
}

export type OnBatchLoaded = (
  assets: MediaAsset[],
  done: boolean,
) => void;

/**
 * Loads media assets progressively, calling onBatch after each page.
 * Returns a controller with a cancel() method to abort mid-load.
 */
export function loadMediaProgressively(
  type: MediaType,
  onBatch: OnBatchLoaded,
  dateFilter?: DateFilter,
): ProgressiveLoadController {
  let cancelled = false;

  const controller: ProgressiveLoadController = {
    cancel: () => {
      cancelled = true;
    },
  };

  (async () => {
    let cursor: string | undefined;
    let hasMore = true;

    while (hasMore && !cancelled) {
      const result = await fetchMediaAssets(type, cursor, dateFilter);
      if (cancelled) return;
      onBatch(result.assets, !result.hasMore);
      cursor = result.endCursor;
      hasMore = result.hasMore;
    }
  })();

  return controller;
}

export async function deleteAsset(uri: string): Promise<boolean> {
  try {
    await CameraRoll.deletePhotos([uri]);
    return true;
  } catch {
    return false;
  }
}
