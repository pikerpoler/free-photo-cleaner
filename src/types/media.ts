export type MediaType = 'photo' | 'video';

export type SortMode =
  | 'random'
  | 'largest_first'
  | 'smallest_first'
  | 'oldest_first'
  | 'newest_first';

export interface MediaAsset {
  id: string;
  uri: string;
  type: MediaType;
  filename: string;
  width: number;
  height: number;
  fileSize: number;
  creationDate: number;
  duration?: number;
  // Metadata flags
  hasExif: boolean;
  isScreenshot: boolean;
  isWhatsApp: boolean;
}

export interface PhotoFilters {
  mode: 'all' | 'categories';
  screenshots: boolean;
  whatsapp: boolean;
  noMetadata: boolean;
}

export interface VideoFilters {
  minDuration: number; // seconds
  maxDuration: number; // seconds, Infinity for no limit
  minSize: number; // bytes
  maxSize: number; // bytes, Infinity for no limit
}

export interface DateFilter {
  enabled: boolean;
  from: number; // Unix ms timestamp
  to: number; // Unix ms timestamp
}

export interface StorageInfo {
  totalSpace: number;
  freeSpace: number;
  photosSize: number;
  videosSize: number;
  isFallback?: boolean;
}
