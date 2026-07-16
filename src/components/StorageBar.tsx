import React, {useEffect, useState} from 'react';
import {StyleSheet, Text, View, useColorScheme} from 'react-native';
import {getNativeStorageInfo, getCachedLibraryStats} from '../services/nativeStorage';
import {StorageInfo} from '../types/media';
import {formatFileSize} from '../utils/format';

export function StorageBar() {
  const [info, setInfo] = useState<StorageInfo | null>(
    () => getCachedLibraryStats(),
  );
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';

  useEffect(() => {
    getNativeStorageInfo().then(setInfo);
  }, []);

  if (!info) return null;

  if (info.isFallback) {
    return (
      <View style={[styles.container, isDark && styles.containerDark]}>
        <View style={styles.bar}>
          <View style={[styles.segment, styles.other, {flex: 1}]} />
        </View>
        <View style={styles.legend}>
          <Text style={[styles.legendText, isDark && styles.textDark, styles.fallbackText]}>
            Storage info unavailable
          </Text>
          <Text style={[styles.available, isDark && styles.textDark]}>
            Available: UNKNOWN
          </Text>
        </View>
      </View>
    );
  }

  const total = info.totalSpace;
  const photoPct = (info.photosSize / total) * 100;
  const videoPct = (info.videosSize / total) * 100;
  const freePct = (info.freeSpace / total) * 100;
  const otherPct = Math.max(0, 100 - photoPct - videoPct - freePct);

  return (
    <View style={[styles.container, isDark && styles.containerDark]}>
      <View style={styles.bar}>
        {photoPct > 0 && (
          <View style={[styles.segment, styles.photos, {flex: photoPct}]} />
        )}
        {videoPct > 0 && (
          <View style={[styles.segment, styles.videos, {flex: videoPct}]} />
        )}
        {otherPct > 0 && (
          <View style={[styles.segment, styles.other, {flex: otherPct}]} />
        )}
        {freePct > 0 && (
          <View style={[styles.segment, styles.free, {flex: freePct}]} />
        )}
      </View>
      <View style={styles.legend}>
        <View style={styles.legendItem}>
          <View style={[styles.dot, styles.photos]} />
          <Text style={[styles.legendText, isDark && styles.textDark]}>
            Photos
          </Text>
        </View>
        <View style={styles.legendItem}>
          <View style={[styles.dot, styles.videos]} />
          <Text style={[styles.legendText, isDark && styles.textDark]}>
            Videos
          </Text>
        </View>
        <View style={styles.legendItem}>
          <View style={[styles.dot, styles.other]} />
          <Text style={[styles.legendText, isDark && styles.textDark]}>
            Other
          </Text>
        </View>
        <Text style={[styles.available, isDark && styles.textDark]}>
          Available: {formatFileSize(info.freeSpace)}
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    paddingHorizontal: 16,
    paddingVertical: 10,
    backgroundColor: '#f8f8f8',
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: '#e0e0e0',
  },
  containerDark: {
    backgroundColor: '#1c1c1e',
    borderTopColor: '#3a3a3c',
  },
  bar: {
    flexDirection: 'row',
    height: 6,
    borderRadius: 3,
    overflow: 'hidden',
    backgroundColor: '#e0e0e0',
  },
  segment: {
    height: '100%',
  },
  photos: {
    backgroundColor: '#007AFF',
  },
  videos: {
    backgroundColor: '#FF9500',
  },
  other: {
    backgroundColor: '#8E8E93',
  },
  free: {
    backgroundColor: '#e0e0e0',
  },
  legend: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 6,
    gap: 12,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  legendText: {
    fontSize: 11,
    color: '#666',
  },
  textDark: {
    color: '#aaa',
  },
  available: {
    fontSize: 11,
    color: '#666',
    marginLeft: 'auto',
    fontWeight: '600',
  },
  fallbackText: {
    fontStyle: 'italic',
  },
});
