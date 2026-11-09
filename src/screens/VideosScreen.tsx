import React, {useCallback, useEffect, useState} from 'react';
import {
  ActivityIndicator,
  Alert,
  StyleSheet,
  Text,
  View,
  useColorScheme,
} from 'react-native';
import {useShallow} from 'zustand/react/shallow';
import {useQueueStore} from '../stores/queueStore';
import {useSettingsStore} from '../stores/settingsStore';
import {SwipeCard} from '../components/SwipeCard';
import {VideoCard} from '../components/VideoCard';
import {MediaCard} from '../components/MediaCard';
import {StorageBar} from '../components/StorageBar';
import {ReviewHeader} from '../components/ReviewHeader';

const BATCH_WARNING_THRESHOLD = 1024;

export function VideosScreen() {
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';

  const {
    videoQueue,
    videoIndex,
    isLoadingVideos,
    isLoadingVideosMore,
    loadQueue,
    applySort,
    applyFilters,
    swipeLeft,
    swipeRight,
    undo,
    flushDeletes,
    videoUndoStack,
    videoPendingCount,
    videoReviewStats,
  } = useQueueStore(
    useShallow(s => ({
      videoQueue: s.videoQueue,
      videoIndex: s.videoIndex,
      isLoadingVideos: s.isLoadingVideos,
      isLoadingVideosMore: s.isLoadingVideosMore,
      loadQueue: s.loadQueue,
      applySort: s.applySort,
      applyFilters: s.applyFilters,
      swipeLeft: s.swipeLeft,
      swipeRight: s.swipeRight,
      undo: s.undo,
      flushDeletes: s.flushDeletes,
      videoUndoStack: s.videoUndoStack,
      videoPendingCount: s.videoPendingCount,
      videoReviewStats: s.videoReviewStats,
    })),
  );

  const {videoSort, videoFilters, dateFilter} = useSettingsStore(
    useShallow(s => ({
      videoSort: s.videoSort,
      videoFilters: s.videoFilters,
      dateFilter: s.dateFilter,
    })),
  );
  const [videosMuted, setVideosMuted] = useState(true);
  const toggleVideoMute = useCallback(() => {
    setVideosMuted(prev => !prev);
  }, []);

  const videoFiltersRef = React.useRef(videoFilters);
  const dateFilterRef = React.useRef(dateFilter);
  const isFirstLoad = React.useRef(true);

  useEffect(() => {
    if (isFirstLoad.current) {
      isFirstLoad.current = false;
      loadQueue('video', videoSort, undefined, videoFilters, dateFilter);
      return;
    }

    const filtersChanged =
      JSON.stringify(videoFilters) !== JSON.stringify(videoFiltersRef.current);
    const dateChanged =
      JSON.stringify(dateFilter) !== JSON.stringify(dateFilterRef.current);
    videoFiltersRef.current = videoFilters;
    dateFilterRef.current = dateFilter;

    if (dateChanged) {
      loadQueue('video', videoSort, undefined, videoFilters, dateFilter);
    } else if (filtersChanged) {
      applyFilters('video', videoSort, undefined, videoFilters);
    } else {
      applySort('video', videoSort);
    }
  }, [videoSort, videoFilters, dateFilter, loadQueue, applySort, applyFilters]);

  const currentAsset = videoQueue[videoIndex];
  const nextAsset = videoQueue[videoIndex + 1] || null;
  const canUndo = videoUndoStack.length > 0;

  const handleSwipeLeft = useCallback(() => {
    swipeLeft('video');
  }, [swipeLeft]);

  const handleSwipeRight = useCallback(() => {
    swipeRight('video');
  }, [swipeRight]);

  const handleUndo = useCallback(() => {
    undo('video');
  }, [undo]);

  const handleDelete = useCallback(() => {
    if (videoPendingCount >= BATCH_WARNING_THRESHOLD) {
      Alert.alert(
        'Large batch',
        'You may want to delete your videos in smaller batches to avoid system timeouts.',
        [
          {text: 'Cancel', style: 'cancel'},
          {
            text: 'Delete anyway',
            style: 'destructive',
            onPress: () => flushDeletes('video'),
          },
        ],
      );
    } else {
      flushDeletes('video');
    }
  }, [videoPendingCount, flushDeletes]);

  if (isLoadingVideos) {
    return (
      <View style={[styles.center, isDark && styles.bgDark]}>
        <ActivityIndicator size="large" color="#007AFF" />
        <Text style={[styles.loadingText, isDark && styles.textDark]}>
          Loading videos...
        </Text>
      </View>
    );
  }

  if (!currentAsset) {
    if (isLoadingVideosMore) {
      return (
        <View style={[styles.center, isDark && styles.bgDark]}>
          <ActivityIndicator size="large" color="#007AFF" />
          <Text style={[styles.loadingText, isDark && styles.textDark]}>
            Loading more videos...
          </Text>
        </View>
      );
    }

    return (
      <View style={[styles.container, isDark && styles.bgDark]}>
        <ReviewHeader
          stats={videoReviewStats}
          canUndo={canUndo}
          pendingCount={videoPendingCount}
          batchWarningThreshold={BATCH_WARNING_THRESHOLD}
          onUndo={handleUndo}
          onDelete={handleDelete}
        />
        <View style={styles.centerFill}>
          <Text style={[styles.emptyTitle, isDark && styles.textDark]}>
            All caught up!
          </Text>
          <Text style={[styles.emptySubtitle, isDark && styles.textSecondary]}>
            No more videos to review.
          </Text>
        </View>
        <StorageBar />
      </View>
    );
  }

  return (
    <View style={[styles.container, isDark && styles.bgDark]}>
      <ReviewHeader
        stats={videoReviewStats}
        canUndo={canUndo}
        pendingCount={videoPendingCount}
        batchWarningThreshold={BATCH_WARNING_THRESHOLD}
        onUndo={handleUndo}
        onDelete={handleDelete}
      />

      <SwipeCard
        onSwipeLeft={handleSwipeLeft}
        onSwipeRight={handleSwipeRight}
        cardKey={currentAsset.id}
        behindContent={nextAsset ? <MediaCard asset={nextAsset} /> : undefined}>
        <VideoCard
          asset={currentAsset}
          isActive
          muted={videosMuted}
          onToggleMute={toggleVideoMute}
        />
      </SwipeCard>

      <StorageBar />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  bgDark: {
    backgroundColor: '#000',
  },
  center: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#fff',
    padding: 20,
  },
  centerFill: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  textDark: {
    color: '#fff',
  },
  textSecondary: {
    color: '#8e8e93',
  },
  loadingText: {
    marginTop: 12,
    fontSize: 16,
    color: '#666',
  },
  emptyTitle: {
    fontSize: 22,
    fontWeight: '700',
    color: '#333',
  },
  emptySubtitle: {
    fontSize: 15,
    color: '#666',
    marginTop: 6,
  },
});
