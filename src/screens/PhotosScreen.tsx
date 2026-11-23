import React, {useCallback, useEffect} from 'react';
import {useFocusEffect} from '@react-navigation/native';
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
import {useAIStore} from '../stores/aiStore';
import {SwipeCard} from '../components/SwipeCard';
import {MediaCard} from '../components/MediaCard';
import {StorageBar} from '../components/StorageBar';
import {ReviewHeader} from '../components/ReviewHeader';

const BATCH_WARNING_THRESHOLD = 1024;

export function PhotosScreen() {
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';

  const {
    photoQueue,
    photoIndex,
    isLoadingPhotos,
    isScoringPhotos,
    isLoadingPhotosMore,
    loadQueue,
    applySort,
    applyFilters,
    swipeLeft,
    swipeRight,
    undo,
    flushDeletes,
    photoUndoStack,
    photoPendingCount,
    photoReviewStats,
    photoICloudRemainder,
    reconcileAIScoringOnFocus,
  } = useQueueStore(
    useShallow(s => ({
      photoQueue: s.photoQueue,
      photoIndex: s.photoIndex,
      isLoadingPhotos: s.isLoadingPhotos,
      isScoringPhotos: s.isScoringPhotos,
      isLoadingPhotosMore: s.isLoadingPhotosMore,
      loadQueue: s.loadQueue,
      applySort: s.applySort,
      applyFilters: s.applyFilters,
      swipeLeft: s.swipeLeft,
      swipeRight: s.swipeRight,
      undo: s.undo,
      flushDeletes: s.flushDeletes,
      photoUndoStack: s.photoUndoStack,
      photoPendingCount: s.photoPendingCount,
      photoReviewStats: s.photoReviewStats,
      photoICloudRemainder: s.photoICloudRemainder,
      reconcileAIScoringOnFocus: s.reconcileAIScoringOnFocus,
    })),
  );

  const {photoSort, photoFilters, dateFilter} = useSettingsStore(
    useShallow(s => ({
      photoSort: s.photoSort,
      photoFilters: s.photoFilters,
      dateFilter: s.dateFilter,
    })),
  );

  const photoFiltersRef = React.useRef(photoFilters);
  const dateFilterRef = React.useRef(dateFilter);
  const isFirstLoad = React.useRef(true);

  useEffect(() => {
    if (isFirstLoad.current) {
      isFirstLoad.current = false;
      loadQueue('photo', photoSort, photoFilters, undefined, dateFilter);
      return;
    }

    const filtersChanged =
      JSON.stringify(photoFilters) !== JSON.stringify(photoFiltersRef.current);
    const dateChanged =
      JSON.stringify(dateFilter) !== JSON.stringify(dateFilterRef.current);
    photoFiltersRef.current = photoFilters;
    dateFilterRef.current = dateFilter;

    if (dateChanged) {
      loadQueue('photo', photoSort, photoFilters, undefined, dateFilter);
    } else if (filtersChanged) {
      applyFilters('photo', photoSort, photoFilters, undefined);
    } else {
      applySort('photo', photoSort);
    }
  }, [photoSort, photoFilters, dateFilter, loadQueue, applySort, applyFilters]);

  useFocusEffect(
    useCallback(() => {
      if (photoSort !== 'ai') return;
      void reconcileAIScoringOnFocus(); // eslint-disable-line no-void
    }, [photoSort, reconcileAIScoringOnFocus]),
  );

  const currentAsset = photoQueue[photoIndex];
  const nextAsset = photoQueue[photoIndex + 1] || null;
  const canUndo = photoUndoStack.length > 0;
  const showScores = photoSort === 'ai';
  const currentId = currentAsset?.id;
  const nextId = nextAsset?.id;
  const currentScore = useAIStore(s =>
    currentId ? s.scoreById[currentId] : undefined,
  );
  const nextScore = useAIStore(s =>
    nextId ? s.scoreById[nextId] : undefined,
  );

  const handleSwipeLeft = useCallback(() => {
    swipeLeft('photo');
  }, [swipeLeft]);

  const handleSwipeRight = useCallback(() => {
    swipeRight('photo');
  }, [swipeRight]);

  const handleUndo = useCallback(() => {
    undo('photo');
  }, [undo]);

  const handleDelete = useCallback(() => {
    if (photoPendingCount >= BATCH_WARNING_THRESHOLD) {
      Alert.alert(
        'Large batch',
        'You may want to delete your photos in smaller batches to avoid system timeouts.',
        [
          {text: 'Cancel', style: 'cancel'},
          {
            text: 'Delete anyway',
            style: 'destructive',
            onPress: () => flushDeletes('photo'),
          },
        ],
      );
    } else {
      flushDeletes('photo');
    }
  }, [photoPendingCount, flushDeletes]);

  if (isLoadingPhotos) {
    return (
      <View style={[styles.center, isDark && styles.bgDark]}>
        <ActivityIndicator size="large" color="#007AFF" />
        <Text style={[styles.loadingText, isDark && styles.textDark]}>
          {isScoringPhotos ? 'Scoring photos...' : 'Loading photos...'}
        </Text>
      </View>
    );
  }

  if (!currentAsset) {
    if (isLoadingPhotosMore) {
      return (
        <View style={[styles.center, isDark && styles.bgDark]}>
          <ActivityIndicator size="large" color="#007AFF" />
          <Text style={[styles.loadingText, isDark && styles.textDark]}>
            Loading more photos...
          </Text>
        </View>
      );
    }

    return (
      <View style={[styles.container, isDark && styles.bgDark]}>
        <ReviewHeader
          stats={photoReviewStats}
          canUndo={canUndo}
          pendingCount={photoPendingCount}
          batchWarningThreshold={BATCH_WARNING_THRESHOLD}
          onUndo={handleUndo}
          onDelete={handleDelete}
        />
        <View style={styles.centerFill}>
          <Text style={[styles.emptyTitle, isDark && styles.textDark]}>
            All caught up!
          </Text>
          {photoICloudRemainder ? (
            <>
              <Text
                style={[styles.emptySubtitle, isDark && styles.textSecondary]}>
                No more photos to review locally.
              </Text>
              <Text
                style={[styles.emptySubtitle, isDark && styles.textSecondary]}>
                You appear to have some photos backed on iCloud but we can't
                review those right now.
              </Text>
            </>
          ) : (
            <Text style={[styles.emptySubtitle, isDark && styles.textSecondary]}>
              No more photos to review.
            </Text>
          )}
        </View>
        <StorageBar />
      </View>
    );
  }

  return (
    <View style={[styles.container, isDark && styles.bgDark]}>
      <ReviewHeader
        stats={photoReviewStats}
        canUndo={canUndo}
        pendingCount={photoPendingCount}
        batchWarningThreshold={BATCH_WARNING_THRESHOLD}
        onUndo={handleUndo}
        onDelete={handleDelete}
      />

      <SwipeCard
        onSwipeLeft={handleSwipeLeft}
        onSwipeRight={handleSwipeRight}
        cardKey={currentAsset.id}
        behindContent={
          nextAsset ? (
            <MediaCard
              asset={nextAsset}
              showScore={showScores}
              score={nextScore ?? null}
            />
          ) : undefined
        }>
        <MediaCard
          asset={currentAsset}
          showScore={showScores}
          score={currentScore ?? null}
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
    textAlign: 'center',
    paddingHorizontal: 24,
  },
});
