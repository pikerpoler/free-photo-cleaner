import React, {useCallback, useEffect, useState} from 'react';
import {
  ActivityIndicator,
  Alert,
  Modal,
  SafeAreaView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  useColorScheme,
} from 'react-native';
import {useQueueStore} from '../stores/queueStore';
import {useSettingsStore} from '../stores/settingsStore';
import {useAIStore} from '../stores/aiStore';
import {SwipeCard} from '../components/SwipeCard';
import {MediaCard} from '../components/MediaCard';
import {StorageBar} from '../components/StorageBar';
import {TrainAIScreen} from './TrainAIScreen';

const BATCH_WARNING_THRESHOLD = 1024;

export function PhotosScreen() {
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';
  const [showTrainAI, setShowTrainAI] = useState(false);

  const {
    photoQueue,
    photoIndex,
    isLoadingPhotos,
    isLoadingPhotosMore,
    loadQueue,
    applySort,
    swipeLeft,
    swipeRight,
    undo,
    flushDeletes,
    photoUndoStack,
    photoPendingCount,
  } = useQueueStore();

  const {photoSort, photoFilters, dateFilter} = useSettingsStore();
  const scoreById = useAIStore(s => s.scoreById);

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

    if (filtersChanged || dateChanged) {
      loadQueue('photo', photoSort, photoFilters, undefined, dateFilter);
    } else {
      applySort('photo', photoSort);
    }
  }, [photoSort, photoFilters, dateFilter, loadQueue, applySort]);

  const currentAsset = photoQueue[photoIndex];
  const nextAsset = photoQueue[photoIndex + 1] || null;
  const remaining = photoQueue.length - photoIndex;
  const canUndo = photoUndoStack.length > 0;
  const showScores = photoSort === 'ai';
  const currentScore = currentAsset
    ? scoreById[currentAsset.id]
    : undefined;
  const nextScore = nextAsset ? scoreById[nextAsset.id] : undefined;

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

  const trainButton = (
    <TouchableOpacity
      onPress={() => setShowTrainAI(true)}
      style={styles.trainButton}>
      <Text style={styles.trainText}>Train AI</Text>
    </TouchableOpacity>
  );

  const trainModal = (
    <Modal
      visible={showTrainAI}
      animationType="slide"
      presentationStyle="pageSheet"
      onRequestClose={() => setShowTrainAI(false)}>
      <SafeAreaView style={[styles.container, isDark && styles.bgDark]}>
        <TrainAIScreen onClose={() => setShowTrainAI(false)} />
      </SafeAreaView>
    </Modal>
  );

  if (isLoadingPhotos) {
    return (
      <View style={[styles.center, isDark && styles.bgDark]}>
        <ActivityIndicator size="large" color="#007AFF" />
        <Text style={[styles.loadingText, isDark && styles.textDark]}>
          Loading photos...
        </Text>
        {trainModal}
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
          {trainModal}
        </View>
      );
    }

    return (
      <View style={[styles.center, isDark && styles.bgDark]}>
        <Text style={[styles.emptyTitle, isDark && styles.textDark]}>
          All caught up!
        </Text>
        <Text style={[styles.emptySubtitle, isDark && styles.textSecondary]}>
          No more photos to review.
        </Text>
        {photoPendingCount > 0 && (
          <TouchableOpacity
            onPress={handleDelete}
            style={styles.deleteButtonLarge}>
            {photoPendingCount >= BATCH_WARNING_THRESHOLD && (
              <Text style={styles.warningIcon}>⚠️ </Text>
            )}
            <Text style={styles.deleteButtonLargeText}>
              Delete {photoPendingCount}
            </Text>
          </TouchableOpacity>
        )}
        {trainButton}
        <StorageBar />
        {trainModal}
      </View>
    );
  }

  return (
    <View style={[styles.container, isDark && styles.bgDark]}>
      <View style={styles.header}>
        <Text style={[styles.counter, isDark && styles.textDark]}>
          {remaining} remaining
        </Text>
        <View style={styles.headerButtons}>
          {trainButton}
          {canUndo && (
            <TouchableOpacity onPress={handleUndo} style={styles.undoButton}>
              <Text style={styles.undoText}>Undo</Text>
            </TouchableOpacity>
          )}
          {photoPendingCount > 0 && (
            <TouchableOpacity onPress={handleDelete} style={styles.deleteButton}>
              {photoPendingCount >= BATCH_WARNING_THRESHOLD && (
                <Text style={styles.warningIcon}>⚠️</Text>
              )}
              <Text style={styles.deleteText}>
                Delete {photoPendingCount}
              </Text>
            </TouchableOpacity>
          )}
        </View>
      </View>

      <SwipeCard
        onSwipeLeft={handleSwipeLeft}
        onSwipeRight={handleSwipeRight}
        cardKey={currentAsset.id}
        behindContent={
          nextAsset ? (
            <MediaCard
              asset={nextAsset}
              score={showScores ? nextScore : null}
            />
          ) : undefined
        }>
        <MediaCard
          asset={currentAsset}
          score={showScores ? currentScore : null}
        />
      </SwipeCard>

      <StorageBar />
      {trainModal}
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
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 8,
  },
  headerButtons: {
    flexDirection: 'row',
    gap: 8,
    alignItems: 'center',
  },
  counter: {
    fontSize: 14,
    color: '#666',
    fontWeight: '500',
  },
  textDark: {
    color: '#fff',
  },
  textSecondary: {
    color: '#8e8e93',
  },
  trainButton: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: '#5856D6',
    borderRadius: 14,
  },
  trainText: {
    color: '#fff',
    fontSize: 13,
    fontWeight: '600',
  },
  undoButton: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: '#007AFF',
    borderRadius: 14,
  },
  undoText: {
    color: '#fff',
    fontSize: 13,
    fontWeight: '600',
  },
  deleteButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: '#ff3b30',
    borderRadius: 14,
    gap: 4,
  },
  deleteText: {
    color: '#fff',
    fontSize: 13,
    fontWeight: '600',
  },
  deleteButtonLarge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 12,
    backgroundColor: '#ff3b30',
    borderRadius: 10,
    marginTop: 20,
    gap: 4,
  },
  deleteButtonLargeText: {
    color: '#fff',
    fontSize: 17,
    fontWeight: '700',
  },
  warningIcon: {
    fontSize: 14,
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
