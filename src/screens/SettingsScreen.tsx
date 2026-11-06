import React, {useCallback, useEffect, useState} from 'react';
import {
  Alert,
  Linking,
  Modal,
  Platform,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TouchableOpacity,
  View,
  useColorScheme,
  SafeAreaView,
} from 'react-native';
import Slider from '@react-native-community/slider';
import DateTimePicker, {
  DateTimePickerEvent,
} from '@react-native-community/datetimepicker';
import {useSettingsStore} from '../stores/settingsStore';
import {useAIStore} from '../stores/aiStore';
import {useQueueStore} from '../stores/queueStore';
import {resetKeepHistory} from '../services/database';
import {SortMode} from '../types/media';
import {TrainableModelId, TRAINABLE_MODELS} from '../ai/cnnTypes';
import {formatFileSize, formatDuration} from '../utils/format';

const SORT_OPTIONS: {label: string; value: SortMode}[] = [
  {label: 'Random', value: 'random'},
  {label: 'Largest first', value: 'largest_first'},
  {label: 'Smallest first', value: 'smallest_first'},
  {label: 'Oldest first', value: 'oldest_first'},
  {label: 'Newest first', value: 'newest_first'},
];

const PHOTO_SORT_OPTIONS: {label: string; value: SortMode}[] = [
  ...SORT_OPTIONS,
  {label: 'AI', value: 'ai'},
];

const BUG_MAILTO =
  'mailto:pikerpoler@gmail.com?subject=' +
  encodeURIComponent('free-photo-cleaner [BUG]') +
  '&body=' +
  encodeURIComponent(
    'Hey Nadav,\nI found a bug in your app.\n<bug description>',
  );

const DONATE_URL = 'https://buymeacoffee.com/pikerpoler';

function SortPicker({
  value,
  onChange,
  isDark,
  options,
}: {
  value: SortMode;
  onChange: (v: SortMode) => void;
  isDark: boolean;
  options?: {label: string; value: SortMode}[];
}) {
  const items = options ?? SORT_OPTIONS;
  return (
    <View style={styles.sortContainer}>
      {items.map(opt => (
        <TouchableOpacity
          key={opt.value}
          style={[
            styles.sortOption,
            isDark && styles.sortOptionDark,
            value === opt.value && styles.sortOptionActive,
          ]}
          onPress={() => onChange(opt.value)}>
          <Text
            style={[
              styles.sortText,
              isDark && styles.textDark,
              value === opt.value && styles.sortTextActive,
            ]}>
            {opt.label}
          </Text>
        </TouchableOpacity>
      ))}
    </View>
  );
}

export function SettingsScreen() {
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';

  const {
    photoSort,
    videoSort,
    photoFilters,
    videoFilters,
    dateFilter,
    setPhotoSort,
    setVideoSort,
    setPhotoFilters,
    setVideoFilters,
    setDateFilter,
    setAIModel,
  } = useSettingsStore();

  const {
    hasModel,
    activeModelInfo,
    refreshModelStatus,
    refreshCheckpoints,
    checkpoints,
    selectSortModel,
    deleteCheckpoint,
  } = useAIStore();
  const resetAIQueue = useQueueStore(s => s.resetAIQueue);
  const onKeepHistoryReset = useQueueStore(s => s.onKeepHistoryReset);
  const setAISortModel = useSettingsStore(s => s.setAISortModel);

  const [showFromPicker, setShowFromPicker] = useState(false);
  const [showToPicker, setShowToPicker] = useState(false);
  const [showAbout, setShowAbout] = useState(false);

  useEffect(() => {
    refreshModelStatus();
    refreshCheckpoints();
  }, [refreshModelStatus, refreshCheckpoints]);

  const handleResetKeepHistory = useCallback(() => {
    Alert.alert(
      'Reset Keep History',
      'Previously kept media will appear again in the queue. Deleted media will NOT be restored.',
      [
        {text: 'Cancel', style: 'cancel'},
        {
          text: 'Reset',
          style: 'destructive',
          onPress: () => {
            resetKeepHistory();
            onKeepHistoryReset();
          },
        },
      ],
    );
  }, [onKeepHistoryReset]);

  const handleResetAIQueue = useCallback(() => {
    Alert.alert(
      'Reset AI Queue',
      'Clears AI scores and rebuilds the photo swipe queue. Kept / marked-for-deletion labels are kept.',
      [
        {text: 'Cancel', style: 'cancel'},
        {
          text: 'Reset',
          style: 'destructive',
          onPress: () => resetAIQueue(),
        },
      ],
    );
  }, [resetAIQueue]);

  const handleSelectSortModel = useCallback(
    async (modelId: TrainableModelId) => {
      try {
        await selectSortModel(modelId);
        setAISortModel(modelId);
        setAIModel(modelId);
        await resetAIQueue();
      } catch (e: unknown) {
        Alert.alert(
          'Load failed',
          e instanceof Error ? e.message : String(e),
        );
      }
    },
    [selectSortModel, setAISortModel, setAIModel, resetAIQueue],
  );

  const handleDeleteCheckpoint = useCallback(
    (modelId: string) => {
      const label =
        TRAINABLE_MODELS.find(m => m.id === modelId)?.label ?? modelId;
      Alert.alert(
        `Delete ${label}?`,
        `This will delete ${label}'s weights. This action is irreversible. Are you sure?`,
        [
          {text: 'Cancel', style: 'cancel'},
          {
            text: 'Delete',
            style: 'destructive',
            onPress: async () => {
              const prevSort = useSettingsStore.getState().aiSortModel;
              await deleteCheckpoint(modelId);
              if (prevSort === modelId) {
                setAISortModel(null);
              }
              await resetAIQueue();
              await refreshCheckpoints();
              await refreshModelStatus();
            },
          },
        ],
      );
    },
    [
      deleteCheckpoint,
      resetAIQueue,
      refreshCheckpoints,
      refreshModelStatus,
      setAISortModel,
    ],
  );

  const handleFromChange = useCallback(
    (event: DateTimePickerEvent, selectedDate?: Date) => {
      if (Platform.OS === 'android') {
        setShowFromPicker(false);
      }
      if (selectedDate) {
        setDateFilter({from: selectedDate.getTime()});
      }
    },
    [setDateFilter],
  );

  const handleToChange = useCallback(
    (event: DateTimePickerEvent, selectedDate?: Date) => {
      if (Platform.OS === 'android') {
        setShowToPicker(false);
      }
      if (selectedDate) {
        setDateFilter({to: selectedDate.getTime()});
      }
    },
    [setDateFilter],
  );

  const formatShortDate = (timestamp: number) => {
    return new Date(timestamp).toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  };

  const maxDurationDisplay =
    videoFilters.maxDuration === Infinity
      ? '∞'
      : formatDuration(videoFilters.maxDuration);

  const maxSizeDisplay =
    videoFilters.maxSize === Infinity
      ? '∞'
      : formatFileSize(videoFilters.maxSize);

  const openMailBug = useCallback(async () => {
    try {
      const supported = await Linking.canOpenURL(BUG_MAILTO);
      if (supported) {
        await Linking.openURL(BUG_MAILTO);
      } else {
        Alert.alert(
          'Mail unavailable',
          'Could not open the Mail app. Email pikerpoler@gmail.com manually.',
        );
      }
    } catch {
      Alert.alert('Error', 'Could not open the Mail app.');
    }
  }, []);

  const openDonate = useCallback(async () => {
    try {
      await Linking.openURL(DONATE_URL);
    } catch {
      Alert.alert('Error', 'Could not open the link.');
    }
  }, []);

  return (
    <>
      <ScrollView
        style={[styles.container, isDark && styles.bgDark]}
        contentContainerStyle={styles.content}>
        <View style={[styles.row, isDark && styles.rowDark]}>
          <Text style={[styles.label, isDark && styles.textDark]}>
            Filter by Date
          </Text>
          <Switch
            value={dateFilter.enabled}
            onValueChange={v => setDateFilter({enabled: v})}
            trackColor={{false: isDark ? '#39393D' : '#e9e9ea', true: '#34C759'}}
          />
        </View>
        {dateFilter.enabled && (
          <>
            <View style={[styles.row, isDark && styles.rowDark]}>
              <Text style={[styles.label, isDark && styles.textDark]}>From</Text>
              <TouchableOpacity
                style={[styles.dateButton, isDark && styles.dateButtonDark]}
                onPress={() => {
                  setShowFromPicker(v => !v);
                  setShowToPicker(false);
                }}>
                <Text style={[styles.dateButtonText, isDark && styles.textDark]}>
                  {formatShortDate(dateFilter.from)}
                </Text>
              </TouchableOpacity>
            </View>
            {showFromPicker && (
              <DateTimePicker
                value={new Date(dateFilter.from)}
                mode="date"
                display={Platform.OS === 'ios' ? 'inline' : 'default'}
                maximumDate={new Date(dateFilter.to)}
                onChange={handleFromChange}
              />
            )}
            <View style={[styles.row, isDark && styles.rowDark]}>
              <Text style={[styles.label, isDark && styles.textDark]}>To</Text>
              <TouchableOpacity
                style={[styles.dateButton, isDark && styles.dateButtonDark]}
                onPress={() => {
                  setShowToPicker(v => !v);
                  setShowFromPicker(false);
                }}>
                <Text style={[styles.dateButtonText, isDark && styles.textDark]}>
                  {formatShortDate(dateFilter.to)}
                </Text>
              </TouchableOpacity>
            </View>
            {showToPicker && (
              <DateTimePicker
                value={new Date(dateFilter.to)}
                mode="date"
                display={Platform.OS === 'ios' ? 'inline' : 'default'}
                minimumDate={new Date(dateFilter.from)}
                maximumDate={new Date()}
                onChange={handleToChange}
              />
            )}
          </>
        )}

        <Text style={[styles.sectionTitle, isDark && styles.textDark]}>
          Photos Sort
        </Text>
        <SortPicker
          value={photoSort}
          onChange={setPhotoSort}
          isDark={isDark}
          options={PHOTO_SORT_OPTIONS}
        />

        {photoSort === 'ai' && (
          <>
            <Text style={[styles.filterLabel, isDark && styles.textSecondary]}>
              {hasModel && activeModelInfo
                ? `Active: ${activeModelInfo.label ?? activeModelInfo.modelId} · ${(activeModelInfo.testSize ?? 0) > 0 ? 'test accuracy' : 'accuracy'} ${((activeModelInfo.accuracy ?? activeModelInfo.bestTestAcc ?? 0) * 100).toFixed(0)}% · F1 ${(activeModelInfo.f1 ?? 0).toFixed(2)}`
                : 'No trained model yet'}
            </Text>
            {checkpoints.length === 0 ? (
              <Text style={[styles.filterLabel, isDark && styles.textSecondary]}>
                No saved checkpoints. Train a model from the Train AI tab.
              </Text>
            ) : (
              checkpoints.map(cp => {
                const label =
                  TRAINABLE_MODELS.find(m => m.id === cp.modelId)?.label ??
                  cp.modelId;
                const isActive = activeModelInfo?.modelId === cp.modelId;
                const date = new Date(cp.trainedAt || cp.savedAt || 0);
                return (
                  <View
                    key={cp.modelId}
                    style={[styles.checkpointRow, isDark && styles.rowDark]}>
                    <TouchableOpacity
                      style={styles.checkpointInfo}
                      onPress={() =>
                        handleSelectSortModel(cp.modelId as TrainableModelId)
                      }>
                      <Text
                        style={[
                          styles.checkpointTitle,
                          isDark && styles.textDark,
                          isActive && styles.checkpointActive,
                        ]}>
                        {label}
                        {isActive ? ' (active)' : ''}
                      </Text>
                      <Text
                        style={[
                          styles.checkpointMeta,
                          isDark && styles.textSecondary,
                        ]}>
                        {(cp.testSize ?? 0) > 0 ? 'test accuracy' : 'accuracy'}{' '}
                        {((cp.accuracy ?? cp.bestTestAcc ?? 0) * 100).toFixed(0)}% · F1{' '}
                        {(cp.f1 ?? 0).toFixed(2)} ·{' '}
                        {(cp.testSize ?? 0) > 0
                          ? `trained ${cp.trainSize ?? 0} · tested ${cp.testSize} · `
                          : `${cp.trainSize ?? 0} photos in last update · `}
                        {isNaN(date.getTime()) || date.getTime() === 0
                          ? '—'
                          : date.toLocaleDateString()}
                      </Text>
                    </TouchableOpacity>
                    <TouchableOpacity
                      style={styles.checkpointDelete}
                      onPress={() => handleDeleteCheckpoint(cp.modelId)}>
                      <Text style={styles.checkpointDeleteText}>Delete</Text>
                    </TouchableOpacity>
                  </View>
                );
              })
            )}
          </>
        )}

        <View style={[styles.row, isDark && styles.rowDark, styles.additionalFiltersRow]}>
          <Text style={[styles.label, isDark && styles.textDark]}>
            Additional Filters
          </Text>
          <Switch
            value={photoFilters.mode === 'categories'}
            onValueChange={v =>
              setPhotoFilters({mode: v ? 'categories' : 'all'})
            }
            trackColor={{false: isDark ? '#39393D' : '#e9e9ea', true: '#34C759'}}
          />
        </View>
        {photoFilters.mode === 'categories' && (
          <>
            <View style={[styles.row, isDark && styles.rowDark]}>
              <Text style={[styles.label, isDark && styles.textDark]}>
                Screenshots
              </Text>
              <Switch
                value={photoFilters.screenshots}
                onValueChange={v => setPhotoFilters({screenshots: v})}
                trackColor={{
                  false: isDark ? '#39393D' : '#e9e9ea',
                  true: '#34C759',
                }}
              />
            </View>
            <View style={[styles.row, isDark && styles.rowDark]}>
              <Text style={[styles.label, isDark && styles.textDark]}>
                WhatsApp Images
              </Text>
              <Switch
                value={photoFilters.whatsapp}
                onValueChange={v => setPhotoFilters({whatsapp: v})}
                trackColor={{
                  false: isDark ? '#39393D' : '#e9e9ea',
                  true: '#34C759',
                }}
              />
            </View>
            <View style={[styles.row, isDark && styles.rowDark]}>
              <Text style={[styles.label, isDark && styles.textDark]}>
                No Metadata
              </Text>
              <Switch
                value={photoFilters.noMetadata}
                onValueChange={v => setPhotoFilters({noMetadata: v})}
                trackColor={{
                  false: isDark ? '#39393D' : '#e9e9ea',
                  true: '#34C759',
                }}
              />
            </View>
          </>
        )}

        <Text style={[styles.sectionTitle, isDark && styles.textDark]}>
          Videos Sort
        </Text>
        <SortPicker value={videoSort} onChange={setVideoSort} isDark={isDark} />

        <Text style={[styles.sectionTitle, isDark && styles.textDark]}>
          Videos Filters
        </Text>

        <Text style={[styles.filterLabel, isDark && styles.textDark]}>
          Duration: {formatDuration(videoFilters.minDuration)} –{' '}
          {maxDurationDisplay}
        </Text>
        <View style={styles.sliderRow}>
          <Text style={[styles.sliderLabel, isDark && styles.textSecondary]}>
            Min
          </Text>
          <Slider
            style={styles.slider}
            minimumValue={0}
            maximumValue={600}
            step={5}
            value={videoFilters.minDuration}
            onValueChange={v => setVideoFilters({minDuration: v})}
            minimumTrackTintColor="#007AFF"
          />
        </View>
        <View style={styles.sliderRow}>
          <Text style={[styles.sliderLabel, isDark && styles.textSecondary]}>
            Max
          </Text>
          <Slider
            style={styles.slider}
            minimumValue={0}
            maximumValue={601}
            step={5}
            value={
              videoFilters.maxDuration === Infinity
                ? 601
                : videoFilters.maxDuration
            }
            onValueChange={v =>
              setVideoFilters({maxDuration: v >= 601 ? Infinity : v})
            }
            minimumTrackTintColor="#007AFF"
          />
        </View>

        <Text style={[styles.filterLabel, isDark && styles.textDark]}>
          Size: {formatFileSize(videoFilters.minSize)} – {maxSizeDisplay}
        </Text>
        <View style={styles.sliderRow}>
          <Text style={[styles.sliderLabel, isDark && styles.textSecondary]}>
            Min
          </Text>
          <Slider
            style={styles.slider}
            minimumValue={0}
            maximumValue={500 * 1024 * 1024}
            step={1024 * 1024}
            value={videoFilters.minSize}
            onValueChange={v => setVideoFilters({minSize: v})}
            minimumTrackTintColor="#007AFF"
          />
        </View>
        <View style={styles.sliderRow}>
          <Text style={[styles.sliderLabel, isDark && styles.textSecondary]}>
            Max
          </Text>
          <Slider
            style={styles.slider}
            minimumValue={0}
            maximumValue={501 * 1024 * 1024}
            step={1024 * 1024}
            value={
              videoFilters.maxSize === Infinity
                ? 501 * 1024 * 1024
                : videoFilters.maxSize
            }
            onValueChange={v =>
              setVideoFilters({
                maxSize: v >= 501 * 1024 * 1024 ? Infinity : v,
              })
            }
            minimumTrackTintColor="#007AFF"
          />
        </View>

        <Text style={[styles.sectionTitle, isDark && styles.textDark]}>
          AI Queue &amp; Models
        </Text>
        <TouchableOpacity
          style={styles.resetQueueButton}
          onPress={handleResetAIQueue}>
          <Text style={styles.trainButtonText}>Reset AI Queue</Text>
        </TouchableOpacity>

        <Text style={[styles.sectionTitle, isDark && styles.textDark]}>
          Data Management
        </Text>
        <TouchableOpacity
          style={styles.actionButton}
          onPress={handleResetKeepHistory}>
          <Text style={styles.actionButtonText}>Reset Keep History</Text>
        </TouchableOpacity>

        <Text style={[styles.sectionTitle, isDark && styles.textDark]}>
          About &amp; Support
        </Text>
        <TouchableOpacity
          style={styles.linkButton}
          onPress={() => setShowAbout(true)}>
          <Text style={[styles.linkButtonText, isDark && styles.textDark]}>
            About
          </Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.linkButton} onPress={openMailBug}>
          <Text style={[styles.linkButtonText, isDark && styles.textDark]}>
            Report a bug
          </Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.linkButton} onPress={openDonate}>
          <Text style={[styles.linkButtonText, isDark && styles.textDark]}>
            Donate to the project
          </Text>
        </TouchableOpacity>
      </ScrollView>

      <Modal
        visible={showAbout}
        animationType="slide"
        presentationStyle="pageSheet"
        onRequestClose={() => setShowAbout(false)}>
        <SafeAreaView style={[styles.container, isDark && styles.bgDark]}>
          <View style={styles.aboutHeader}>
            <Text style={[styles.aboutTitle, isDark && styles.textDark]}>
              About
            </Text>
            <TouchableOpacity onPress={() => setShowAbout(false)}>
              <Text style={styles.close}>Close</Text>
            </TouchableOpacity>
          </View>
          <ScrollView contentContainerStyle={styles.aboutContent}>
            <Text style={[styles.aboutBody, isDark && styles.textSecondary]}>
              This is the story of why I made this app and why it is free to use
              without ads.
            </Text>
          </ScrollView>
        </SafeAreaView>
      </Modal>
    </>
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
  content: {
    padding: 16,
    paddingBottom: 40,
  },
  sectionTitle: {
    fontSize: 17,
    fontWeight: '700',
    color: '#333',
    marginTop: 24,
    marginBottom: 12,
  },
  textDark: {
    color: '#fff',
  },
  textSecondary: {
    color: '#8e8e93',
  },
  sortContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  sortOption: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
    backgroundColor: '#f0f0f0',
  },
  sortOptionDark: {
    backgroundColor: '#2c2c2e',
  },
  sortOptionActive: {
    backgroundColor: '#007AFF',
  },
  sortText: {
    fontSize: 14,
    color: '#333',
  },
  sortTextActive: {
    color: '#fff',
    fontWeight: '600',
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 10,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#e0e0e0',
  },
  rowDark: {
    borderBottomColor: '#3a3a3c',
  },
  additionalFiltersRow: {
    marginTop: 16,
  },
  label: {
    fontSize: 15,
    color: '#333',
  },
  filterLabel: {
    fontSize: 14,
    color: '#333',
    marginTop: 12,
    marginBottom: 4,
  },
  sliderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 4,
  },
  sliderLabel: {
    width: 32,
    fontSize: 12,
    color: '#8e8e93',
  },
  slider: {
    flex: 1,
    height: 32,
  },
  dateButton: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 8,
    backgroundColor: '#f0f0f0',
  },
  dateButtonDark: {
    backgroundColor: '#2c2c2e',
  },
  dateButtonText: {
    fontSize: 14,
    color: '#333',
  },
  resetQueueButton: {
    marginTop: 8,
    paddingVertical: 14,
    borderRadius: 10,
    backgroundColor: '#5856D6',
    alignItems: 'center',
  },
  trainButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
  },
  checkpointRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#e0e0e0',
  },
  checkpointInfo: {flex: 1, paddingRight: 8},
  checkpointTitle: {fontSize: 15, fontWeight: '600', color: '#333'},
  checkpointActive: {color: '#007AFF'},
  checkpointMeta: {fontSize: 12, color: '#8e8e93', marginTop: 2},
  checkpointDelete: {
    paddingHorizontal: 10,
    paddingVertical: 6,
    backgroundColor: '#ff3b30',
    borderRadius: 6,
  },
  checkpointDeleteText: {color: '#fff', fontSize: 12, fontWeight: '600'},
  actionButton: {
    marginTop: 12,
    paddingVertical: 14,
    borderRadius: 10,
    backgroundColor: '#ff3b30',
    alignItems: 'center',
  },
  actionButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
  },
  linkButton: {
    paddingVertical: 12,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#e0e0e0',
  },
  linkButtonText: {
    fontSize: 16,
    color: '#007AFF',
    fontWeight: '500',
  },
  aboutHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  aboutTitle: {
    fontSize: 22,
    fontWeight: '700',
    color: '#111',
  },
  close: {
    fontSize: 16,
    color: '#007AFF',
    fontWeight: '600',
  },
  aboutContent: {
    padding: 16,
    paddingBottom: 40,
  },
  aboutBody: {
    fontSize: 16,
    lineHeight: 24,
    color: '#444',
  },
});
