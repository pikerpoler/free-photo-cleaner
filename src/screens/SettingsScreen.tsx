import React, {useCallback, useEffect, useState} from 'react';
import {
  Alert,
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
import {resetKeepHistory} from '../services/database';
import {SortMode} from '../types/media';
import {formatFileSize, formatDuration} from '../utils/format';
import {TrainAIScreen} from './TrainAIScreen';

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
  } = useSettingsStore();

  const {hasModel, activeModelInfo, resetModel, refreshModelStatus} =
    useAIStore();

  const [showFromPicker, setShowFromPicker] = useState(false);
  const [showToPicker, setShowToPicker] = useState(false);
  const [showTrainAI, setShowTrainAI] = useState(false);

  useEffect(() => {
    refreshModelStatus();
  }, [refreshModelStatus]);

  const handleResetKeepHistory = useCallback(() => {
    Alert.alert(
      'Reset Keep History',
      'Previously kept media will appear again in the queue. Deleted media will NOT be restored.',
      [
        {text: 'Cancel', style: 'cancel'},
        {
          text: 'Reset',
          style: 'destructive',
          onPress: () => resetKeepHistory(),
        },
      ],
    );
  }, []);

  const handleResetModel = useCallback(() => {
    Alert.alert(
      'Reset trained model',
      'This will delete the active on-device model. Train again to rebuild it.',
      [
        {text: 'Cancel', style: 'cancel'},
        {
          text: 'Reset',
          style: 'destructive',
          onPress: () => resetModel(),
        },
      ],
    );
  }, [resetModel]);

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

  return (
    <>
      <ScrollView
        style={[styles.container, isDark && styles.bgDark]}
        contentContainerStyle={styles.content}>
        <Text style={[styles.sectionTitle, isDark && styles.textDark]}>
          Date Range
        </Text>
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

        <Text style={[styles.sectionTitle, isDark && styles.textDark]}>
          Photos Filters
        </Text>
        <View style={[styles.row, isDark && styles.rowDark]}>
          <Text style={[styles.label, isDark && styles.textDark]}>
            All Photos
          </Text>
          <Switch
            value={photoFilters.mode === 'all'}
            onValueChange={v =>
              setPhotoFilters({mode: v ? 'all' : 'categories'})
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
          AI (Photos)
        </Text>
        <Text style={[styles.filterLabel, isDark && styles.textSecondary]}>
          {hasModel && activeModelInfo
            ? `Active: ${activeModelInfo.modelId} @ ${activeModelInfo.trainResize}px`
            : 'No trained model yet'}
        </Text>
        <TouchableOpacity
          style={styles.trainButton}
          onPress={() => setShowTrainAI(true)}>
          <Text style={styles.trainButtonText}>Train AI</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[
            styles.resetModelButton,
            !hasModel && styles.resetModelButtonDisabled,
          ]}
          disabled={!hasModel}
          onPress={handleResetModel}>
          <Text
            style={[
              styles.resetModelButtonText,
              !hasModel && styles.resetModelButtonTextDisabled,
            ]}>
            Reset Active Model
          </Text>
        </TouchableOpacity>

        <Text style={[styles.sectionTitle, isDark && styles.textDark]}>
          Data Management
        </Text>
        <TouchableOpacity
          style={styles.actionButton}
          onPress={handleResetKeepHistory}>
          <Text style={styles.actionButtonText}>Reset Keep History</Text>
        </TouchableOpacity>
      </ScrollView>

      <Modal
        visible={showTrainAI}
        animationType="slide"
        presentationStyle="pageSheet"
        onRequestClose={() => setShowTrainAI(false)}>
        <SafeAreaView style={[styles.container, isDark && styles.bgDark]}>
          <TrainAIScreen
            onClose={() => {
              setShowTrainAI(false);
              refreshModelStatus();
            }}
          />
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
  trainButton: {
    marginTop: 12,
    paddingVertical: 14,
    borderRadius: 10,
    backgroundColor: '#007AFF',
    alignItems: 'center',
  },
  trainButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
  },
  resetModelButton: {
    marginTop: 8,
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 8,
    backgroundColor: '#ff3b30',
    alignItems: 'center',
  },
  resetModelButtonDisabled: {
    backgroundColor: '#ccc',
  },
  resetModelButtonText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#fff',
  },
  resetModelButtonTextDisabled: {
    color: '#999',
  },
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
});
