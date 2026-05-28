import React, {useCallback} from 'react';
import {
  Alert,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TouchableOpacity,
  View,
  useColorScheme,
} from 'react-native';
import Slider from '@react-native-community/slider';
import {useSettingsStore} from '../stores/settingsStore';
import {resetKeepHistory} from '../services/database';
import {SortMode} from '../types/media';
import {formatFileSize, formatDuration} from '../utils/format';

const SORT_OPTIONS: {label: string; value: SortMode}[] = [
  {label: 'Random', value: 'random'},
  {label: 'Largest first', value: 'largest_first'},
  {label: 'Smallest first', value: 'smallest_first'},
  {label: 'Oldest first', value: 'oldest_first'},
  {label: 'Newest first', value: 'newest_first'},
];

function SortPicker({
  value,
  onChange,
  isDark,
}: {
  value: SortMode;
  onChange: (v: SortMode) => void;
  isDark: boolean;
}) {
  return (
    <View style={styles.sortContainer}>
      {SORT_OPTIONS.map(opt => (
        <TouchableOpacity
          key={opt.value}
          style={[
            styles.sortOption,
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
    setPhotoSort,
    setVideoSort,
    setPhotoFilters,
    setVideoFilters,
  } = useSettingsStore();

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


  // Duration slider: 0 to 600s (10 min), with Infinity as max
  const maxDurationDisplay =
    videoFilters.maxDuration === Infinity
      ? '∞'
      : formatDuration(videoFilters.maxDuration);

  // Size slider: 0 to 500MB, with Infinity as max
  const maxSizeDisplay =
    videoFilters.maxSize === Infinity
      ? '∞'
      : formatFileSize(videoFilters.maxSize);

  return (
    <ScrollView
      style={[styles.container, isDark && styles.bgDark]}
      contentContainerStyle={styles.content}>
      {/* Photo Sort */}
      <Text style={[styles.sectionTitle, isDark && styles.textDark]}>
        Photos Sort
      </Text>
      <SortPicker value={photoSort} onChange={setPhotoSort} isDark={isDark} />

      {/* Photo Filters */}
      <Text style={[styles.sectionTitle, isDark && styles.textDark]}>
        Photos Filters
      </Text>
      <View style={styles.row}>
        <Text style={[styles.label, isDark && styles.textDark]}>All Photos</Text>
        <Switch
          value={photoFilters.mode === 'all'}
          onValueChange={v => setPhotoFilters({mode: v ? 'all' : 'categories'})}
        />
      </View>
      {photoFilters.mode === 'categories' && (
        <>
          <View style={styles.row}>
            <Text style={[styles.label, isDark && styles.textDark]}>
              Screenshots
            </Text>
            <Switch
              value={photoFilters.screenshots}
              onValueChange={v => setPhotoFilters({screenshots: v})}
            />
          </View>
          <View style={styles.row}>
            <Text style={[styles.label, isDark && styles.textDark]}>
              WhatsApp Images
            </Text>
            <Switch
              value={photoFilters.whatsapp}
              onValueChange={v => setPhotoFilters({whatsapp: v})}
            />
          </View>
          <View style={styles.row}>
            <Text style={[styles.label, isDark && styles.textDark]}>
              No Metadata
            </Text>
            <Switch
              value={photoFilters.noMetadata}
              onValueChange={v => setPhotoFilters({noMetadata: v})}
            />
          </View>
        </>
      )}

      {/* Video Sort */}
      <Text style={[styles.sectionTitle, isDark && styles.textDark]}>
        Videos Sort
      </Text>
      <SortPicker value={videoSort} onChange={setVideoSort} isDark={isDark} />

      {/* Video Filters */}
      <Text style={[styles.sectionTitle, isDark && styles.textDark]}>
        Videos Filters
      </Text>

      <Text style={[styles.filterLabel, isDark && styles.textDark]}>
        Duration: {formatDuration(videoFilters.minDuration)} – {maxDurationDisplay}
      </Text>
      <View style={styles.sliderRow}>
        <Text style={[styles.sliderLabel, isDark && styles.textSecondary]}>Min</Text>
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
        <Text style={[styles.sliderLabel, isDark && styles.textSecondary]}>Max</Text>
        <Slider
          style={styles.slider}
          minimumValue={0}
          maximumValue={601}
          step={5}
          value={videoFilters.maxDuration === Infinity ? 601 : videoFilters.maxDuration}
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
        <Text style={[styles.sliderLabel, isDark && styles.textSecondary]}>Min</Text>
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
        <Text style={[styles.sliderLabel, isDark && styles.textSecondary]}>Max</Text>
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

      {/* Actions */}
      <Text style={[styles.sectionTitle, isDark && styles.textDark]}>
        Data Management
      </Text>
      <TouchableOpacity
        style={styles.actionButton}
        onPress={handleResetKeepHistory}>
        <Text style={styles.actionButtonText}>Reset Keep History</Text>
      </TouchableOpacity>
    </ScrollView>
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
