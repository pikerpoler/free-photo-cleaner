import React from 'react';
import {StyleSheet, Text, TouchableOpacity, View, useColorScheme} from 'react-native';
import {RotateCcwIcon} from './RotateCcwIcon';
import {formatFileSize} from '../utils/format';
import type {ReviewAggregateStats} from '../stores/queueStore';

type Props = {
  stats: ReviewAggregateStats;
  canUndo: boolean;
  pendingCount: number;
  batchWarningThreshold?: number;
  onUndo: () => void;
  onDelete: () => void;
};

export function ReviewHeader({
  stats,
  canUndo,
  pendingCount,
  batchWarningThreshold = 1024,
  onUndo,
  onDelete,
}: Props) {
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';
  const showWarning = pendingCount >= batchWarningThreshold;

  return (
    <View style={styles.row}>
      <View style={styles.leftCol}>
        <Text style={[styles.statLine, isDark && styles.textDark]}>
          {stats.toReviewCount} To review {formatFileSize(stats.toReviewBytes)}
        </Text>
        <Text style={[styles.statLine, isDark && styles.textDark]}>
          {stats.pendingDeleteCount} Marked for deletion{' '}
          {formatFileSize(stats.pendingDeleteBytes)}
        </Text>
        <Text style={[styles.statLine, isDark && styles.textDark]}>
          {stats.keptCount} Kept {formatFileSize(stats.keptBytes)}
        </Text>
        <Text style={[styles.statLine, isDark && styles.textDark]}>
          {stats.deletedCount} Deleted {formatFileSize(stats.deletedBytes)}
        </Text>
      </View>
      <View style={styles.rightCol}>
        <TouchableOpacity
          onPress={onDelete}
          disabled={pendingCount <= 0}
          style={[
            styles.deleteButton,
            pendingCount <= 0 && styles.btnDisabled,
          ]}>
          {showWarning && pendingCount > 0 && (
            <Text style={styles.warningIcon}>⚠️</Text>
          )}
          <Text
            style={[
              styles.deleteText,
              pendingCount <= 0 && styles.textDisabled,
            ]}>
            Move {pendingCount} to trash
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          onPress={onUndo}
          disabled={!canUndo}
          style={[styles.undoButton, !canUndo && styles.btnDisabled]}
          accessibilityRole="button"
          accessibilityLabel="Undo">
          <RotateCcwIcon color="#fff" size={22} strokeWidth={2.5} />
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    paddingHorizontal: 16,
    paddingVertical: 8,
    gap: 12,
  },
  leftCol: {
    flex: 1,
    gap: 4,
  },
  rightCol: {
    justifyContent: 'space-between',
    gap: 8,
    alignItems: 'stretch',
  },
  statLine: {
    fontSize: 13,
    color: '#444',
    fontWeight: '500',
  },
  textDark: {
    color: '#e5e5ea',
  },
  undoButton: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#007AFF',
    alignItems: 'center',
    justifyContent: 'center',
    alignSelf: 'flex-end',
  },
  deleteButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 14,
    paddingVertical: 12,
    backgroundColor: '#ff3b30',
    borderRadius: 14,
    gap: 4,
    maxWidth: 160,
  },
  btnDisabled: {
    opacity: 0.4,
  },
  deleteText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
    textAlign: 'center',
  },
  textDisabled: {
    color: '#ccc',
  },
  warningIcon: {
    fontSize: 12,
  },
});