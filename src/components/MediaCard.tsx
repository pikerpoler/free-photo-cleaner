import React from 'react';
import {Image, StyleSheet, Text, View} from 'react-native';
import {MediaAsset} from '../types/media';
import {formatFileSize, formatDate, formatResolution, formatDuration} from '../utils/format';

interface MediaCardProps {
  asset: MediaAsset;
  /** Reserve the AI score slot so the strip doesn't reflow while scoring. */
  showScore?: boolean;
  score?: number | null;
}

export const MediaCard = React.memo(
  function MediaCard({asset, showScore = false, score}: MediaCardProps) {
    return (
      <View style={styles.container}>
        <Image
          key={asset.id}
          style={styles.image}
          source={{uri: asset.uri}}
          resizeMode="contain"
        />
        <View style={styles.metadataStrip}>
          <Text style={styles.metaText}>{formatDate(asset.creationDate)}</Text>
          <Text style={styles.metaText}>{formatFileSize(asset.fileSize)}</Text>
          {asset.width > 0 && (
            <Text style={styles.metaText}>
              {formatResolution(asset.width, asset.height)}
            </Text>
          )}
          {asset.duration !== undefined && asset.duration > 0 && (
            <Text style={styles.metaText}>{formatDuration(asset.duration)}</Text>
          )}
          {showScore && (
            <Text style={[styles.metaText, styles.scoreText]}>
              {score != null ? `AI ${score.toFixed(2)}` : 'AI …'}
            </Text>
          )}
        </View>
      </View>
    );
  },
  (prev, next) =>
    prev.asset.id === next.asset.id &&
    prev.showScore === next.showScore &&
    prev.score === next.score,
);

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  image: {
    flex: 1,
  },
  metadataStrip: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 12,
    paddingVertical: 8,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
  },
  metaText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '500',
  },
  scoreText: {
    minWidth: 58,
    textAlign: 'right',
    fontVariant: ['tabular-nums'],
  },
});
