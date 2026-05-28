import React, {useState, useCallback, useRef} from 'react';
import {StyleSheet, Text, TouchableWithoutFeedback, View} from 'react-native';
import Video, {VideoRef} from 'react-native-video';
import {MediaAsset} from '../types/media';
import {formatFileSize, formatDate, formatDuration} from '../utils/format';

interface VideoCardProps {
  asset: MediaAsset;
  isActive: boolean;
}

export function VideoCard({asset, isActive}: VideoCardProps) {
  const [muted, setMuted] = useState(true);
  const videoRef = useRef<VideoRef>(null);

  const toggleMute = useCallback(() => {
    setMuted(prev => !prev);
  }, []);

  // Reset mute when card becomes active
  React.useEffect(() => {
    if (isActive) {
      setMuted(true);
    }
  }, [isActive]);

  return (
    <TouchableWithoutFeedback onPress={toggleMute}>
      <View style={styles.container}>
        <Video
          ref={videoRef}
          source={{uri: asset.uri}}
          style={styles.video}
          resizeMode="cover"
          repeat
          muted={muted}
          paused={!isActive}
          playInBackground={false}
        />
        {muted && (
          <View style={styles.muteIndicator}>
            <Text style={styles.muteText}>🔇</Text>
          </View>
        )}
        <View style={styles.metadataStrip}>
          <Text style={styles.metaText}>{formatDate(asset.creationDate)}</Text>
          <Text style={styles.metaText}>{formatFileSize(asset.fileSize)}</Text>
          {asset.duration !== undefined && (
            <Text style={styles.metaText}>
              {formatDuration(asset.duration)}
            </Text>
          )}
        </View>
      </View>
    </TouchableWithoutFeedback>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  video: {
    flex: 1,
  },
  muteIndicator: {
    position: 'absolute',
    top: 12,
    right: 12,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    borderRadius: 16,
    width: 32,
    height: 32,
    alignItems: 'center',
    justifyContent: 'center',
  },
  muteText: {
    fontSize: 16,
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
});
