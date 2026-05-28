import React, {useCallback, useRef} from 'react';
import {Dimensions, StyleSheet, View} from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withTiming,
  withSpring,
  runOnJS,
  interpolate,
  interpolateColor,
  Extrapolation,
  Easing,
} from 'react-native-reanimated';
import {Gesture, GestureDetector} from 'react-native-gesture-handler';

const {width: SCREEN_WIDTH} = Dimensions.get('window');
const SWIPE_THRESHOLD = SCREEN_WIDTH * 0.25;
const EXIT_DISTANCE = SCREEN_WIDTH * 1.3;
const EXIT_DURATION_MS = 350;

interface SwipeCardProps {
  children: React.ReactNode;
  behindContent?: React.ReactNode;
  onSwipeLeft: () => void;
  onSwipeRight: () => void;
  cardKey?: string;
  enabled?: boolean;
}

export function SwipeCard({
  children,
  behindContent,
  onSwipeLeft,
  onSwipeRight,
  cardKey,
  enabled = true,
}: SwipeCardProps) {
  const translateX = useSharedValue(0);
  const translateY = useSharedValue(0);
  const isExiting = useRef(false);

  const handleSwipeLeft = useCallback(() => {
    onSwipeLeft();
  }, [onSwipeLeft]);

  const handleSwipeRight = useCallback(() => {
    onSwipeRight();
  }, [onSwipeRight]);

  const panGesture = Gesture.Pan()
    .enabled(enabled)
    .onUpdate(event => {
      if (isExiting.current) return;
      translateX.value = event.translationX;
      translateY.value = event.translationY * 0.15;
    })
    .onEnd(event => {
      if (isExiting.current) return;

      if (event.translationX < -SWIPE_THRESHOLD) {
        isExiting.current = true;
        translateX.value = withTiming(
          -EXIT_DISTANCE,
          {duration: EXIT_DURATION_MS, easing: Easing.inOut(Easing.ease)},
          finished => {
            if (finished) {
              runOnJS(handleSwipeLeft)();
            }
          },
        );
      } else if (event.translationX > SWIPE_THRESHOLD) {
        isExiting.current = true;
        translateX.value = withTiming(
          EXIT_DISTANCE,
          {duration: EXIT_DURATION_MS, easing: Easing.inOut(Easing.ease)},
          finished => {
            if (finished) {
              runOnJS(handleSwipeRight)();
            }
          },
        );
      } else {
        translateX.value = withSpring(0, {damping: 20, stiffness: 200});
        translateY.value = withSpring(0, {damping: 20, stiffness: 200});
      }
    });

  const cardStyle = useAnimatedStyle(() => {
    const rotation = interpolate(
      translateX.value,
      [-SCREEN_WIDTH, 0, SCREEN_WIDTH],
      [-10, 0, 10],
      Extrapolation.CLAMP,
    );

    return {
      transform: [
        {translateX: translateX.value},
        {translateY: translateY.value},
        {rotateZ: `${rotation}deg`},
      ],
    };
  });

  const borderStyle = useAnimatedStyle(() => {
    const absX = Math.abs(translateX.value);
    const borderWidth = interpolate(
      absX,
      [0, SWIPE_THRESHOLD * 0.4, SWIPE_THRESHOLD],
      [0, 1.5, 4],
      Extrapolation.CLAMP,
    );

    const borderColor = interpolateColor(
      translateX.value,
      [-SWIPE_THRESHOLD, -SWIPE_THRESHOLD * 0.2, 0, SWIPE_THRESHOLD * 0.2, SWIPE_THRESHOLD],
      ['#ff3b30', '#ff3b3000', 'transparent', '#34c75900', '#34c759'],
    );

    return {borderWidth, borderColor};
  });

  const behindScale = useAnimatedStyle(() => {
    const absX = Math.abs(translateX.value);
    const scale = interpolate(
      absX,
      [0, SWIPE_THRESHOLD, EXIT_DISTANCE],
      [0.92, 0.96, 1],
      Extrapolation.CLAMP,
    );
    const opacity = interpolate(
      absX,
      [0, SWIPE_THRESHOLD * 0.5, SWIPE_THRESHOLD],
      [0.6, 0.8, 1],
      Extrapolation.CLAMP,
    );
    return {transform: [{scale}], opacity};
  });

  const deleteIndicatorStyle = useAnimatedStyle(() => ({
    opacity: interpolate(
      translateX.value,
      [-SWIPE_THRESHOLD, -SWIPE_THRESHOLD * 0.3, 0],
      [1, 0.3, 0],
      Extrapolation.CLAMP,
    ),
  }));

  const keepIndicatorStyle = useAnimatedStyle(() => ({
    opacity: interpolate(
      translateX.value,
      [0, SWIPE_THRESHOLD * 0.3, SWIPE_THRESHOLD],
      [0, 0.3, 1],
      Extrapolation.CLAMP,
    ),
  }));

  React.useEffect(() => {
    isExiting.current = false;
    translateX.value = 0;
    translateY.value = 0;
  }, [cardKey, translateX, translateY]);

  return (
    <View style={styles.container}>
      {behindContent && (
        <Animated.View style={[styles.card, styles.behindCard, behindScale]}>
          {behindContent}
        </Animated.View>
      )}
      <GestureDetector gesture={panGesture}>
        <Animated.View style={[styles.card, cardStyle, borderStyle]}>
          <Animated.View style={[styles.indicator, styles.deleteIndicator, deleteIndicatorStyle]}>
            <Animated.Text style={[styles.indicatorText, styles.deleteText]}>DELETE</Animated.Text>
          </Animated.View>
          <Animated.View style={[styles.indicator, styles.keepIndicator, keepIndicatorStyle]}>
            <Animated.Text style={[styles.indicatorText, styles.keepText]}>KEEP</Animated.Text>
          </Animated.View>
          {children}
        </Animated.View>
      </GestureDetector>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  card: {
    width: SCREEN_WIDTH - 32,
    aspectRatio: 3 / 4,
    borderRadius: 16,
    overflow: 'hidden',
    backgroundColor: '#1a1a1a',
    elevation: 5,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.25,
    shadowRadius: 8,
  },
  behindCard: {
    position: 'absolute',
  },
  indicator: {
    position: 'absolute',
    top: 20,
    zIndex: 10,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
    borderWidth: 2,
  },
  deleteIndicator: {
    right: 20,
    borderColor: '#ff3b30',
    backgroundColor: 'rgba(255, 59, 48, 0.2)',
  },
  keepIndicator: {
    left: 20,
    borderColor: '#34c759',
    backgroundColor: 'rgba(52, 199, 89, 0.2)',
  },
  indicatorText: {
    fontSize: 16,
    fontWeight: '700',
  },
  deleteText: {
    color: '#ff3b30',
  },
  keepText: {
    color: '#34c759',
  },
});
