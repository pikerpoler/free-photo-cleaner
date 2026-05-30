import React, {useCallback, useState} from 'react';
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
  enabled = true,
}: SwipeCardProps) {
  const translateX0 = useSharedValue(0);
  const translateY0 = useSharedValue(0);
  const translateX1 = useSharedValue(0);
  const translateY1 = useSharedValue(0);
  const isAnimatingExit = useSharedValue(0);

  const topSlotSV = useSharedValue(0);
  const [topSlot, setTopSlot] = useState(0);

  const slot0Content = topSlot === 0 ? children : behindContent;
  const slot1Content = topSlot === 0 ? behindContent : children;

  const advanceIndex = useCallback(
    (direction: 'left' | 'right') => {
      if (topSlot === 0) {
        translateX0.value = 0;
        translateY0.value = 0;
      } else {
        translateX1.value = 0;
        translateY1.value = 0;
      }

      if (direction === 'left') {
        onSwipeLeft();
      } else {
        onSwipeRight();
      }
      setTopSlot(prev => (prev === 0 ? 1 : 0));
    },
    [topSlot, onSwipeLeft, onSwipeRight, translateX0, translateY0, translateX1, translateY1],
  );

  const panGesture = Gesture.Pan()
    .enabled(enabled)
    .onUpdate(event => {
      if (isAnimatingExit.value === 1) {
        return;
      }
      if (topSlotSV.value === 0) {
        translateX0.value = event.translationX;
        translateY0.value = event.translationY * 0.15;
      } else {
        translateX1.value = event.translationX;
        translateY1.value = event.translationY * 0.15;
      }
    })
    .onEnd(event => {
      if (isAnimatingExit.value === 1) {
        return;
      }

      const isSlot0 = topSlotSV.value === 0;
      const tx = isSlot0 ? translateX0 : translateX1;
      const ty = isSlot0 ? translateY0 : translateY1;

      if (event.translationX < -SWIPE_THRESHOLD) {
        isAnimatingExit.value = 1;
        tx.value = withTiming(
          -EXIT_DISTANCE,
          {duration: EXIT_DURATION_MS, easing: Easing.inOut(Easing.ease)},
          finished => {
            if (finished) {
              topSlotSV.value = isSlot0 ? 1 : 0;
              isAnimatingExit.value = 0;
              runOnJS(advanceIndex)('left');
            }
          },
        );
      } else if (event.translationX > SWIPE_THRESHOLD) {
        isAnimatingExit.value = 1;
        tx.value = withTiming(
          EXIT_DISTANCE,
          {duration: EXIT_DURATION_MS, easing: Easing.inOut(Easing.ease)},
          finished => {
            if (finished) {
              topSlotSV.value = isSlot0 ? 1 : 0;
              isAnimatingExit.value = 0;
              runOnJS(advanceIndex)('right');
            }
          },
        );
      } else {
        tx.value = withSpring(0, {damping: 20, stiffness: 200});
        ty.value = withSpring(0, {damping: 20, stiffness: 200});
      }
    });

  // ── Slot 0 animated style ──────────────────────────────────────────

  const slot0Style = useAnimatedStyle(() => {
    if (topSlotSV.value === 0) {
      const rotation = interpolate(
        translateX0.value,
        [-SCREEN_WIDTH, 0, SCREEN_WIDTH],
        [-10, 0, 10],
        Extrapolation.CLAMP,
      );
      const absX = Math.abs(translateX0.value);
      return {
        zIndex: 2,
        opacity: 1,
        transform: [
          {translateX: translateX0.value},
          {translateY: translateY0.value},
          {rotateZ: `${rotation}deg`},
        ],
        borderWidth: interpolate(
          absX,
          [0, SWIPE_THRESHOLD * 0.4, SWIPE_THRESHOLD],
          [0, 1.5, 4],
          Extrapolation.CLAMP,
        ),
        borderColor: interpolateColor(
          translateX0.value,
          [-SWIPE_THRESHOLD, -SWIPE_THRESHOLD * 0.2, 0, SWIPE_THRESHOLD * 0.2, SWIPE_THRESHOLD],
          ['#ff3b30', '#ff3b3000', 'transparent', '#34c75900', '#34c759'],
        ),
      };
    }
    const absX = Math.abs(translateX1.value);
    return {
      zIndex: 1,
      opacity: interpolate(
        absX,
        [0, SWIPE_THRESHOLD * 0.5, SWIPE_THRESHOLD],
        [0.6, 0.8, 1],
        Extrapolation.CLAMP,
      ),
      transform: [
        {
          scale: interpolate(
            absX,
            [0, SWIPE_THRESHOLD, EXIT_DISTANCE],
            [0.92, 0.96, 1],
            Extrapolation.CLAMP,
          ),
        },
      ],
      borderWidth: 0,
      borderColor: 'transparent',
    };
  });

  // ── Slot 1 animated style ──────────────────────────────────────────

  const slot1Style = useAnimatedStyle(() => {
    if (topSlotSV.value === 1) {
      const rotation = interpolate(
        translateX1.value,
        [-SCREEN_WIDTH, 0, SCREEN_WIDTH],
        [-10, 0, 10],
        Extrapolation.CLAMP,
      );
      const absX = Math.abs(translateX1.value);
      return {
        zIndex: 2,
        opacity: 1,
        transform: [
          {translateX: translateX1.value},
          {translateY: translateY1.value},
          {rotateZ: `${rotation}deg`},
        ],
        borderWidth: interpolate(
          absX,
          [0, SWIPE_THRESHOLD * 0.4, SWIPE_THRESHOLD],
          [0, 1.5, 4],
          Extrapolation.CLAMP,
        ),
        borderColor: interpolateColor(
          translateX1.value,
          [-SWIPE_THRESHOLD, -SWIPE_THRESHOLD * 0.2, 0, SWIPE_THRESHOLD * 0.2, SWIPE_THRESHOLD],
          ['#ff3b30', '#ff3b3000', 'transparent', '#34c75900', '#34c759'],
        ),
      };
    }
    const absX = Math.abs(translateX0.value);
    return {
      zIndex: 1,
      opacity: interpolate(
        absX,
        [0, SWIPE_THRESHOLD * 0.5, SWIPE_THRESHOLD],
        [0.6, 0.8, 1],
        Extrapolation.CLAMP,
      ),
      transform: [
        {
          scale: interpolate(
            absX,
            [0, SWIPE_THRESHOLD, EXIT_DISTANCE],
            [0.92, 0.96, 1],
            Extrapolation.CLAMP,
          ),
        },
      ],
      borderWidth: 0,
      borderColor: 'transparent',
    };
  });

  // ── Indicator styles per slot ──────────────────────────────────────

  const slot0DeleteStyle = useAnimatedStyle(() => ({
    opacity:
      topSlotSV.value === 0
        ? interpolate(
            translateX0.value,
            [-SWIPE_THRESHOLD, -SWIPE_THRESHOLD * 0.3, 0],
            [1, 0.3, 0],
            Extrapolation.CLAMP,
          )
        : 0,
  }));

  const slot0KeepStyle = useAnimatedStyle(() => ({
    opacity:
      topSlotSV.value === 0
        ? interpolate(
            translateX0.value,
            [0, SWIPE_THRESHOLD * 0.3, SWIPE_THRESHOLD],
            [0, 0.3, 1],
            Extrapolation.CLAMP,
          )
        : 0,
  }));

  const slot1DeleteStyle = useAnimatedStyle(() => ({
    opacity:
      topSlotSV.value === 1
        ? interpolate(
            translateX1.value,
            [-SWIPE_THRESHOLD, -SWIPE_THRESHOLD * 0.3, 0],
            [1, 0.3, 0],
            Extrapolation.CLAMP,
          )
        : 0,
  }));

  const slot1KeepStyle = useAnimatedStyle(() => ({
    opacity:
      topSlotSV.value === 1
        ? interpolate(
            translateX1.value,
            [0, SWIPE_THRESHOLD * 0.3, SWIPE_THRESHOLD],
            [0, 0.3, 1],
            Extrapolation.CLAMP,
          )
        : 0,
  }));

  return (
    <View style={styles.container}>
      <GestureDetector gesture={panGesture}>
        <Animated.View style={styles.cardWrapper}>
          <Animated.View style={[styles.card, slot0Style]}>
            <Animated.View
              style={[
                styles.indicator,
                styles.deleteIndicator,
                slot0DeleteStyle,
              ]}>
              <Animated.Text style={[styles.indicatorText, styles.deleteText]}>
                DELETE
              </Animated.Text>
            </Animated.View>
            <Animated.View
              style={[
                styles.indicator,
                styles.keepIndicator,
                slot0KeepStyle,
              ]}>
              <Animated.Text style={[styles.indicatorText, styles.keepText]}>
                KEEP
              </Animated.Text>
            </Animated.View>
            {slot0Content}
          </Animated.View>

          <Animated.View style={[styles.card, slot1Style]}>
            <Animated.View
              style={[
                styles.indicator,
                styles.deleteIndicator,
                slot1DeleteStyle,
              ]}>
              <Animated.Text style={[styles.indicatorText, styles.deleteText]}>
                DELETE
              </Animated.Text>
            </Animated.View>
            <Animated.View
              style={[
                styles.indicator,
                styles.keepIndicator,
                slot1KeepStyle,
              ]}>
              <Animated.Text style={[styles.indicatorText, styles.keepText]}>
                KEEP
              </Animated.Text>
            </Animated.View>
            {slot1Content}
          </Animated.View>
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
  cardWrapper: {
    width: SCREEN_WIDTH - 32,
    aspectRatio: 3 / 4,
  },
  card: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    borderRadius: 16,
    overflow: 'hidden',
    backgroundColor: '#1a1a1a',
    elevation: 5,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.25,
    shadowRadius: 8,
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
