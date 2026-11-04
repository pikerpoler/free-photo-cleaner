import React, {useCallback, useEffect, useMemo, useState} from 'react';
import {Dimensions, StyleSheet, View} from 'react-native';
import Animated, {
  cancelAnimation,
  useAnimatedReaction,
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  runOnJS,
  interpolate,
  Extrapolation,
} from 'react-native-reanimated';
import {Gesture, GestureDetector} from 'react-native-gesture-handler';

const {width: SCREEN_WIDTH} = Dimensions.get('window');
const SWIPE_THRESHOLD = SCREEN_WIDTH * 0.25;
const VELOCITY_THRESHOLD = 800;
const EXIT_DISTANCE = SCREEN_WIDTH * 1.5;
const RECYCLE_DISTANCE = SCREEN_WIDTH;

function exitSpring(velocity: number) {
  'worklet';
  return {
    damping: 20,
    stiffness: 200,
    mass: 1,
    velocity,
    overshootClamping: true,
  };
}

function returnSpring(velocity: number) {
  'worklet';
  return {
    damping: 120,
    stiffness: 900,
    mass: 4,
    velocity,
  };
}

function cardTransform(translateX: number, translateY: number) {
  'worklet';
  const rotation = interpolate(
    translateX,
    [-SCREEN_WIDTH, 0, SCREEN_WIDTH],
    [-10, 0, 10],
    Extrapolation.CLAMP,
  );
  return [
    {translateX},
    {translateY},
    {rotateZ: `${rotation}deg`},
  ];
}

type SwipeDirection = 'left' | 'right';

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
  const startX0 = useSharedValue(0);
  const startY0 = useSharedValue(0);
  const startX1 = useSharedValue(0);
  const startY1 = useSharedValue(0);

  /** -1 = none, 0 | 1 = slot flying off screen */
  const flyingSlot = useSharedValue(-1);
  /** 0 = none, 1 = left, 2 = right */
  const pendingCommit = useSharedValue(0);

  const topSlotSV = useSharedValue(0);
  const [topSlot, setTopSlot] = useState(0);

  useEffect(() => {
    topSlotSV.value = topSlot;
  }, [topSlot, topSlotSV]);

  const slot0Content = topSlot === 0 ? children : behindContent;
  const slot1Content = topSlot === 0 ? behindContent : children;

  const applyAdvance = useCallback(
    (direction: SwipeDirection, newTopSlot: 0 | 1) => {
      if (direction === 'left') {
        onSwipeLeft();
      } else {
        onSwipeRight();
      }
      setTopSlot(newTopSlot);
    },
    [onSwipeLeft, onSwipeRight],
  );

  const panGesture = useMemo(() => {
    const shouldCommitLeft = (x: number, vx: number) => {
      'worklet';
      return (
        x < -SWIPE_THRESHOLD ||
        (Math.abs(vx) > VELOCITY_THRESHOLD && vx < 0)
      );
    };

    const shouldCommitRight = (x: number, vx: number) => {
      'worklet';
      return (
        x > SWIPE_THRESHOLD ||
        (Math.abs(vx) > VELOCITY_THRESHOLD && vx > 0)
      );
    };

    const springFlyingOff = (
      slot: 0 | 1,
      direction: SwipeDirection,
      vx: number,
      vy: number,
    ) => {
      'worklet';
      const tx = slot === 0 ? translateX0 : translateX1;
      const ty = slot === 0 ? translateY0 : translateY1;
      const target = direction === 'left' ? -EXIT_DISTANCE : EXIT_DISTANCE;
      tx.value = withSpring(target, exitSpring(vx), finished => {
        if (!finished) {
          return;
        }
      });
      ty.value = withSpring(0, exitSpring(vy));
    };

    const beginFly = (
      slot: 0 | 1,
      direction: SwipeDirection,
      vx: number,
      vy: number,
    ) => {
      'worklet';
      if (flyingSlot.value !== -1) {
        pendingCommit.value = direction === 'left' ? 1 : 2;
        return;
      }
      flyingSlot.value = slot;
      topSlotSV.value = slot === 0 ? 1 : 0;
      springFlyingOff(slot, direction, vx, vy);
    };

    return Gesture.Pan()
      .enabled(enabled)
      .onBegin(() => {
        const active = topSlotSV.value === 0 ? 0 : 1;
        if (active === 0) {
          cancelAnimation(translateX0);
          cancelAnimation(translateY0);
          startX0.value = translateX0.value;
          startY0.value = translateY0.value;
        } else {
          cancelAnimation(translateX1);
          cancelAnimation(translateY1);
          startX1.value = translateX1.value;
          startY1.value = translateY1.value;
        }
      })
      .onUpdate(event => {
        const active = topSlotSV.value === 0 ? 0 : 1;
        if (active === 0 && flyingSlot.value === 0) {
          return;
        }
        if (active === 1 && flyingSlot.value === 1) {
          return;
        }
        if (active === 0) {
          translateX0.value = startX0.value + event.translationX;
          translateY0.value = startY0.value + event.translationY * 0.15;
        } else {
          translateX1.value = startX1.value + event.translationX;
          translateY1.value = startY1.value + event.translationY * 0.15;
        }
      })
      .onEnd(event => {
        const active = topSlotSV.value === 0 ? 0 : 1;
        if (active === 0 && flyingSlot.value === 0) {
          return;
        }
        if (active === 1 && flyingSlot.value === 1) {
          return;
        }

        const tx = active === 0 ? translateX0 : translateX1;
        const ty = active === 0 ? translateY0 : translateY1;
        const x = tx.value;
        const vx = event.velocityX;
        const vy = event.velocityY;

        if (shouldCommitLeft(x, vx)) {
          beginFly(active, 'left', vx, vy);
          return;
        }
        if (shouldCommitRight(x, vx)) {
          beginFly(active, 'right', vx, vy);
          return;
        }

        tx.value = withSpring(0, returnSpring(vx));
        ty.value = withSpring(0, returnSpring(vy));
      });
  }, [
    enabled,
    applyAdvance,
    flyingSlot,
    pendingCommit,
    startX0,
    startY0,
    startX1,
    startY1,
    topSlotSV,
    translateX0,
    translateX1,
    translateY0,
    translateY1,
  ]);

  useAnimatedReaction(
    () => {
      const fs = flyingSlot.value;
      if (fs === 0) {
        return Math.abs(translateX0.value);
      }
      if (fs === 1) {
        return Math.abs(translateX1.value);
      }
      return 0;
    },
    (absX, prev) => {
      if (flyingSlot.value === -1) {
        return;
      }
      if (absX < RECYCLE_DISTANCE) {
        return;
      }
      if (prev !== null && absX <= prev) {
        return;
      }
      const fs = flyingSlot.value;
      const direction: SwipeDirection =
        fs === 0
          ? translateX0.value < 0
            ? 'left'
            : 'right'
          : translateX1.value < 0
            ? 'left'
            : 'right';

      const tx = fs === 0 ? translateX0 : translateX1;
      const ty = fs === 0 ? translateY0 : translateY1;
      cancelAnimation(tx);
      cancelAnimation(ty);
      tx.value = 0;
      ty.value = 0;
      flyingSlot.value = -1;
      const newTop = topSlotSV.value === 0 ? 0 : 1;
      runOnJS(applyAdvance)(direction, newTop);

      const pending = pendingCommit.value;
      if (pending !== 0) {
        pendingCommit.value = 0;
        const pendingDir: SwipeDirection = pending === 1 ? 'left' : 'right';
        const activeSlot = topSlotSV.value === 0 ? 0 : 1;
        flyingSlot.value = activeSlot;
        topSlotSV.value = activeSlot === 0 ? 1 : 0;
        const ptx = activeSlot === 0 ? translateX0 : translateX1;
        const pty = activeSlot === 0 ? translateY0 : translateY1;
        const target =
          pendingDir === 'left' ? -EXIT_DISTANCE : EXIT_DISTANCE;
        ptx.value = withSpring(target, exitSpring(0));
        pty.value = withSpring(0, exitSpring(0));
      }
    },
    [applyAdvance],
  );

  const slot0Style = useAnimatedStyle(() => {
    if (flyingSlot.value === 0 || topSlotSV.value === 0) {
      return {
        zIndex: flyingSlot.value === 0 ? 3 : 2,
        opacity: 1,
        transform: cardTransform(translateX0.value, translateY0.value),
      };
    }
    const absX = Math.abs(
      topSlotSV.value === 1 ? translateX1.value : translateX0.value,
    );
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
    };
  });

  const slot1Style = useAnimatedStyle(() => {
    if (flyingSlot.value === 1 || topSlotSV.value === 1) {
      return {
        zIndex: flyingSlot.value === 1 ? 3 : 2,
        opacity: 1,
        transform: cardTransform(translateX1.value, translateY1.value),
      };
    }
    const absX = Math.abs(
      topSlotSV.value === 0 ? translateX0.value : translateX1.value,
    );
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
    };
  });

  const slot0DeleteStyle = useAnimatedStyle(() => ({
    opacity:
      topSlotSV.value === 0 && flyingSlot.value !== 0
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
      topSlotSV.value === 0 && flyingSlot.value !== 0
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
      topSlotSV.value === 1 && flyingSlot.value !== 1
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
      topSlotSV.value === 1 && flyingSlot.value !== 1
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
