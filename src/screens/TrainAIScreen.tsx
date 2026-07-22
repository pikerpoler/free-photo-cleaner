import React, {useCallback, useEffect, useMemo, useState} from 'react';
import {
  ActivityIndicator,
  Alert,
  Platform,
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
import {useAIStore} from '../stores/aiStore';
import {
  getLabeledCounts,
  getLabeledPhotoSamples,
} from '../services/database';
import {
  isTrainerAvailable,
  startTraining,
  stopTraining,
  subscribeTrainingProgress,
} from '../services/cnnTrainer';
import {
  DEFAULT_AUGMENTATIONS,
  LR_SCHEDULERS,
  LrSchedulerId,
  MIN_LABELED_FOR_TRAIN,
  TRAIN_RESIZE_OPTIONS,
  TRAIN_TEST_RATIO,
  TRAINABLE_MODELS,
  TrainableModelId,
  TrainingProgress,
} from '../ai/cnnTypes';

interface TrainAIScreenProps {
  onClose?: () => void;
}

type HistoryPoint = {
  epoch: number;
  trainLoss: number;
  testLoss: number;
  trainAcc: number;
  testAcc: number;
};

function MetricChart({
  points,
  isDark,
  mode,
  emptyHint,
}: {
  points: HistoryPoint[];
  isDark: boolean;
  mode: 'loss' | 'acc';
  emptyHint: string;
}) {
  const width = 320;
  const height = 140;
  const pad = 8;

  if (points.length === 0) {
    return (
      <View style={[styles.chartEmpty, {height}]}>
        <Text style={[styles.hint, isDark && styles.textSecondary]}>
          {emptyHint}
        </Text>
      </View>
    );
  }

  const trainVals = points.map(p => (mode === 'loss' ? p.trainLoss : p.trainAcc));
  const testVals = points.map(p => (mode === 'loss' ? p.testLoss : p.testAcc));
  const maxV =
    mode === 'acc'
      ? 1
      : Math.max(0.01, ...trainVals, ...testVals);

  const toX = (i: number) =>
    pad +
    (points.length === 1
      ? width / 2
      : (i / (points.length - 1)) * (width - pad * 2));
  const toY = (v: number) => height - pad - (v / maxV) * (height - pad * 2);

  const renderDots = (key: string, vals: number[], color: string) =>
    vals.map((v, i) => (
      <View
        key={`${key}-${i}`}
        style={{
          position: 'absolute',
          left: toX(i) - 3,
          top: toY(v) - 3,
          width: 6,
          height: 6,
          borderRadius: 3,
          backgroundColor: color,
        }}
      />
    ));

  return (
    <View style={[styles.chart, {width, height}, isDark && styles.chartDark]}>
      {renderDots('train', trainVals, '#007AFF')}
      {renderDots('test', testVals, '#FF9500')}
      <Text style={styles.chartLegend}>
        <Text style={{color: '#007AFF'}}>Train</Text>
        {'  '}
        <Text style={{color: '#FF9500'}}>Test</Text>
        {mode === 'loss' ? `  max ${maxV.toFixed(3)}` : '  0–1'}
      </Text>
    </View>
  );
}

function AugToggle({
  label,
  value,
  onChange,
  disabled,
  isDark,
  children,
}: {
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
  isDark: boolean;
  children?: React.ReactNode;
}) {
  return (
    <View style={styles.augBlock}>
      <View style={styles.augRow}>
        <Text style={[styles.label, isDark && styles.textDark]}>{label}</Text>
        <Switch
          value={value}
          disabled={disabled}
          onValueChange={onChange}
          trackColor={{false: isDark ? '#39393D' : '#e9e9ea', true: '#34C759'}}
        />
      </View>
      {value && children}
    </View>
  );
}

export function TrainAIScreen({onClose}: TrainAIScreenProps) {
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';

  const {
    aiModel,
    aiBatchSize,
    aiStepSize,
    aiEpochs,
    aiTrainResize,
    aiLrScheduler,
    aiStepGamma,
    aiStepSizeEpochs,
    aiExpDecay,
    aiAugmentations,
    setAIModel,
    setAIBatchSize,
    setAIStepSize,
    setAIEpochs,
    setAITrainResize,
    setAILrScheduler,
    setAIStepGamma,
    setAIStepSizeEpochs,
    setAIExpDecay,
    setAIAugmentations,
  } = useSettingsStore();

  const {refreshModelStatus, activeModelInfo, hasModel} = useAIStore();

  const [available, setAvailable] = useState(Platform.OS === 'ios');
  const [isTraining, setIsTraining] = useState(false);
  const [counts, setCounts] = useState({kept: 0, pendingDelete: 0});
  const [progress, setProgress] = useState<TrainingProgress | null>(null);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [statusText, setStatusText] = useState('');

  const labeledTotal = counts.kept + counts.pendingDelete;
  const trainSize = Math.max(1, Math.floor(labeledTotal * TRAIN_TEST_RATIO));
  const testSize = Math.max(0, labeledTotal - trainSize);
  const canTrain =
    available &&
    !isTraining &&
    labeledTotal >= MIN_LABELED_FOR_TRAIN &&
    counts.kept > 0 &&
    counts.pendingDelete > 0;

  const augs = aiAugmentations ?? DEFAULT_AUGMENTATIONS;

  useEffect(() => {
    setCounts(getLabeledCounts());
    refreshModelStatus();
    isTrainerAvailable().then(setAvailable);
  }, [refreshModelStatus]);

  useEffect(() => {
    return subscribeTrainingProgress(p => {
      setProgress(p);
      setHistory(prev => {
        const next = prev.filter(h => h.epoch !== p.epoch);
        next.push({
          epoch: p.epoch,
          trainLoss: p.trainLoss,
          testLoss: p.testLoss,
          trainAcc: p.trainAcc ?? 0,
          testAcc: p.testAcc ?? 0,
        });
        next.sort((a, b) => a.epoch - b.epoch);
        return next;
      });
    });
  }, []);

  const heavyWarning = useMemo(() => {
    if (aiTrainResize >= 256 || aiModel === 'cnn-small') {
      return 'High resize and/or CNN Small may be slow on weaker devices.';
    }
    return null;
  }, [aiTrainResize, aiModel]);

  const handleTrain = useCallback(async () => {
    if (!canTrain) return;
    const samples = getLabeledPhotoSamples();
    if (samples.length < MIN_LABELED_FOR_TRAIN) {
      Alert.alert(
        'Not enough labels',
        `Need at least ${MIN_LABELED_FOR_TRAIN} labeled photos.`,
      );
      return;
    }

    setIsTraining(true);
    setHistory([]);
    setProgress(null);
    setStatusText('Starting training...');

    try {
      const result = await startTraining({
        uris: samples.map(s => s.uri),
        labels: samples.map(s => s.label),
        modelId: aiModel,
        batchSize: aiBatchSize,
        learningRate: aiStepSize,
        epochs: aiEpochs,
        trainResize: aiTrainResize,
        trainRatio: TRAIN_TEST_RATIO,
        lrScheduler: aiLrScheduler,
        stepGamma: aiStepGamma,
        stepSize: aiStepSizeEpochs,
        expDecay: aiExpDecay,
        augmentations: augs,
      });
      setStatusText(
        result.cancelled
          ? `Stopped after ${result.epochsRan} epoch(s). Best test loss: ${
              result.bestTestLoss?.toFixed?.(4) ?? 'n/a'
            }`
          : `Done. Best test loss: ${
              result.bestTestLoss?.toFixed?.(4) ?? 'n/a'
            } · acc ${result.bestTestAcc?.toFixed?.(3) ?? 'n/a'}`,
      );
      await refreshModelStatus();
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      Alert.alert('Training failed', msg);
      setStatusText(msg);
    } finally {
      setIsTraining(false);
    }
  }, [
    canTrain,
    aiModel,
    aiBatchSize,
    aiStepSize,
    aiEpochs,
    aiTrainResize,
    aiLrScheduler,
    aiStepGamma,
    aiStepSizeEpochs,
    aiExpDecay,
    augs,
    refreshModelStatus,
  ]);

  return (
    <ScrollView
      style={[styles.container, isDark && styles.bgDark]}
      contentContainerStyle={styles.content}>
      <View style={styles.headerRow}>
        <Text style={[styles.title, isDark && styles.textDark]}>Train AI</Text>
        {onClose && (
          <TouchableOpacity onPress={onClose}>
            <Text style={styles.close}>Close</Text>
          </TouchableOpacity>
        )}
      </View>

      {!available && (
        <Text style={styles.warn}>
          On-device CNN training is currently available on iOS only.
        </Text>
      )}

      <Text style={[styles.section, isDark && styles.textDark]}>Dataset</Text>
      <Text style={[styles.body, isDark && styles.textSecondary]}>
        Kept: {counts.kept} · Marked for deletion: {counts.pendingDelete} ·
        Total: {labeledTotal}
      </Text>
      <Text style={[styles.body, isDark && styles.textSecondary]}>
        Train / test (~80/20): {trainSize} / {testSize}
      </Text>

      <Text style={[styles.section, isDark && styles.textDark]}>Model</Text>
      <View style={styles.chipRow}>
        {TRAINABLE_MODELS.map(opt => (
          <TouchableOpacity
            key={opt.id}
            style={[
              styles.chip,
              isDark && styles.chipDark,
              aiModel === opt.id && styles.chipActive,
            ]}
            disabled={isTraining}
            onPress={() => setAIModel(opt.id as TrainableModelId)}>
            <Text
              style={[
                styles.chipText,
                isDark && styles.textDark,
                aiModel === opt.id && styles.chipTextActive,
              ]}>
              {opt.label}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      <Text style={[styles.label, isDark && styles.textDark]}>
        Training resize
      </Text>
      <View style={styles.chipRow}>
        {TRAIN_RESIZE_OPTIONS.map(v => (
          <TouchableOpacity
            key={v}
            style={[
              styles.chip,
              isDark && styles.chipDark,
              aiTrainResize === v && styles.chipActive,
            ]}
            disabled={isTraining}
            onPress={() => setAITrainResize(v)}>
            <Text
              style={[
                styles.chipText,
                isDark && styles.textDark,
                aiTrainResize === v && styles.chipTextActive,
              ]}>
              {v}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
      {heavyWarning && <Text style={styles.warn}>{heavyWarning}</Text>}

      <Text style={[styles.label, isDark && styles.textDark]}>
        Batch size: {aiBatchSize}
      </Text>
      <Slider
        style={styles.slider}
        minimumValue={1}
        maximumValue={32}
        step={1}
        value={aiBatchSize}
        disabled={isTraining}
        onSlidingComplete={v => setAIBatchSize(v)}
        minimumTrackTintColor="#007AFF"
      />

      <Text style={[styles.label, isDark && styles.textDark]}>
        Base learning rate: {aiStepSize.toFixed(3)}
      </Text>
      <Slider
        style={styles.slider}
        minimumValue={0.001}
        maximumValue={0.1}
        step={0.001}
        value={aiStepSize}
        disabled={isTraining}
        onSlidingComplete={v => setAIStepSize(parseFloat(v.toFixed(3)))}
        minimumTrackTintColor="#007AFF"
      />

      <Text style={[styles.label, isDark && styles.textDark]}>LR scheduler</Text>
      <View style={styles.chipRow}>
        {LR_SCHEDULERS.map(opt => (
          <TouchableOpacity
            key={opt.id}
            style={[
              styles.chip,
              isDark && styles.chipDark,
              aiLrScheduler === opt.id && styles.chipActive,
            ]}
            disabled={isTraining}
            onPress={() => setAILrScheduler(opt.id as LrSchedulerId)}>
            <Text
              style={[
                styles.chipText,
                isDark && styles.textDark,
                aiLrScheduler === opt.id && styles.chipTextActive,
              ]}>
              {opt.label}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
      {aiLrScheduler === 'step' && (
        <>
          <Text style={[styles.label, isDark && styles.textDark]}>
            Step gamma: {aiStepGamma.toFixed(2)}
          </Text>
          <Slider
            style={styles.slider}
            minimumValue={0.1}
            maximumValue={0.9}
            step={0.05}
            value={aiStepGamma}
            disabled={isTraining}
            onSlidingComplete={v => setAIStepGamma(parseFloat(v.toFixed(2)))}
            minimumTrackTintColor="#007AFF"
          />
          <Text style={[styles.label, isDark && styles.textDark]}>
            Step every N epochs: {aiStepSizeEpochs}
          </Text>
          <Slider
            style={styles.slider}
            minimumValue={1}
            maximumValue={20}
            step={1}
            value={aiStepSizeEpochs}
            disabled={isTraining}
            onSlidingComplete={v => setAIStepSizeEpochs(v)}
            minimumTrackTintColor="#007AFF"
          />
        </>
      )}
      {aiLrScheduler === 'exponential' && (
        <>
          <Text style={[styles.label, isDark && styles.textDark]}>
            Decay: {aiExpDecay.toFixed(2)}
          </Text>
          <Slider
            style={styles.slider}
            minimumValue={0.8}
            maximumValue={0.99}
            step={0.01}
            value={aiExpDecay}
            disabled={isTraining}
            onSlidingComplete={v => setAIExpDecay(parseFloat(v.toFixed(2)))}
            minimumTrackTintColor="#007AFF"
          />
        </>
      )}

      <Text style={[styles.label, isDark && styles.textDark]}>
        Max epochs: {aiEpochs}
      </Text>
      <Slider
        style={styles.slider}
        minimumValue={1}
        maximumValue={100}
        step={1}
        value={aiEpochs}
        disabled={isTraining}
        onSlidingComplete={v => setAIEpochs(v)}
        minimumTrackTintColor="#007AFF"
      />

      <Text style={[styles.section, isDark && styles.textDark]}>
        Augmentations
      </Text>
      <AugToggle
        label="Normalize"
        value={augs.normalize}
        disabled={isTraining}
        isDark={isDark}
        onChange={v => setAIAugmentations({normalize: v})}>
        <Text style={[styles.hint, isDark && styles.textSecondary]}>
          Mean [{augs.normalizeMean.map(n => n.toFixed(2)).join(', ')}] · Std [
          {augs.normalizeStd.map(n => n.toFixed(2)).join(', ')}]
        </Text>
      </AugToggle>
      <AugToggle
        label="Random crop"
        value={augs.randomCrop}
        disabled={isTraining}
        isDark={isDark}
        onChange={v => setAIAugmentations({randomCrop: v})}>
        <Text style={[styles.label, isDark && styles.textDark]}>
          Probability: {augs.cropProbability.toFixed(2)}
        </Text>
        <Slider
          style={styles.slider}
          minimumValue={0}
          maximumValue={1}
          step={0.05}
          value={augs.cropProbability}
          disabled={isTraining}
          onSlidingComplete={v =>
            setAIAugmentations({cropProbability: parseFloat(v.toFixed(2))})
          }
          minimumTrackTintColor="#007AFF"
        />
        <Text style={[styles.label, isDark && styles.textDark]}>
          Crop fraction: {augs.cropFraction.toFixed(2)}
        </Text>
        <Slider
          style={styles.slider}
          minimumValue={0.5}
          maximumValue={1}
          step={0.05}
          value={augs.cropFraction}
          disabled={isTraining}
          onSlidingComplete={v =>
            setAIAugmentations({cropFraction: parseFloat(v.toFixed(2))})
          }
          minimumTrackTintColor="#007AFF"
        />
      </AugToggle>
      <AugToggle
        label="Random flip"
        value={augs.randomFlip}
        disabled={isTraining}
        isDark={isDark}
        onChange={v => setAIAugmentations({randomFlip: v})}>
        <Text style={[styles.label, isDark && styles.textDark]}>
          Probability: {augs.flipProbability.toFixed(2)}
        </Text>
        <Slider
          style={styles.slider}
          minimumValue={0}
          maximumValue={1}
          step={0.05}
          value={augs.flipProbability}
          disabled={isTraining}
          onSlidingComplete={v =>
            setAIAugmentations({flipProbability: parseFloat(v.toFixed(2))})
          }
          minimumTrackTintColor="#007AFF"
        />
      </AugToggle>
      <AugToggle
        label="Random rotation"
        value={augs.randomRotation}
        disabled={isTraining}
        isDark={isDark}
        onChange={v => setAIAugmentations({randomRotation: v})}>
        <Text style={[styles.label, isDark && styles.textDark]}>
          Probability: {augs.rotationProbability.toFixed(2)}
        </Text>
        <Slider
          style={styles.slider}
          minimumValue={0}
          maximumValue={1}
          step={0.05}
          value={augs.rotationProbability}
          disabled={isTraining}
          onSlidingComplete={v =>
            setAIAugmentations({
              rotationProbability: parseFloat(v.toFixed(2)),
            })
          }
          minimumTrackTintColor="#007AFF"
        />
        <Text style={[styles.label, isDark && styles.textDark]}>
          Angle ±deg: {augs.rotationDegrees}
        </Text>
        <Slider
          style={styles.slider}
          minimumValue={5}
          maximumValue={45}
          step={1}
          value={augs.rotationDegrees}
          disabled={isTraining}
          onSlidingComplete={v => setAIAugmentations({rotationDegrees: v})}
          minimumTrackTintColor="#007AFF"
        />
      </AugToggle>
      <AugToggle
        label="Gaussian noise"
        value={augs.gaussianNoise}
        disabled={isTraining}
        isDark={isDark}
        onChange={v => setAIAugmentations({gaussianNoise: v})}>
        <Text style={[styles.label, isDark && styles.textDark]}>
          Std (max 0.05): {augs.noiseStd.toFixed(3)}
        </Text>
        <Slider
          style={styles.slider}
          minimumValue={0.005}
          maximumValue={0.05}
          step={0.005}
          value={augs.noiseStd}
          disabled={isTraining}
          onSlidingComplete={v =>
            setAIAugmentations({noiseStd: parseFloat(v.toFixed(3))})
          }
          minimumTrackTintColor="#007AFF"
        />
      </AugToggle>
      <AugToggle
        label="Color jitter"
        value={augs.colorJitter}
        disabled={isTraining}
        isDark={isDark}
        onChange={v => setAIAugmentations({colorJitter: v})}>
        <Text style={[styles.label, isDark && styles.textDark]}>
          Delta: {augs.jitterDelta.toFixed(2)}
        </Text>
        <Slider
          style={styles.slider}
          minimumValue={0.05}
          maximumValue={0.3}
          step={0.01}
          value={augs.jitterDelta}
          disabled={isTraining}
          onSlidingComplete={v =>
            setAIAugmentations({jitterDelta: parseFloat(v.toFixed(2))})
          }
          minimumTrackTintColor="#007AFF"
        />
      </AugToggle>
      <AugToggle
        label="Random grayscale"
        value={augs.randomGrayscale}
        disabled={isTraining}
        isDark={isDark}
        onChange={v => setAIAugmentations({randomGrayscale: v})}>
        <Text style={[styles.label, isDark && styles.textDark]}>
          Probability: {augs.grayscaleProbability.toFixed(2)}
        </Text>
        <Slider
          style={styles.slider}
          minimumValue={0}
          maximumValue={1}
          step={0.05}
          value={augs.grayscaleProbability}
          disabled={isTraining}
          onSlidingComplete={v =>
            setAIAugmentations({
              grayscaleProbability: parseFloat(v.toFixed(2)),
            })
          }
          minimumTrackTintColor="#007AFF"
        />
      </AugToggle>

      <View style={styles.actions}>
        {!isTraining ? (
          <TouchableOpacity
            style={[styles.primaryBtn, !canTrain && styles.btnDisabled]}
            disabled={!canTrain}
            onPress={handleTrain}>
            <Text style={styles.primaryBtnText}>Train</Text>
          </TouchableOpacity>
        ) : (
          <TouchableOpacity
            style={styles.stopBtn}
            onPress={async () => {
              await stopTraining();
              setStatusText('Stop requested...');
            }}>
            <Text style={styles.primaryBtnText}>Stop</Text>
          </TouchableOpacity>
        )}
      </View>

      {isTraining && (
        <View style={styles.statusRow}>
          <ActivityIndicator color="#007AFF" />
          <Text style={[styles.body, isDark && styles.textDark]}>
            Epoch {progress?.epoch ?? 0}/{aiEpochs}
            {progress
              ? ` · lr ${progress.learningRate?.toFixed?.(4) ?? '?'} · loss ${progress.trainLoss.toFixed(3)}/${progress.testLoss.toFixed(3)} · acc ${(progress.trainAcc * 100).toFixed(0)}%/${(progress.testAcc * 100).toFixed(0)}%`
              : ''}
          </Text>
        </View>
      )}

      <Text style={[styles.section, isDark && styles.textDark]}>Loss</Text>
      <MetricChart
        points={history}
        isDark={isDark}
        mode="loss"
        emptyHint="Loss graph appears when training starts"
      />

      <Text style={[styles.section, isDark && styles.textDark]}>Accuracy</Text>
      <MetricChart
        points={history}
        isDark={isDark}
        mode="acc"
        emptyHint="Accuracy graph appears when training starts"
      />

      {!!statusText && (
        <Text style={[styles.body, isDark && styles.textSecondary]}>
          {statusText}
        </Text>
      )}

      <Text style={[styles.section, isDark && styles.textDark]}>
        Active model
      </Text>
      <Text style={[styles.body, isDark && styles.textSecondary]}>
        {hasModel && activeModelInfo
          ? `${activeModelInfo.modelId} @ ${activeModelInfo.trainResize}px · loss ${Number(
              activeModelInfo.bestTestLoss,
            ).toFixed(4)} · acc ${Number(
              activeModelInfo.bestTestAcc ?? 0,
            ).toFixed(3)}`
          : 'No trained model yet.'}
      </Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {flex: 1, backgroundColor: '#fff'},
  bgDark: {backgroundColor: '#000'},
  content: {padding: 16, paddingBottom: 48},
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  title: {fontSize: 22, fontWeight: '700', color: '#111'},
  close: {fontSize: 16, color: '#007AFF', fontWeight: '600'},
  section: {
    fontSize: 17,
    fontWeight: '700',
    color: '#333',
    marginTop: 20,
    marginBottom: 8,
  },
  label: {fontSize: 14, color: '#333', marginTop: 12},
  body: {fontSize: 14, color: '#555', marginBottom: 4},
  textDark: {color: '#fff'},
  textSecondary: {color: '#8e8e93'},
  hint: {fontSize: 12, color: '#8e8e93', marginTop: 4},
  warn: {fontSize: 13, color: '#ff9500', marginTop: 6},
  chipRow: {flexDirection: 'row', flexWrap: 'wrap', gap: 8},
  chip: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
    backgroundColor: '#f0f0f0',
  },
  chipDark: {backgroundColor: '#2c2c2e'},
  chipActive: {backgroundColor: '#007AFF'},
  chipText: {fontSize: 13, color: '#333'},
  chipTextActive: {color: '#fff', fontWeight: '600'},
  slider: {height: 32, marginVertical: 4},
  augBlock: {marginTop: 8},
  augRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  actions: {marginTop: 20},
  primaryBtn: {
    backgroundColor: '#007AFF',
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: 'center',
  },
  stopBtn: {
    backgroundColor: '#ff3b30',
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: 'center',
  },
  btnDisabled: {opacity: 0.4},
  primaryBtnText: {color: '#fff', fontSize: 16, fontWeight: '700'},
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    marginTop: 12,
  },
  chart: {
    marginTop: 8,
    borderRadius: 8,
    backgroundColor: '#f5f5f7',
    overflow: 'hidden',
  },
  chartDark: {backgroundColor: '#1c1c1e'},
  chartEmpty: {
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f5f5f7',
    borderRadius: 8,
  },
  chartLegend: {
    position: 'absolute',
    bottom: 4,
    right: 8,
    fontSize: 11,
    color: '#8e8e93',
  },
});
