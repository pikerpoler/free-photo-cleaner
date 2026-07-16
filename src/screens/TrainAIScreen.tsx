import React, {useCallback, useEffect, useMemo, useState} from 'react';
import {
  ActivityIndicator,
  Alert,
  Platform,
  ScrollView,
  StyleSheet,
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
  MAX_TRAIN_RESIZE,
  MIN_LABELED_FOR_TRAIN,
  MIN_TRAIN_RESIZE,
  TRAIN_TEST_RATIO,
  TRAINABLE_MODELS,
  TrainableModelId,
  TrainingProgress,
} from '../ai/cnnTypes';

interface TrainAIScreenProps {
  onClose?: () => void;
}

function LossChart({
  points,
  isDark,
}: {
  points: {epoch: number; trainLoss: number; testLoss: number}[];
  isDark: boolean;
}) {
  const width = 320;
  const height = 140;
  const pad = 8;

  if (points.length === 0) {
    return (
      <View style={[styles.chartEmpty, {height}]}>
        <Text style={[styles.hint, isDark && styles.textSecondary]}>
          Loss graph appears when training starts
        </Text>
      </View>
    );
  }

  const maxLoss = Math.max(
    0.01,
    ...points.flatMap(p => [p.trainLoss, p.testLoss]),
  );

  const toX = (i: number) =>
    pad + (points.length === 1 ? width / 2 : (i / (points.length - 1)) * (width - pad * 2));
  const toY = (v: number) =>
    height - pad - (v / maxLoss) * (height - pad * 2);

  // Simple polyline via absolute Views (no SVG dep)
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
      {renderDots(
        'train',
        points.map(p => p.trainLoss),
        '#007AFF',
      )}
      {renderDots(
        'test',
        points.map(p => p.testLoss),
        '#FF9500',
      )}
      <Text style={styles.chartLegend}>
        <Text style={{color: '#007AFF'}}>Train</Text>
        {'  '}
        <Text style={{color: '#FF9500'}}>Test</Text>
        {'  '}
        max {maxLoss.toFixed(3)}
      </Text>
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
    setAIModel,
    setAIBatchSize,
    setAIStepSize,
    setAIEpochs,
    setAITrainResize,
  } = useSettingsStore();

  const {refreshModelStatus, activeModelInfo, hasModel} = useAIStore();

  const [available, setAvailable] = useState(Platform.OS === 'ios');
  const [isTraining, setIsTraining] = useState(false);
  const [counts, setCounts] = useState({kept: 0, pendingDelete: 0});
  const [progress, setProgress] = useState<TrainingProgress | null>(null);
  const [history, setHistory] = useState<
    {epoch: number; trainLoss: number; testLoss: number}[]
  >([]);
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

  const refreshCounts = useCallback(() => {
    setCounts(getLabeledCounts());
  }, []);

  useEffect(() => {
    refreshCounts();
    refreshModelStatus();
    isTrainerAvailable().then(setAvailable);
  }, [refreshCounts, refreshModelStatus]);

  useEffect(() => {
    const unsub = subscribeTrainingProgress(p => {
      setProgress(p);
      setHistory(prev => {
        const next = prev.filter(h => h.epoch !== p.epoch);
        next.push({
          epoch: p.epoch,
          trainLoss: p.trainLoss,
          testLoss: p.testLoss,
        });
        next.sort((a, b) => a.epoch - b.epoch);
        return next;
      });
    });
    return unsub;
  }, []);

  const heavyWarning = useMemo(() => {
    if (aiTrainResize >= 256 || aiModel === 'resnet-18') {
      return 'High resize and/or ResNet-18 may be slow and memory-heavy on weaker devices.';
    }
    return null;
  }, [aiTrainResize, aiModel]);

  const handleTrain = useCallback(async () => {
    if (!canTrain) return;
    const samples = getLabeledPhotoSamples();
    if (samples.length < MIN_LABELED_FOR_TRAIN) {
      Alert.alert(
        'Not enough labels',
        `Need at least ${MIN_LABELED_FOR_TRAIN} labeled photos (keep + delete).`,
      );
      return;
    }
    const hasKeep = samples.some(s => s.label === 0);
    const hasDelete = samples.some(s => s.label === 1);
    if (!hasKeep || !hasDelete) {
      Alert.alert(
        'Need both classes',
        'Train on both kept and marked-for-deletion photos.',
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
      });
      setStatusText(
        result.cancelled
          ? `Stopped after ${result.epochsRan} epoch(s). Best test loss: ${
              result.bestTestLoss?.toFixed?.(4) ?? 'n/a'
            }`
          : `Done. Best test loss: ${
              result.bestTestLoss?.toFixed?.(4) ?? 'n/a'
            }`,
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
    refreshModelStatus,
  ]);

  const handleStop = useCallback(async () => {
    await stopTraining();
    setStatusText('Stop requested...');
  }, []);

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
        Total labeled: {labeledTotal}
      </Text>
      <Text style={[styles.body, isDark && styles.textSecondary]}>
        Train / test split (~80/20): {trainSize} / {testSize}
      </Text>
      {labeledTotal < MIN_LABELED_FOR_TRAIN && (
        <Text style={styles.warn}>
          Need at least {MIN_LABELED_FOR_TRAIN} labeled photos with both keep and
          delete examples.
        </Text>
      )}

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
        Training resize: {aiTrainResize}px
      </Text>
      <Slider
        style={styles.slider}
        minimumValue={MIN_TRAIN_RESIZE}
        maximumValue={MAX_TRAIN_RESIZE}
        step={16}
        value={aiTrainResize}
        disabled={isTraining}
        onSlidingComplete={v => setAITrainResize(v)}
        minimumTrackTintColor="#007AFF"
      />
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
        Learning rate: {aiStepSize.toFixed(3)}
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

      <View style={styles.actions}>
        {!isTraining ? (
          <TouchableOpacity
            style={[styles.primaryBtn, !canTrain && styles.btnDisabled]}
            disabled={!canTrain}
            onPress={handleTrain}>
            <Text style={styles.primaryBtnText}>Train</Text>
          </TouchableOpacity>
        ) : (
          <TouchableOpacity style={styles.stopBtn} onPress={handleStop}>
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
              ? ` · train ${progress.trainLoss.toFixed(4)} · test ${progress.testLoss.toFixed(4)}`
              : ''}
          </Text>
        </View>
      )}

      <Text style={[styles.section, isDark && styles.textDark]}>Loss</Text>
      <LossChart points={history} isDark={isDark} />

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
          ? `${activeModelInfo.modelId} @ ${activeModelInfo.trainResize}px · best test loss ${Number(
              activeModelInfo.bestTestLoss,
            ).toFixed(4)}`
          : 'No trained model yet. AI sort uses neutral scores until you train.'}
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
  hint: {fontSize: 13, color: '#8e8e93'},
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
