import React, {useCallback, useEffect, useState} from 'react';
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
  deleteModel,
  getModelStatus,
  isTrainerAvailable,
  startTraining,
  stopTraining,
  subscribeTrainingProgress,
} from '../services/cnnTrainer';
import {
  isMlpModel,
  LR_SCHEDULERS,
  LrSchedulerId,
  MIN_LABELED_FOR_TRAIN,
  ModelStatus,
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

export function TrainAIScreen({onClose}: TrainAIScreenProps) {
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';

  const {
    aiModel,
    setAIModel,
    aiBatchSize,
    aiStepSize,
    aiLrScheduler,
    aiStepGamma,
    aiStepSizeEpochs,
    aiExpDecay,
    setAIBatchSize,
    setAIStepSize,
    setAILrScheduler,
    setAIStepGamma,
    setAIStepSizeEpochs,
    setAIExpDecay,
  } = useSettingsStore();

  const {refreshModelStatus, activeModelInfo, hasModel} = useAIStore();

  const [available, setAvailable] = useState(Platform.OS === 'ios');
  const [isTraining, setIsTraining] = useState(false);
  const [counts, setCounts] = useState({kept: 0, pendingDelete: 0});
  const [progress, setProgress] = useState<TrainingProgress | null>(null);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [statusText, setStatusText] = useState('');
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);

  const mlp = isMlpModel(aiModel);
  const labeledTotal = counts.kept + counts.pendingDelete;
  const canTrain =
    available &&
    !isTraining &&
    labeledTotal >= MIN_LABELED_FOR_TRAIN &&
    counts.kept > 0 &&
    counts.pendingDelete > 0;

  const reloadStatus = useCallback(() => {
    getModelStatus(aiModel).then(setModelStatus);
  }, [aiModel]);

  useEffect(() => {
    setCounts(getLabeledCounts());
    refreshModelStatus();
    isTrainerAvailable().then(setAvailable);
    reloadStatus();
  }, [refreshModelStatus, reloadStatus]);

  useEffect(() => {
    return subscribeTrainingProgress(p => {
      setProgress(p);
      if (p.phase === 'features' && p.index != null && p.total != null) {
        setStatusText(`Extracting features ${p.index}/${p.total}…`);
      }
      if (p.phase === 'fit' && p.done === false) {
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
      }
    });
  }, []);

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
        scheduler: aiLrScheduler,
        stepGamma: aiStepGamma,
        stepEvery: aiStepSizeEpochs,
        expDecay: aiExpDecay,
      });
      const acc =
        result.accuracy != null
          ? `${(result.accuracy * 100).toFixed(0)}%`
          : result.bestTestAcc != null
            ? `${(result.bestTestAcc * 100).toFixed(0)}%`
            : 'n/a';
      const f1 = result.f1 != null ? result.f1.toFixed(2) : 'n/a';
      setStatusText(
        result.cancelled
          ? `Stopped. Accuracy ${acc} · F1 ${f1}`
          : `Done. Accuracy ${acc} · F1 ${f1} on ${result.trainSize} new photos`,
      );
      await refreshModelStatus();
      reloadStatus();
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
    aiLrScheduler,
    aiStepGamma,
    aiStepSizeEpochs,
    aiExpDecay,
    refreshModelStatus,
    reloadStatus,
  ]);

  const explainTraining = useCallback(() => {
    Alert.alert(
      'Incremental training',
      'Train adds only photos this model has not seen. The same labeled photos can train a different model, and their feature vectors stay cached. To fit this model again from scratch, delete it at the bottom of the screen.',
    );
  }, []);

  const confirmDelete = useCallback(() => {
    const label = TRAINABLE_MODELS.find(m => m.id === aiModel)?.label ?? aiModel;
    Alert.alert(
      `Delete ${label}?`,
      'This clears its weights and which photos it has learned. The other models stay as they are.',
      [
        {text: 'Cancel', style: 'cancel'},
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            await deleteModel(aiModel);
            setHistory([]);
            setStatusText(`${label} deleted. The next train starts from scratch.`);
            await refreshModelStatus();
            reloadStatus();
          },
        },
      ],
    );
  }, [aiModel, refreshModelStatus, reloadStatus]);

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
          On-device feature training is available on iOS only.
        </Text>
      )}

      <Text style={[styles.section, isDark && styles.textDark]}>Dataset</Text>
      <Text style={[styles.body, isDark && styles.textSecondary]}>
        Kept: {counts.kept} · Marked for deletion: {counts.pendingDelete} ·
        Total: {labeledTotal}
      </Text>
      <Text style={[styles.body, isDark && styles.textSecondary]}>
        Train updates only the selected model. Photos it has already learned are skipped.
      </Text>

      <View style={styles.headerRow}>
        <Text style={[styles.section, isDark && styles.textDark, styles.sectionInline]}>
          Model
        </Text>
        <TouchableOpacity onPress={explainTraining} hitSlop={8}>
          <Text style={styles.info}>ⓘ</Text>
        </TouchableOpacity>
      </View>
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

      {mlp && (
        <>
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
            Learning rate: {aiStepSize.toFixed(4)}
          </Text>
          <Slider
            style={styles.slider}
            minimumValue={0.0001}
            maximumValue={0.05}
            step={0.0001}
            value={aiStepSize}
            disabled={isTraining}
            onSlidingComplete={v => setAIStepSize(parseFloat(v.toFixed(4)))}
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
        </>
      )}
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
            {progress?.phase === 'features' && progress.index != null
              ? `Extracting features ${progress.index}/${progress.total ?? '?'}`
              : progress
                ? `Fitting · loss ${Number(progress.trainLoss).toFixed(3)} · acc ${(Number(progress.trainAcc) * 100).toFixed(0)}%`
                : 'Starting…'}
          </Text>
        </View>
      )}

      {mlp && (
        <>
          <Text style={[styles.section, isDark && styles.textDark]}>Loss</Text>
          <MetricChart
            points={history}
            isDark={isDark}
            mode="loss"
            emptyHint="Loss graph appears while the MLP trains"
          />
          <Text style={[styles.section, isDark && styles.textDark]}>Accuracy</Text>
          <MetricChart
            points={history}
            isDark={isDark}
            mode="acc"
            emptyHint="Accuracy graph appears while the MLP trains"
          />
        </>
      )}

      <Text style={[styles.section, isDark && styles.textDark]}>Last update</Text>
      {modelStatus?.hasModel ? (
        <View style={[styles.table, isDark && styles.chartDark]}>
          <View style={styles.tableRow}>
            <Text style={[styles.tableCell, isDark && styles.textSecondary]}>Accuracy</Text>
            <Text style={[styles.tableValue, isDark && styles.textDark]}>
              {(modelStatus.accuracy * 100).toFixed(0)}%
            </Text>
          </View>
          <View style={styles.tableRow}>
            <Text style={[styles.tableCell, isDark && styles.textSecondary]}>F1</Text>
            <Text style={[styles.tableValue, isDark && styles.textDark]}>
              {modelStatus.f1.toFixed(2)}
            </Text>
          </View>
          <View style={styles.tableRow}>
            <Text style={[styles.tableCell, isDark && styles.textSecondary]}>
              Photos in last update
            </Text>
            <Text style={[styles.tableValue, isDark && styles.textDark]}>
              {modelStatus.trainSize}
            </Text>
          </View>
          <View style={styles.tableRow}>
            <Text style={[styles.tableCell, isDark && styles.textSecondary]}>
              Photos learned
            </Text>
            <Text style={[styles.tableValue, isDark && styles.textDark]}>
              {modelStatus.learnedCount}
            </Text>
          </View>
        </View>
      ) : (
        <Text style={[styles.hint, isDark && styles.textSecondary]}>
          Accuracy and F1 appear after this model trains.
        </Text>
      )}

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
          ? `${TRAINABLE_MODELS.find(m => m.id === activeModelInfo.modelId)?.label ?? activeModelInfo.modelId}`
          : 'No trained model yet.'}
      </Text>

      <TouchableOpacity
        style={[styles.deleteBtn, (!modelStatus?.hasModel || isTraining) && styles.btnDisabled]}
        disabled={!modelStatus?.hasModel || isTraining}
        onPress={confirmDelete}>
        <Text style={styles.deleteBtnText}>
          Delete {TRAINABLE_MODELS.find(m => m.id === aiModel)?.label ?? 'model'}
        </Text>
      </TouchableOpacity>
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
  sectionInline: {marginTop: 12, marginBottom: 0},
  info: {fontSize: 18, color: '#007AFF', marginTop: 12},
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
  deleteBtn: {
    marginTop: 28,
    paddingVertical: 12,
    borderRadius: 10,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#ff3b30',
  },
  deleteBtnText: {color: '#ff3b30', fontSize: 15, fontWeight: '600'},
  table: {borderRadius: 8, backgroundColor: '#f5f5f7', paddingHorizontal: 12},
  tableRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 10,
  },
  tableCell: {fontSize: 14, color: '#555'},
  tableValue: {fontSize: 14, fontWeight: '600', color: '#111'},
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
