import React, {useCallback, useEffect, useRef, useState} from 'react';
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
import {useFocusEffect} from '@react-navigation/native';
import {useShallow} from 'zustand/react/shallow';
import {useSettingsStore} from '../stores/settingsStore';
import {useAIStore} from '../stores/aiStore';
import {useQueueStore} from '../stores/queueStore';
import {
  getModelStatus,
  getYOLOStatus,
  isTrainerAvailable,
  startTraining,
  stopTraining,
  subscribeTrainingProgress,
} from '../services/cnnTrainer';
import {backfillTrainingDataset} from '../services/trainingCapture';
import {
  clearTrainingDataset,
  getTrainingDatasetStats,
  getTrainingExamples,
  TrainingDatasetStats,
} from '../services/trainingDataset';
import {
  FEATURE_SCHEMA_VERSION,
  isMlpModel,
  LR_SCHEDULERS,
  LrSchedulerId,
  MIN_LABELED_FOR_TRAIN,
  ModelStatus,
  TRAINABLE_MODELS,
  TrainableModelId,
  TrainingProgress,
  YOLODetectorStatus,
} from '../ai/cnnTypes';
import {
  formatMlpLearningRate,
  mlpBatchSizeToSlider,
  mlpLearningRateToSlider,
  sliderToMlpBatchSize,
  sliderToMlpLearningRate,
} from '../ai/trainSliderScale';
import {getSetting, setSetting, aiTrainingHistoryKey} from '../services/storage';

type HistoryPoint = {
  epoch: number;
  trainLoss: number;
  testLoss: number;
  trainAcc: number;
  testAcc: number;
};

function formatDatasetBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

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
  const maxV = mode === 'acc' ? 1 : Math.max(0.01, ...trainVals, ...testVals);

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

function persistMlpHistory(modelId: TrainableModelId, points: HistoryPoint[]) {
  if (!isMlpModel(modelId)) return;
  setSetting(aiTrainingHistoryKey(modelId), points);
}

function loadMlpHistory(modelId: TrainableModelId): HistoryPoint[] {
  if (!isMlpModel(modelId)) return [];
  return getSetting<HistoryPoint[]>(aiTrainingHistoryKey(modelId), []);
}

function clearMlpHistory(modelId: TrainableModelId) {
  if (!isMlpModel(modelId)) return;
  setSetting(aiTrainingHistoryKey(modelId), []);
}

export function TrainAIScreen() {
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';

  const {
    aiModel,
    setAIModel,
    aiBatchSize,
    aiStepSize,
    aiEpochs,
    aiLrScheduler,
    aiStepGamma,
    aiStepSizeEpochs,
    aiExpDecay,
    aiLogisticC,
    aiLogisticMaxIter,
    aiCatboostLearningRate,
    aiCatboostDepth,
    aiCatboostMinLeaf,
    aiCatboostMaxNewTrees,
    setAIBatchSize,
    setAIStepSize,
    setAIEpochs,
    setAILrScheduler,
    setAIStepGamma,
    setAIStepSizeEpochs,
    setAIExpDecay,
    setAILogisticC,
    setAILogisticMaxIter,
    setAICatboostLearningRate,
    setAICatboostDepth,
    setAICatboostMinLeaf,
    setAICatboostMaxNewTrees,
    setAISortModel,
  } = useSettingsStore(
    useShallow(s => ({
      aiModel: s.aiModel,
      setAIModel: s.setAIModel,
      aiBatchSize: s.aiBatchSize,
      aiStepSize: s.aiStepSize,
      aiEpochs: s.aiEpochs,
      aiLrScheduler: s.aiLrScheduler,
      aiStepGamma: s.aiStepGamma,
      aiStepSizeEpochs: s.aiStepSizeEpochs,
      aiExpDecay: s.aiExpDecay,
      aiLogisticC: s.aiLogisticC,
      aiLogisticMaxIter: s.aiLogisticMaxIter,
      aiCatboostLearningRate: s.aiCatboostLearningRate,
      aiCatboostDepth: s.aiCatboostDepth,
      aiCatboostMinLeaf: s.aiCatboostMinLeaf,
      aiCatboostMaxNewTrees: s.aiCatboostMaxNewTrees,
      setAIBatchSize: s.setAIBatchSize,
      setAIStepSize: s.setAIStepSize,
      setAIEpochs: s.setAIEpochs,
      setAILrScheduler: s.setAILrScheduler,
      setAIStepGamma: s.setAIStepGamma,
      setAIStepSizeEpochs: s.setAIStepSizeEpochs,
      setAIExpDecay: s.setAIExpDecay,
      setAILogisticC: s.setAILogisticC,
      setAILogisticMaxIter: s.setAILogisticMaxIter,
      setAICatboostLearningRate: s.setAICatboostLearningRate,
      setAICatboostDepth: s.setAICatboostDepth,
      setAICatboostMinLeaf: s.setAICatboostMinLeaf,
      setAICatboostMaxNewTrees: s.setAICatboostMaxNewTrees,
      setAISortModel: s.setAISortModel,
    })),
  );

  const {
    refreshModelStatus,
    activeModelInfo,
    hasModel,
    selectSortModel,
  } = useAIStore(
    useShallow(s => ({
      refreshModelStatus: s.refreshModelStatus,
      activeModelInfo: s.activeModelInfo,
      hasModel: s.hasModel,
      selectSortModel: s.selectSortModel,
    })),
  );

  const [available, setAvailable] = useState(Platform.OS === 'ios');
  const [isTraining, setIsTraining] = useState(false);
  const [counts, setCounts] = useState<TrainingDatasetStats>({
    kept: 0,
    pendingDelete: 0,
    total: 0,
    bytes: 0,
  });
  const [progress, setProgress] = useState<TrainingProgress | null>(null);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [statusText, setStatusText] = useState('');
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [yoloStatus, setYoloStatus] = useState<YOLODetectorStatus | null>(null);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [selectedModel, setSelectedModel] = useState<TrainableModelId>(aiModel);
  const [trainBlocked, setTrainBlocked] = useState(false);

  const selectedModelRef = useRef(selectedModel);
  selectedModelRef.current = selectedModel;

  const [mlpLrSliderT, setMlpLrSliderT] = useState(() =>
    mlpLearningRateToSlider(aiStepSize),
  );
  const [mlpBatchSliderT, setMlpBatchSliderT] = useState(() =>
    mlpBatchSizeToSlider(aiBatchSize),
  );

  const mlp = isMlpModel(selectedModel);
  const labeledTotal = counts.total;
  const meetsRules =
    labeledTotal >= MIN_LABELED_FOR_TRAIN &&
    counts.kept >= 2 &&
    counts.pendingDelete >= 2;

  const canTrain = available && !isTraining && !trainBlocked && meetsRules;

  const yoloDegraded =
    available &&
    Platform.OS === 'ios' &&
    yoloStatus != null &&
    yoloStatus.status !== 'loaded';

  const yoloRunWarn =
    available &&
    Platform.OS === 'ios' &&
    yoloStatus?.status === 'loaded' &&
    (yoloStatus.lastRunError?.length ?? 0) > 0;

  const needsRetrainAfterSchemaBump =
    hasModel &&
    activeModelInfo != null &&
    (activeModelInfo.featureSchemaVersion == null ||
      activeModelInfo.featureSchemaVersion < FEATURE_SCHEMA_VERSION);

  const blockMessage =
    !available || isTraining || canTrain
      ? ''
      : 'Training needs at least 20 saved photos, including at least 2 kept and 2 marked for deletion.';

  const readyMessage = canTrain
    ? `${labeledTotal} saved photo${labeledTotal === 1 ? '' : 's'} ready to train.`
    : '';

  const reloadStatus = useCallback(() => {
    getModelStatus(selectedModel).then(setModelStatus);
  }, [selectedModel]);

  const reloadYoloStatus = useCallback(() => {
    getYOLOStatus().then(setYoloStatus);
  }, []);

  const reloadCounts = useCallback(() => {
    setCounts(getTrainingDatasetStats());
  }, []);

  useEffect(() => {
    setTrainBlocked(false);
  }, [selectedModel, counts.kept, counts.pendingDelete]);

  useEffect(() => {
    reloadCounts();
    isTrainerAvailable().then(avail => {
      setAvailable(avail);
      if (avail) {
        reloadYoloStatus();
      }
    });
  }, [reloadCounts, reloadYoloStatus]);

  useFocusEffect(
    useCallback(() => {
      setSelectedModel(useSettingsStore.getState().aiModel);
      refreshModelStatus().catch(() => {});
      reloadCounts();
      reloadStatus();
      reloadYoloStatus();
      backfillTrainingDataset()
        .then(() => reloadCounts())
        .catch(() => {});
      return () => {
        const target = selectedModelRef.current;
        const state = useAIStore.getState();
        const activeId = state.activeModelInfo?.modelId;
        setAISortModel(target);
        setAIModel(target);
        if (activeId === target) return;
        (async () => {
          try {
            await selectSortModel(target);
          } catch (e: unknown) {
            Alert.alert(
              'Model sync failed',
              e instanceof Error ? e.message : String(e),
            );
          }
        })().catch(() => {});
      };
    }, [
      refreshModelStatus,
      reloadStatus,
      reloadCounts,
      reloadYoloStatus,
      selectSortModel,
      setAISortModel,
      setAIModel,
    ]),
  );

  useEffect(() => {
    reloadStatus();
  }, [selectedModel, reloadStatus]);

  useEffect(() => {
    if (mlp) {
      setHistory(loadMlpHistory(selectedModel));
    } else {
      setHistory([]);
    }
  }, [selectedModel, mlp]);

  useEffect(() => {
    setMlpLrSliderT(mlpLearningRateToSlider(aiStepSize));
  }, [aiStepSize]);

  useEffect(() => {
    setMlpBatchSliderT(mlpBatchSizeToSlider(aiBatchSize));
  }, [aiBatchSize]);

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
          if (isMlpModel(selectedModelRef.current)) {
            persistMlpHistory(selectedModelRef.current, next);
          }
          return next;
        });
      }
    });
  }, []);

  const explainDataset = useCallback(() => {
    Alert.alert(
      'Train locally',
      'The images are passed through a feature extractor and fed into AI models to be trained locally. Nothing is uploaded. Older devices may experience slower training.\n\nEach train rebuilds the selected model on every saved swipe: 80% to learn, 20% to score accuracy and F1. The current model stays until that run finishes. Feature vectors are kept even after photos are deleted.',
    );
  }, []);

  const handleTrain = useCallback(async () => {
    if (!canTrain) return;
    setIsTraining(true);
    const trainModel = selectedModel;
    if (isMlpModel(trainModel)) {
      clearMlpHistory(trainModel);
      setHistory([]);
    }
    setProgress(null);
    setStatusText('Saving features...');

    try {
      await backfillTrainingDataset();
      reloadCounts();
      const examples = getTrainingExamples();
      setStatusText('Starting training...');
      const result = await startTraining({
        features: examples.map(s => s.features),
        labels: examples.map(s => s.label),
        modelId: trainModel,
        batchSize: aiBatchSize,
        learningRate: aiStepSize,
        maxEpochs: aiEpochs,
        scheduler: aiLrScheduler,
        stepGamma: aiStepGamma,
        stepEvery: aiStepSizeEpochs,
        expDecay: aiExpDecay,
        logisticC: aiLogisticC,
        logisticMaxIter: aiLogisticMaxIter,
        catboostLearningRate: aiCatboostLearningRate,
        catboostDepth: aiCatboostDepth,
        catboostMinLeaf: aiCatboostMinLeaf,
        catboostMaxNewTrees: aiCatboostMaxNewTrees,
      });
      const acc =
        result.accuracy != null
          ? `${(result.accuracy * 100).toFixed(0)}%`
          : result.bestTestAcc != null
            ? `${(result.bestTestAcc * 100).toFixed(0)}%`
            : 'n/a';
      const f1 = result.f1 != null ? result.f1.toFixed(2) : 'n/a';
      const skipped = result.skippedUnreadable ?? 0;
      const skipNote =
        skipped === 1
          ? ' 1 labeled photo could not be read and was skipped.'
          : skipped > 1
            ? ` ${skipped} labeled photos could not be read and were skipped.`
            : '';
      const trainedNote = `Trained on ${result.trainSize}, tested on ${result.testSize}.${skipNote}`;
      setStatusText(
        result.cancelled
          ? 'Stopped. The current model was left unchanged.'
          : `Done. Test accuracy ${acc} · F1 ${f1}. ${trainedNote}`,
      );
      await refreshModelStatus();
      reloadStatus();
      reloadCounts();
    } catch (e: unknown) {
      const err = e as {code?: string; message?: string};
      const code = err.code ?? '';
      const msg = e instanceof Error ? e.message : String(e);
      const blocked =
        code === 'NO_NEW' ||
        code === 'TOO_FEW' ||
        msg.includes('already includes every labeled') ||
        msg.includes('Training needs at least') ||
        msg.includes('Each training session needs') ||
        msg.includes('Need at least');
      if (blocked) {
        setTrainBlocked(true);
        setStatusText('');
      } else {
        Alert.alert('Training failed', msg);
        setStatusText(msg);
      }
      await refreshModelStatus();
      reloadStatus();
      reloadCounts();
    } finally {
      setIsTraining(false);
    }
  }, [
    canTrain,
    selectedModel,
    aiBatchSize,
    aiStepSize,
    aiEpochs,
    aiLrScheduler,
    aiStepGamma,
    aiStepSizeEpochs,
    aiExpDecay,
    aiLogisticC,
    aiLogisticMaxIter,
    aiCatboostLearningRate,
    aiCatboostDepth,
    aiCatboostMinLeaf,
    aiCatboostMaxNewTrees,
    refreshModelStatus,
    reloadStatus,
    reloadCounts,
  ]);

  const confirmDeleteDataset = useCallback(() => {
    Alert.alert(
      'Delete dataset?',
      'This cannot be undone. Are you sure?',
      [
        {text: 'Cancel', style: 'cancel'},
        {
          text: 'Delete',
          style: 'destructive',
          onPress: () => {
            clearTrainingDataset();
            setStatusText('Dataset deleted. Swipe more photos to build a new one.');
            reloadCounts();
          },
        },
      ],
    );
  }, [reloadCounts]);

  const onSelectModel = (id: TrainableModelId) => {
    if (isTraining) return;
    setSelectedModel(id);
    setAIModel(id);
  };

  return (
    <ScrollView
      style={[styles.container, isDark && styles.bgDark]}
      contentContainerStyle={styles.content}>
      {!available && (
        <Text style={styles.warn}>
          On-device feature training is available on iOS only.
        </Text>
      )}

      {needsRetrainAfterSchemaBump && (
        <Text style={styles.warn}>
          Feature extraction was upgraded (v{FEATURE_SCHEMA_VERSION}). Retrain your
          model so scores use the new object-detection inputs. Old checkpoints used a
          previous layout.
        </Text>
      )}

      {yoloDegraded && (
        <Text style={styles.warn}>
          Object detection unavailable — AI runs without YOLO; scores may be less
          accurate.
          {yoloStatus?.loadError
            ? ` (${yoloStatus.loadError})`
            : ''}
        </Text>
      )}

      {yoloRunWarn && (
        <Text style={styles.warn}>
          YOLO model loaded but inference reported an error:{' '}
          {yoloStatus?.lastRunError}
        </Text>
      )}

      <View style={styles.headerRow}>
        <Text style={[styles.section, isDark && styles.textDark]}>Dataset</Text>
        <TouchableOpacity onPress={explainDataset} hitSlop={8}>
          <Text style={styles.info}>ⓘ</Text>
        </TouchableOpacity>
      </View>
      <Text style={[styles.body, isDark && styles.textSecondary]}>
        Kept: {counts.kept} · Marked for deletion: {counts.pendingDelete} ·
        Total: {labeledTotal}
      </Text>
      <Text style={[styles.body, isDark && styles.textSecondary]}>
        Train rebuilds the selected model on every saved swipe. It learns 80% of
        the dataset and scores the other 20%. The current model stays until the
        run finishes.
      </Text>

      <Text style={[styles.section, isDark && styles.textDark, styles.sectionSpaced]}>
        Model
      </Text>
      <View style={styles.chipRow}>
        {TRAINABLE_MODELS.map(opt => (
          <TouchableOpacity
            key={opt.id}
            style={[
              styles.chip,
              isDark && styles.chipDark,
              selectedModel === opt.id && styles.chipActive,
            ]}
            disabled={isTraining}
            onPress={() => onSelectModel(opt.id as TrainableModelId)}>
            <Text
              style={[
                styles.chipText,
                isDark && styles.textDark,
                selectedModel === opt.id && styles.chipTextActive,
              ]}>
              {opt.label}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      <Text style={[styles.section, isDark && styles.textDark]}>Last update</Text>
      {modelStatus?.hasModel ? (
        <View style={[styles.table, isDark && styles.chartDark]}>
          <View style={styles.tableRow}>
            <Text style={[styles.tableCell, isDark && styles.textSecondary]}>
              {(modelStatus.testSize ?? 0) > 0 ? 'Test accuracy' : 'Accuracy'}
            </Text>
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
              {(modelStatus.testSize ?? 0) > 0 ? 'Trained' : 'Photos in last update'}
            </Text>
            <Text style={[styles.tableValue, isDark && styles.textDark]}>
              {modelStatus.trainSize}
            </Text>
          </View>
          {(modelStatus.testSize ?? 0) > 0 && (
            <View style={styles.tableRow}>
              <Text style={[styles.tableCell, isDark && styles.textSecondary]}>Tested</Text>
              <Text style={[styles.tableValue, isDark && styles.textDark]}>
                {modelStatus.testSize}
              </Text>
            </View>
          )}
        </View>
      ) : (
        <Text style={[styles.hint, isDark && styles.textSecondary]}>
          Accuracy and F1 appear after this model trains.
        </Text>
      )}

      {!!blockMessage && <Text style={styles.prereq}>{blockMessage}</Text>}
      {!!readyMessage && (
        <Text style={[styles.readyHint, isDark && styles.textSecondary]}>
          {readyMessage}
        </Text>
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
                ? `Fitting · test loss ${Number(progress.testLoss).toFixed(3)} · test acc ${(Number(progress.testAcc) * 100).toFixed(0)}%`
                : 'Starting…'}
          </Text>
        </View>
      )}

      {selectedModel !== 'gaussian-nb' && (
        <TouchableOpacity
          style={styles.advancedToggle}
          onPress={() => setAdvancedOpen(o => !o)}
          disabled={isTraining}>
          <Text style={[styles.advancedToggleText, isDark && styles.textDark]}>
            {advancedOpen ? '▾ Advanced' : '▸ Advanced'}
          </Text>
        </TouchableOpacity>
      )}

      {advancedOpen && selectedModel !== 'gaussian-nb' && (
        <>
          {selectedModel === 'logistic-l2' && (
            <>
              <Text style={[styles.label, isDark && styles.textDark]}>
                Regularization C: {aiLogisticC.toFixed(2)}
              </Text>
              <Slider
                style={styles.slider}
                minimumValue={0.01}
                maximumValue={10}
                step={0.01}
                value={aiLogisticC}
                disabled={isTraining}
                onSlidingComplete={v => setAILogisticC(parseFloat(v.toFixed(2)))}
                minimumTrackTintColor="#007AFF"
              />
              <Text style={[styles.label, isDark && styles.textDark]}>
                Max iterations: {aiLogisticMaxIter}
              </Text>
              <Slider
                style={styles.slider}
                minimumValue={50}
                maximumValue={2000}
                step={50}
                value={aiLogisticMaxIter}
                disabled={isTraining}
                onSlidingComplete={v => setAILogisticMaxIter(v)}
                minimumTrackTintColor="#007AFF"
              />
            </>
          )}

          {selectedModel === 'catboost' && (
            <>
              <Text style={[styles.label, isDark && styles.textDark]}>
                Learning rate: {aiCatboostLearningRate.toFixed(2)}
              </Text>
              <Slider
                style={styles.slider}
                minimumValue={0.01}
                maximumValue={0.5}
                step={0.01}
                value={aiCatboostLearningRate}
                disabled={isTraining}
                onSlidingComplete={v =>
                  setAICatboostLearningRate(parseFloat(v.toFixed(2)))
                }
                minimumTrackTintColor="#007AFF"
              />
              <Text style={[styles.label, isDark && styles.textDark]}>
                Tree depth: {aiCatboostDepth}
              </Text>
              <Slider
                style={styles.slider}
                minimumValue={1}
                maximumValue={8}
                step={1}
                value={aiCatboostDepth}
                disabled={isTraining}
                onSlidingComplete={v => setAICatboostDepth(v)}
                minimumTrackTintColor="#007AFF"
              />
              <Text style={[styles.label, isDark && styles.textDark]}>
                Min photos per leaf: {aiCatboostMinLeaf}
              </Text>
              <Slider
                style={styles.slider}
                minimumValue={2}
                maximumValue={32}
                step={1}
                value={aiCatboostMinLeaf}
                disabled={isTraining}
                onSlidingComplete={v => setAICatboostMinLeaf(v)}
                minimumTrackTintColor="#007AFF"
              />
              <Text style={[styles.label, isDark && styles.textDark]}>
                New trees per train: {aiCatboostMaxNewTrees}
              </Text>
              <Slider
                style={styles.slider}
                minimumValue={5}
                maximumValue={100}
                step={5}
                value={aiCatboostMaxNewTrees}
                disabled={isTraining}
                onSlidingComplete={v => setAICatboostMaxNewTrees(v)}
                minimumTrackTintColor="#007AFF"
              />
            </>
          )}

          {mlp && (
            <>
              <Text style={[styles.label, isDark && styles.textDark]}>
                Train epochs: {aiEpochs}
              </Text>
              <Slider
                style={styles.slider}
                minimumValue={10}
                maximumValue={200}
                step={5}
                value={aiEpochs}
                disabled={isTraining}
                onSlidingComplete={v => setAIEpochs(v)}
                minimumTrackTintColor="#007AFF"
              />
              <Text style={[styles.label, isDark && styles.textDark]}>
                Batch size: {sliderToMlpBatchSize(mlpBatchSliderT)}
              </Text>
              <Slider
                style={styles.slider}
                minimumValue={0}
                maximumValue={1}
                value={mlpBatchSliderT}
                disabled={isTraining}
                onValueChange={setMlpBatchSliderT}
                onSlidingComplete={t => {
                  const batch = sliderToMlpBatchSize(t);
                  setAIBatchSize(batch);
                  setMlpBatchSliderT(mlpBatchSizeToSlider(batch));
                }}
                minimumTrackTintColor="#007AFF"
              />
              <Text style={[styles.label, isDark && styles.textDark]}>
                Learning rate:{' '}
                {formatMlpLearningRate(sliderToMlpLearningRate(mlpLrSliderT))}
              </Text>
              <Slider
                style={styles.slider}
                minimumValue={0}
                maximumValue={1}
                value={mlpLrSliderT}
                disabled={isTraining}
                onValueChange={setMlpLrSliderT}
                onSlidingComplete={t => {
                  const lr = sliderToMlpLearningRate(t);
                  setAIStepSize(lr);
                  setMlpLrSliderT(mlpLearningRateToSlider(lr));
                }}
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
                    onSlidingComplete={v =>
                      setAIStepGamma(parseFloat(v.toFixed(2)))
                    }
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
                    onSlidingComplete={v =>
                      setAIExpDecay(parseFloat(v.toFixed(2)))
                    }
                    minimumTrackTintColor="#007AFF"
                  />
                </>
              )}
            </>
          )}
        </>
      )}

      {mlp && (
        <>
          <Text style={[styles.section, isDark && styles.textDark]}>Loss</Text>
          <MetricChart
            points={history}
            isDark={isDark}
            mode="loss"
            emptyHint="No training run yet for this model"
          />
          <Text style={[styles.section, isDark && styles.textDark]}>Accuracy</Text>
          <MetricChart
            points={history}
            isDark={isDark}
            mode="acc"
            emptyHint="No training run yet for this model"
          />
        </>
      )}

      {!!statusText && (
        <Text style={[styles.body, isDark && styles.textSecondary]}>
          {statusText}
        </Text>
      )}

      <Text style={[styles.body, isDark && styles.textSecondary]}>
        {hasModel && activeModelInfo
          ? `Scores use: ${TRAINABLE_MODELS.find(m => m.id === activeModelInfo.modelId)?.label ?? activeModelInfo.modelId}`
          : 'No trained model loaded for scoring yet.'}
      </Text>

      <Text style={[styles.body, isDark && styles.textSecondary, styles.datasetNote]}>
        Feature vectors from past swipes are kept even after photos are deleted.
        Dataset storage: {formatDatasetBytes(counts.bytes)}.
      </Text>

      <TouchableOpacity
        style={[
          styles.deleteBtn,
          (counts.total === 0 || isTraining) && styles.btnDisabled,
        ]}
        disabled={counts.total === 0 || isTraining}
        onPress={confirmDeleteDataset}>
        <Text style={styles.deleteBtnText}>Delete dataset</Text>
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
    marginTop: 4,
    marginBottom: 4,
  },
  section: {
    fontSize: 17,
    fontWeight: '700',
    color: '#333',
    marginTop: 16,
    marginBottom: 8,
  },
  sectionSpaced: {marginTop: 20},
  info: {fontSize: 18, color: '#007AFF'},
  label: {fontSize: 14, color: '#333', marginTop: 12},
  body: {fontSize: 14, color: '#555', marginBottom: 4},
  datasetNote: {marginTop: 20, lineHeight: 20},
  textDark: {color: '#fff'},
  textSecondary: {color: '#8e8e93'},
  hint: {fontSize: 12, color: '#8e8e93', marginTop: 4},
  warn: {fontSize: 13, color: '#ff9500', marginTop: 6},
  prereq: {
    fontSize: 13,
    color: '#ff9500',
    marginTop: 12,
    lineHeight: 18,
  },
  readyHint: {
    fontSize: 13,
    color: '#8e8e93',
    marginTop: 12,
  },
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
  actions: {marginTop: 12},
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
  advancedToggle: {
    marginTop: 16,
    paddingVertical: 8,
  },
  advancedToggleText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#007AFF',
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
