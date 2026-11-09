import {NativeEventEmitter, NativeModules, Platform} from 'react-native';
import {
  FEATURE_DIM,
  ModelCheckpointInfo,
  ModelStatus,
  TrainingConfig,
  TrainingProgress,
  TrainingResult,
  YOLODetectorStatus,
} from '../ai/cnnTypes';

const {CNNTrainerModule} = NativeModules;

type ProgressListener = (progress: TrainingProgress) => void;
type CompleteListener = (result: TrainingResult) => void;

let emitter: NativeEventEmitter | null = null;

function getEmitter(): NativeEventEmitter | null {
  if (Platform.OS !== 'ios' || !CNNTrainerModule) return null;
  if (!emitter) {
    emitter = new NativeEventEmitter(CNNTrainerModule);
  }
  return emitter;
}

export async function isTrainerAvailable(): Promise<boolean> {
  if (Platform.OS !== 'ios' || !CNNTrainerModule) return false;
  try {
    return Boolean(await CNNTrainerModule.isAvailable());
  } catch {
    return false;
  }
}

export async function hasActiveModel(): Promise<boolean> {
  if (!(await isTrainerAvailable())) return false;
  return Boolean(await CNNTrainerModule.hasActiveModel());
}

export async function getActiveModelInfo(): Promise<ModelCheckpointInfo | null> {
  if (!(await isTrainerAvailable())) return null;
  const info = await CNNTrainerModule.getActiveModelInfo();
  if (!info || typeof info !== 'object') return null;
  return info as ModelCheckpointInfo;
}

export async function listModelCheckpoints(): Promise<ModelCheckpointInfo[]> {
  if (!(await isTrainerAvailable())) return [];
  try {
    const list = await CNNTrainerModule.listModelCheckpoints();
    return (list as ModelCheckpointInfo[]) || [];
  } catch {
    return [];
  }
}

export async function loadModel(modelId: string): Promise<void> {
  if (!(await isTrainerAvailable())) {
    throw new Error('Trainer unavailable');
  }
  await CNNTrainerModule.loadModel(modelId);
}

export async function getModelStatus(modelId: string): Promise<ModelStatus | null> {
  if (!(await isTrainerAvailable())) return null;
  try {
    const status = await CNNTrainerModule.getModelStatus(modelId);
    if (!status || typeof status !== 'object') return null;
    return status as ModelStatus;
  } catch {
    return null;
  }
}

export async function deleteModel(modelId: string): Promise<void> {
  if (!(await isTrainerAvailable())) return;
  await CNNTrainerModule.deleteModel(modelId);
}

export async function resetActiveModel(): Promise<void> {
  if (!(await isTrainerAvailable())) return;
  await CNNTrainerModule.resetActiveModel();
}

export async function stopTraining(): Promise<void> {
  if (!(await isTrainerAvailable())) return;
  await CNNTrainerModule.stopTraining();
}

export async function extractPhotoFeatures(
  uri: string,
): Promise<number[] | null> {
  if (!(await isTrainerAvailable())) return null;
  try {
    const vec = await CNNTrainerModule.extractFeatures(uri);
    if (!Array.isArray(vec) || vec.length !== FEATURE_DIM) return null;
    return vec.map((n: number) => Number(n));
  } catch {
    return null;
  }
}

export async function startTraining(
  config: TrainingConfig,
): Promise<TrainingResult> {
  if (!(await isTrainerAvailable())) {
    throw new Error('On-device CNN training is only available on iOS');
  }
  const result = await CNNTrainerModule.startTraining({
    ...config,
  });
  return result as TrainingResult;
}

export async function predictScores(uris: string[]): Promise<number[]> {
  if (!(await isTrainerAvailable())) {
    return uris.map(() => 0.5);
  }
  try {
    const scores = await CNNTrainerModule.predict(uris);
    return scores as number[];
  } catch {
    return uris.map(() => 0.5);
  }
}

export async function getYOLOStatus(): Promise<YOLODetectorStatus | null> {
  if (!(await isTrainerAvailable())) return null;
  try {
    const s = await CNNTrainerModule.getYOLOStatus();
    if (!s || typeof s !== 'object') return null;
    return s as YOLODetectorStatus;
  } catch {
    return null;
  }
}

export async function deleteCachedFeatures(uris: string[]): Promise<void> {
  if (!(await isTrainerAvailable())) return;
  try {
    await CNNTrainerModule.deleteCachedFeatures(uris);
  } catch {
    // best-effort
  }
}

export async function pruneFeatureCacheFromUris(uris: string[]): Promise<void> {
  if (!(await isTrainerAvailable())) return;
  try {
    await CNNTrainerModule.pruneFeatureCache(uris);
  } catch {
    // best-effort
  }
}

export function subscribeTrainingProgress(
  onProgress: ProgressListener,
  onComplete?: CompleteListener,
): () => void {
  const em = getEmitter();
  if (!em) return () => {};

  const pSub = em.addListener('trainingProgress', (event: TrainingProgress) => {
    onProgress(event);
  });
  const cSub = onComplete
    ? em.addListener('trainingComplete', (event: TrainingResult) => {
        onComplete(event);
      })
    : null;

  return () => {
    pSub.remove();
    cSub?.remove();
  };
}
