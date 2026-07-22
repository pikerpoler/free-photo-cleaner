import {NativeEventEmitter, NativeModules, Platform} from 'react-native';
import {
  ModelCheckpointInfo,
  TrainingConfig,
  TrainingProgress,
  TrainingResult,
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

export async function startTraining(
  config: TrainingConfig,
): Promise<TrainingResult> {
  if (!(await isTrainerAvailable())) {
    throw new Error('On-device CNN training is only available on iOS');
  }
  const result = await CNNTrainerModule.startTraining({
    ...config,
    trainRatio: config.trainRatio ?? 0.8,
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
