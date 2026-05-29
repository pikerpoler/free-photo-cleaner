import {InteractionManager} from 'react-native';
import {AIModelSize} from '../types/media';
import {ClassifierWeights, ClassifierGradients, TrainingEntry} from './types';
import {
  forward,
  backward,
  sgdStep,
  createGradients,
  zeroGradients,
} from './classifier';
import {getCachedEmbedding, setCachedEmbedding} from './embeddingCache';
import {computeEmbedding} from './embeddingModel';
import {saveWeights} from './weights';

export interface TrainerState {
  queue: TrainingEntry[];
  isProcessing: boolean;
  gradients: ClassifierGradients | null;
  processedInBatch: number;
}

let trainerState: TrainerState = {
  queue: [],
  isProcessing: false,
  gradients: null,
  processedInBatch: 0,
};

let currentWeights: ClassifierWeights | null = null;
let currentModelSize: AIModelSize = 'tiny';
let batchSize = 10;
let stepSize = 0.01;
let onStateChange: (() => void) | null = null;

export function configureTrainer(config: {
  weights: ClassifierWeights | null;
  modelSize: AIModelSize;
  batch: number;
  lr: number;
  onChange: () => void;
}): void {
  currentWeights = config.weights;
  currentModelSize = config.modelSize;
  batchSize = config.batch;
  stepSize = config.lr;
  onStateChange = config.onChange;

  if (currentWeights && !trainerState.gradients) {
    trainerState.gradients = createGradients(currentWeights);
  }
}

export function updateTrainerWeights(weights: ClassifierWeights | null): void {
  currentWeights = weights;
  if (weights && !trainerState.gradients) {
    trainerState.gradients = createGradients(weights);
  }
}

export function updateTrainerConfig(batch: number, lr: number): void {
  batchSize = batch;
  stepSize = lr;
}

export function getTrainerState(): TrainerState {
  return trainerState;
}

export function getQueueSize(): number {
  return trainerState.queue.length;
}

export function enqueueTraining(entry: TrainingEntry): void {
  trainerState.queue.push(entry);
  onStateChange?.();
  scheduleProcessing();
}

export function clearTrainingQueue(): void {
  trainerState.queue = [];
  trainerState.processedInBatch = 0;
  if (trainerState.gradients) {
    zeroGradients(trainerState.gradients);
  }
  onStateChange?.();
}

/**
 * Process all remaining items, performing partial SGD if needed.
 * Used for "Train & Delete" flow.
 */
export async function flushTraining(): Promise<void> {
  if (!currentWeights || trainerState.queue.length === 0) {
    clearTrainingQueue();
    return;
  }

  // Process all remaining items
  while (trainerState.queue.length > 0) {
    await processOneItem();
  }

  // If there are accumulated gradients from a partial batch, do a step
  if (trainerState.processedInBatch > 0 && trainerState.gradients) {
    sgdStep(currentWeights, trainerState.gradients, stepSize);
    await saveWeights(currentModelSize, currentWeights);
    trainerState.processedInBatch = 0;
    zeroGradients(trainerState.gradients);
  }

  onStateChange?.();
}

/**
 * Do a partial SGD step on whatever gradients have been accumulated.
 * Used when queue is < batchSize and user wants to train before delete.
 */
export async function flushPartialBatch(): Promise<void> {
  if (
    !currentWeights ||
    !trainerState.gradients ||
    trainerState.processedInBatch === 0
  ) {
    return;
  }

  sgdStep(currentWeights, trainerState.gradients, stepSize);
  await saveWeights(currentModelSize, currentWeights);
  trainerState.processedInBatch = 0;
  zeroGradients(trainerState.gradients);
  onStateChange?.();
}

let scheduledHandle: ReturnType<typeof InteractionManager.runAfterInteractions> | null = null;

function scheduleProcessing(): void {
  if (trainerState.isProcessing || trainerState.queue.length === 0) return;
  if (scheduledHandle) return;

  scheduledHandle = InteractionManager.runAfterInteractions(() => {
    scheduledHandle = null;
    processNext();
  });
}

async function processNext(): Promise<void> {
  if (trainerState.queue.length === 0 || !currentWeights) {
    trainerState.isProcessing = false;
    onStateChange?.();
    return;
  }

  trainerState.isProcessing = true;
  onStateChange?.();

  await processOneItem();

  // Check if we've accumulated a full batch
  if (trainerState.processedInBatch >= batchSize && trainerState.gradients) {
    sgdStep(currentWeights, trainerState.gradients, stepSize);
    await saveWeights(currentModelSize, currentWeights);
    trainerState.processedInBatch = 0;
    zeroGradients(trainerState.gradients);
    onStateChange?.();
  }

  // Yield back to UI, then process next if queue isn't empty
  if (trainerState.queue.length > 0) {
    scheduledHandle = InteractionManager.runAfterInteractions(() => {
      scheduledHandle = null;
      processNext();
    });
  } else {
    trainerState.isProcessing = false;
    onStateChange?.();
  }
}

async function processOneItem(): Promise<void> {
  if (!currentWeights || !trainerState.gradients) return;

  const entry = trainerState.queue.shift();
  if (!entry) return;

  // Get or compute embedding
  let embedding = getCachedEmbedding(entry.assetId);
  if (!embedding) {
    try {
      embedding = await computeEmbedding(entry.assetId);
      setCachedEmbedding(entry.assetId, embedding);
    } catch {
      // If embedding computation fails, skip this item
      onStateChange?.();
      return;
    }
  }

  // Forward + backward pass
  const fwdResult = forward(currentWeights, embedding);
  backward(currentWeights, trainerState.gradients, fwdResult, entry.label);
  trainerState.processedInBatch++;
  onStateChange?.();
}
