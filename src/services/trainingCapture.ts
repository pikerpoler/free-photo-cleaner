import {FEATURE_SCHEMA_VERSION} from '../ai/cnnTypes';
import {extractPhotoFeatures} from './cnnTrainer';
import {getLabeledPhotoSamples} from './database';
import {
  deleteTrainingExample,
  getTrainingExampleMeta,
  upsertTrainingExample,
} from './trainingDataset';

const generations = new Map<string, number>();
const inflight = new Map<string, Promise<void>>();

function bumpGeneration(assetId: string): number {
  const next = (generations.get(assetId) ?? 0) + 1;
  generations.set(assetId, next);
  return next;
}

export function captureTrainingExample(
  assetId: string,
  uri: string,
  label: 0 | 1,
): void {
  const gen = bumpGeneration(assetId);
  const run = (async () => {
    const features = await extractPhotoFeatures(uri);
    if (generations.get(assetId) !== gen) return;
    if (!features) return;
    upsertTrainingExample(assetId, label, features);
  })().catch(() => {});
  inflight.set(assetId, run);
  void run.finally(() => {
    if (inflight.get(assetId) === run) inflight.delete(assetId);
  });
}

export function cancelTrainingExample(assetId: string): void {
  bumpGeneration(assetId);
  deleteTrainingExample(assetId);
}

export async function waitForTrainingCaptures(assetIds: string[]): Promise<void> {
  await Promise.all(assetIds.map(id => inflight.get(id) ?? Promise.resolve()));
}

/** Re-extract live labeled photos that are missing or on an older schema. */
export async function backfillTrainingDataset(): Promise<void> {
  const live = getLabeledPhotoSamples();
  const liveById = new Map(live.map(sample => [sample.id, sample]));
  const meta = getTrainingExampleMeta();
  const existingIds = new Set(meta.map(row => row.id));

  for (const row of meta) {
    if (row.schemaVersion === FEATURE_SCHEMA_VERSION) continue;
    const sample = liveById.get(row.id);
    if (!sample) {
      deleteTrainingExample(row.id);
    } else {
      await captureAndWait(sample.id, sample.uri, sample.label);
    }
  }

  for (const sample of live) {
    if (existingIds.has(sample.id)) continue;
    await captureAndWait(sample.id, sample.uri, sample.label);
  }
}

async function captureAndWait(
  assetId: string,
  uri: string,
  label: 0 | 1,
): Promise<void> {
  captureTrainingExample(assetId, uri, label);
  await (inflight.get(assetId) ?? Promise.resolve());
}
