import {FEATURE_DIM, FEATURE_SCHEMA_VERSION} from '../ai/cnnTypes';
import {getDatabase} from './database';

export type TrainingExample = {
  id: string;
  label: 0 | 1;
  features: number[];
};

export type TrainingExampleMeta = {
  id: string;
  schemaVersion: number;
};

export type TrainingDatasetStats = {
  kept: number;
  pendingDelete: number;
  total: number;
  bytes: number;
};

export function upsertTrainingExample(
  assetId: string,
  label: 0 | 1,
  features: number[],
): void {
  if (features.length !== FEATURE_DIM) return;
  const db = getDatabase();
  db.executeSync(
    `INSERT OR REPLACE INTO training_examples
     (asset_id, label, features, schema_version, updated_at)
     VALUES (?, ?, ?, ?, ?)`,
    [
      assetId,
      label,
      float32ToBase64(features),
      FEATURE_SCHEMA_VERSION,
      Date.now(),
    ],
  );
}

export function deleteTrainingExample(assetId: string): void {
  const db = getDatabase();
  db.executeSync('DELETE FROM training_examples WHERE asset_id = ?', [assetId]);
}

export function hasTrainingExample(assetId: string): boolean {
  const db = getDatabase();
  const result = db.executeSync(
    'SELECT 1 FROM training_examples WHERE asset_id = ? LIMIT 1',
    [assetId],
  );
  return result.rows.length > 0;
}

export function clearTrainingDataset(): void {
  const db = getDatabase();
  db.executeSync('DELETE FROM training_examples');
}

export function getTrainingExampleMeta(): TrainingExampleMeta[] {
  const db = getDatabase();
  const result = db.executeSync(
    'SELECT asset_id, schema_version FROM training_examples',
  );
  return result.rows.map((row: Record<string, unknown>) => ({
    id: row.asset_id as string,
    schemaVersion: (row.schema_version as number) || 0,
  }));
}

export function getTrainingExamples(): TrainingExample[] {
  const db = getDatabase();
  const result = db.executeSync(
    'SELECT asset_id, label, features FROM training_examples WHERE schema_version = ?',
    [FEATURE_SCHEMA_VERSION],
  );
  const out: TrainingExample[] = [];
  for (const row of result.rows) {
    const features = decodeFeatureVector(row.features);
    if (features == null) continue;
    out.push({
      id: row.asset_id as string,
      label: (row.label as number) >= 0.5 ? 1 : 0,
      features,
    });
  }
  return out;
}

export function getTrainingDatasetStats(): TrainingDatasetStats {
  const db = getDatabase();
  const result = db.executeSync(
    `SELECT
       COUNT(*) AS n,
       COALESCE(SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END), 0) AS kept,
       COALESCE(SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END), 0) AS pending_delete,
       COALESCE(SUM(LENGTH(features)), 0) AS bytes
     FROM training_examples`,
  );
  const row = result.rows[0] as Record<string, unknown> | undefined;
  const kept = (row?.kept as number) || 0;
  const pendingDelete = (row?.pending_delete as number) || 0;
  return {
    kept,
    pendingDelete,
    total: (row?.n as number) || kept + pendingDelete,
    bytes: (row?.bytes as number) || 0,
  };
}

function decodeFeatureVector(raw: unknown): number[] | null {
  if (typeof raw !== 'string' || raw.length === 0) return null;
  try {
    const binary = globalThis.atob(raw);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    const floats = new Float32Array(
      bytes.buffer,
      bytes.byteOffset,
      Math.floor(bytes.byteLength / 4),
    );
    if (floats.length !== FEATURE_DIM) return null;
    return Array.from(floats);
  } catch {
    return null;
  }
}

function float32ToBase64(values: number[]): string {
  const floats = new Float32Array(values);
  const bytes = new Uint8Array(floats.buffer, floats.byteOffset, floats.byteLength);
  let binary = '';
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return globalThis.btoa(binary);
}
