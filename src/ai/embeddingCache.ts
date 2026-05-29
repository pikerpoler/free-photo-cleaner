import {getDatabase} from '../services/database';
import {EMBEDDING_DIM} from './types';

export function getCachedEmbedding(assetId: string): Float32Array | null {
  const db = getDatabase();
  const result = db.executeSync(
    'SELECT embedding FROM embedding_cache WHERE asset_id = ?',
    [assetId],
  );
  if (result.rows.length === 0) return null;

  const blob = result.rows[0].embedding;
  if (!blob) return null;

  // The blob is stored as a base64 string
  const binary = globalThis.atob(blob as string);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Float32Array(bytes.buffer);
}

export function setCachedEmbedding(
  assetId: string,
  embedding: Float32Array,
): void {
  const db = getDatabase();
  const base64 = float32ToBase64(embedding);
  db.executeSync(
    'INSERT OR REPLACE INTO embedding_cache (asset_id, embedding, computed_at) VALUES (?, ?, ?)',
    [assetId, base64, Date.now()],
  );
}

export function hasCachedEmbedding(assetId: string): boolean {
  const db = getDatabase();
  const result = db.executeSync(
    'SELECT 1 FROM embedding_cache WHERE asset_id = ? LIMIT 1',
    [assetId],
  );
  return result.rows.length > 0;
}

export function clearEmbeddingCache(): void {
  const db = getDatabase();
  db.executeSync('DELETE FROM embedding_cache');
}

export function getEmbeddingCacheCount(): number {
  const db = getDatabase();
  const result = db.executeSync('SELECT COUNT(*) as cnt FROM embedding_cache');
  return (result.rows[0]?.cnt as number) ?? 0;
}

function float32ToBase64(arr: Float32Array): string {
  const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
  let binary = '';
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return globalThis.btoa(binary);
}

export {EMBEDDING_DIM};
