import {open, type DB} from '@op-engineering/op-sqlite';

let db: DB | null = null;

export function getDatabase(): DB {
  if (db) return db;

  db = open({name: 'aiphotocleaner.db'});

  db.executeSync(`
    CREATE TABLE IF NOT EXISTS kept_assets (
      id TEXT PRIMARY KEY,
      kept_at INTEGER NOT NULL
    )
  `);

  db.executeSync(`
    CREATE TABLE IF NOT EXISTS pending_deletes (
      id TEXT PRIMARY KEY,
      uri TEXT NOT NULL,
      filename TEXT,
      file_size INTEGER,
      media_type TEXT NOT NULL,
      added_at INTEGER NOT NULL
    )
  `);

  db.executeSync(`
    CREATE TABLE IF NOT EXISTS embedding_cache (
      asset_id TEXT PRIMARY KEY,
      embedding BLOB NOT NULL,
      computed_at INTEGER NOT NULL
    )
  `);

  return db;
}

// --- Kept assets ---

export function markAssetKept(assetId: string): void {
  const database = getDatabase();
  database.executeSync(
    'INSERT OR REPLACE INTO kept_assets (id, kept_at) VALUES (?, ?)',
    [assetId, Date.now()],
  );
}

export function removeAssetFromKept(assetId: string): void {
  const database = getDatabase();
  database.executeSync('DELETE FROM kept_assets WHERE id = ?', [assetId]);
}

export function getKeptAssetIds(): Set<string> {
  const database = getDatabase();
  const result = database.executeSync('SELECT id FROM kept_assets');
  const ids = new Set<string>();
  for (const row of result.rows) {
    ids.add(row.id as string);
  }
  return ids;
}

export function resetKeepHistory(): void {
  const database = getDatabase();
  database.executeSync('DELETE FROM kept_assets');
}

// --- Pending deletes ---

export function addPendingDelete(
  assetId: string,
  uri: string,
  filename: string,
  fileSize: number,
  mediaType: string,
): void {
  const database = getDatabase();
  database.executeSync(
    `INSERT OR REPLACE INTO pending_deletes (id, uri, filename, file_size, media_type, added_at)
     VALUES (?, ?, ?, ?, ?, ?)`,
    [assetId, uri, filename, fileSize, mediaType, Date.now()],
  );
}

export function removePendingDelete(assetId: string): void {
  const database = getDatabase();
  database.executeSync('DELETE FROM pending_deletes WHERE id = ?', [assetId]);
}

export function getPendingDeletesByType(
  mediaType: string,
): Array<{id: string; uri: string; filename: string; fileSize: number}> {
  const database = getDatabase();
  const result = database.executeSync(
    'SELECT id, uri, filename, file_size FROM pending_deletes WHERE media_type = ? ORDER BY added_at ASC',
    [mediaType],
  );
  return result.rows.map((row: Record<string, unknown>) => ({
    id: row.id as string,
    uri: row.uri as string,
    filename: row.filename as string,
    fileSize: row.file_size as number,
  }));
}

export function getPendingDeleteIds(): Set<string> {
  const database = getDatabase();
  const result = database.executeSync('SELECT id FROM pending_deletes');
  const ids = new Set<string>();
  for (const row of result.rows) {
    ids.add(row.id as string);
  }
  return ids;
}

export function clearPendingDeletesByType(mediaType: string): void {
  const database = getDatabase();
  database.executeSync('DELETE FROM pending_deletes WHERE media_type = ?', [
    mediaType,
  ]);
}
