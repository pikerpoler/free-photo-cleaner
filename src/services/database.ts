import {open, type DB} from '@op-engineering/op-sqlite';
import {MediaAsset, MediaType} from '../types/media';
import {LabeledSample} from '../ai/cnnTypes';
import {getSetting, setSetting, KEYS} from './storage';

let db: DB | null = null;
let keptIdsCache: Set<string> | null = null;
let pendingIdsCache: Set<string> | null = null;

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

  db.executeSync(`
    CREATE TABLE IF NOT EXISTS asset_catalog (
      id TEXT PRIMARY KEY,
      uri TEXT NOT NULL,
      media_type TEXT NOT NULL,
      filename TEXT,
      width INTEGER,
      height INTEGER,
      file_size INTEGER,
      creation_date INTEGER,
      duration REAL,
      has_exif INTEGER,
      is_screenshot INTEGER,
      is_whatsapp INTEGER,
      is_local INTEGER,
      updated_at INTEGER NOT NULL
    )
  `);

  try {
    db.executeSync('ALTER TABLE asset_catalog ADD COLUMN is_local INTEGER');
  } catch {
    // Column already exists on upgraded databases.
  }

  db.executeSync(`
    CREATE INDEX IF NOT EXISTS idx_asset_catalog_type_date
    ON asset_catalog(media_type, creation_date)
  `);

  db.executeSync(`
    CREATE TABLE IF NOT EXISTS swipe_decisions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      asset_id TEXT NOT NULL,
      decision TEXT NOT NULL CHECK(decision IN ('keep','delete')),
      media_type TEXT NOT NULL CHECK(media_type IN ('photo','video')),
      created_at INTEGER NOT NULL
    )
  `);

  db.executeSync(`
    CREATE INDEX IF NOT EXISTS idx_swipe_decisions_type_id
    ON swipe_decisions(media_type, id DESC)
  `);

  db.executeSync(`
    CREATE INDEX IF NOT EXISTS idx_swipe_decisions_asset
    ON swipe_decisions(asset_id)
  `);

  db.executeSync(`
    CREATE TABLE IF NOT EXISTS training_examples (
      asset_id TEXT PRIMARY KEY,
      label INTEGER NOT NULL CHECK(label IN (0, 1)),
      features TEXT NOT NULL,
      schema_version INTEGER NOT NULL,
      updated_at INTEGER NOT NULL
    )
  `);

  maybeBackfillSwipeDecisionsFromLegacy(db);

  return db;
}

export type SwipeDecisionType = 'keep' | 'delete';

export type PersistedUndoEntry = {
  type: SwipeDecisionType;
  asset: MediaAsset;
  queue: MediaType;
};

function maybeBackfillSwipeDecisionsFromLegacy(database: DB): void {
  if (getSetting(KEYS.SWIPE_DECISIONS_BACKFILL_DONE, false)) {
    return;
  }

  const existing = database.executeSync(
    'SELECT COUNT(*) AS n FROM swipe_decisions',
  );
  const n = (existing.rows[0]?.n as number) ?? 0;
  if (n > 0) {
    setSetting(KEYS.SWIPE_DECISIONS_BACKFILL_DONE, true);
    return;
  }

  type TimedRow = {
    assetId: string;
    decision: SwipeDecisionType;
    mediaType: MediaType;
    createdAt: number;
  };

  const rows: TimedRow[] = [];

  const keptRows = database.executeSync(
    'SELECT id, kept_at FROM kept_assets ORDER BY kept_at ASC',
  );
  for (const row of keptRows.rows) {
    const assetId = row.id as string;
    const keptAt = row.kept_at as number;
    const cat = database.executeSync(
      'SELECT media_type FROM asset_catalog WHERE id = ?',
      [assetId],
    );
    const mt =
      (cat.rows[0]?.media_type as MediaType | undefined) ?? 'photo';
    rows.push({
      assetId,
      decision: 'keep',
      mediaType: mt,
      createdAt: keptAt,
    });
  }

  const pendingRows = database.executeSync(
    `SELECT id, media_type, added_at FROM pending_deletes ORDER BY added_at ASC`,
  );
  for (const row of pendingRows.rows) {
    rows.push({
      assetId: row.id as string,
      decision: 'delete',
      mediaType: row.media_type as MediaType,
      createdAt: row.added_at as number,
    });
  }

  rows.sort((a, b) => a.createdAt - b.createdAt);

  database.executeSync('BEGIN');
  try {
    for (const r of rows) {
      database.executeSync(
        `INSERT INTO swipe_decisions (asset_id, decision, media_type, created_at)
         VALUES (?, ?, ?, ?)`,
        [r.assetId, r.decision, r.mediaType, r.createdAt],
      );
    }
    database.executeSync('COMMIT');
  } catch (e) {
    database.executeSync('ROLLBACK');
    throw e;
  }

  setSetting(KEYS.SWIPE_DECISIONS_BACKFILL_DONE, true);
}

/** Load undo stack in chronological swipe order (oldest first) for a media type. */
export function getPersistedUndoStack(mediaType: MediaType): PersistedUndoEntry[] {
  const database = getDatabase();
  const result = database.executeSync(
    `SELECT
       d.asset_id AS d_asset_id,
       d.decision AS d_decision,
       d.media_type AS d_media_type,
       c.id AS c_id,
       c.uri AS c_uri,
       c.media_type AS c_media_type,
       c.filename AS c_filename,
       c.width AS c_width,
       c.height AS c_height,
       c.file_size AS c_file_size,
       c.creation_date AS c_creation_date,
       c.duration AS c_duration,
       c.has_exif AS c_has_exif,
       c.is_screenshot AS c_is_screenshot,
       c.is_whatsapp AS c_is_whatsapp,
       p.id AS p_id,
       p.uri AS p_uri,
       p.filename AS p_filename,
       p.file_size AS p_file_size,
       p.media_type AS p_media_type
     FROM swipe_decisions d
     LEFT JOIN asset_catalog c ON c.id = d.asset_id
     LEFT JOIN pending_deletes p ON p.id = d.asset_id
     WHERE d.media_type = ?
     ORDER BY d.id ASC`,
    [mediaType],
  );
  const out: PersistedUndoEntry[] = [];
  for (const row of result.rows) {
    const decision = row.d_decision as SwipeDecisionType;
    const mt = row.d_media_type as MediaType;
    let asset: MediaAsset | null = null;
    if (row.c_id != null) {
      asset = rowToAsset({
        id: row.c_id,
        uri: row.c_uri,
        media_type: row.c_media_type,
        filename: row.c_filename,
        width: row.c_width,
        height: row.c_height,
        file_size: row.c_file_size,
        creation_date: row.c_creation_date,
        duration: row.c_duration,
        has_exif: row.c_has_exif,
        is_screenshot: row.c_is_screenshot,
        is_whatsapp: row.c_is_whatsapp,
      });
    } else if (decision === 'delete' && row.p_id != null) {
      asset = {
        id: row.p_id as string,
        uri: row.p_uri as string,
        type: (row.p_media_type as MediaType) ?? mt,
        filename: (row.p_filename as string) || 'unknown',
        width: 0,
        height: 0,
        fileSize: (row.p_file_size as number) || 0,
        creationDate: 0,
        hasExif: false,
        isScreenshot: false,
        isWhatsApp: false,
      };
    }
    if (asset) {
      out.push({type: decision, asset, queue: mt});
    }
  }
  return out;
}

// --- Kept assets ---

export function markAssetKept(assetId: string, mediaType: MediaType): void {
  const database = getDatabase();
  const now = Date.now();
  database.executeSync('BEGIN');
  try {
    database.executeSync(
      'INSERT OR REPLACE INTO kept_assets (id, kept_at) VALUES (?, ?)',
      [assetId, now],
    );
    database.executeSync(
      `INSERT INTO swipe_decisions (asset_id, decision, media_type, created_at)
       VALUES (?, 'keep', ?, ?)`,
      [assetId, mediaType, now],
    );
    database.executeSync('COMMIT');
  } catch (e) {
    database.executeSync('ROLLBACK');
    throw e;
  }
  keptIdsCache?.add(assetId);
}

export function removeAssetFromKept(assetId: string): void {
  const database = getDatabase();
  database.executeSync('BEGIN');
  try {
    database.executeSync('DELETE FROM kept_assets WHERE id = ?', [assetId]);
    removeLatestSwipeDecisionForAssetInTx(database, assetId);
    database.executeSync('COMMIT');
  } catch (e) {
    database.executeSync('ROLLBACK');
    throw e;
  }
  keptIdsCache?.delete(assetId);
}

export function getKeptAssetIds(): ReadonlySet<string> {
  if (keptIdsCache) return keptIdsCache;
  const database = getDatabase();
  const result = database.executeSync('SELECT id FROM kept_assets');
  const ids = new Set<string>();
  for (const row of result.rows) {
    ids.add(row.id as string);
  }
  keptIdsCache = ids;
  return ids;
}

export function resetKeepHistory(): void {
  const database = getDatabase();
  database.executeSync('BEGIN');
  try {
    database.executeSync('DELETE FROM kept_assets');
    clearSwipeDecisionsKeepInTx(database);
    database.executeSync('COMMIT');
  } catch (e) {
    database.executeSync('ROLLBACK');
    throw e;
  }
  keptIdsCache = new Set();
}

function clearSwipeDecisionsKeepInTx(database: DB): void {
  database.executeSync(`DELETE FROM swipe_decisions WHERE decision = 'keep'`);
}

// --- Pending deletes ---

export function addPendingDelete(
  assetId: string,
  uri: string,
  filename: string,
  fileSize: number,
  mediaType: MediaType,
): void {
  const database = getDatabase();
  const now = Date.now();
  database.executeSync('BEGIN');
  try {
    database.executeSync(
      `INSERT OR REPLACE INTO pending_deletes (id, uri, filename, file_size, media_type, added_at)
       VALUES (?, ?, ?, ?, ?, ?)`,
      [assetId, uri, filename, fileSize, mediaType, now],
    );
    database.executeSync(
      `INSERT INTO swipe_decisions (asset_id, decision, media_type, created_at)
       VALUES (?, 'delete', ?, ?)`,
      [assetId, mediaType, now],
    );
    database.executeSync('COMMIT');
  } catch (e) {
    database.executeSync('ROLLBACK');
    throw e;
  }
  pendingIdsCache?.add(assetId);
}

export function removePendingDelete(assetId: string): void {
  const database = getDatabase();
  database.executeSync('BEGIN');
  try {
    database.executeSync('DELETE FROM pending_deletes WHERE id = ?', [assetId]);
    removeLatestSwipeDecisionForAssetInTx(database, assetId);
    database.executeSync('COMMIT');
  } catch (e) {
    database.executeSync('ROLLBACK');
    throw e;
  }
  pendingIdsCache?.delete(assetId);
}

function removeLatestSwipeDecisionForAssetInTx(database: DB, assetId: string): void {
  database.executeSync(
    `DELETE FROM swipe_decisions WHERE id = (
       SELECT id FROM swipe_decisions WHERE asset_id = ? ORDER BY id DESC LIMIT 1
     )`,
    [assetId],
  );
}

export function getPendingDeleteAggregate(
  mediaType: MediaType,
): {count: number; totalBytes: number} {
  const database = getDatabase();
  const result = database.executeSync(
    `SELECT COUNT(*) AS n, COALESCE(SUM(file_size), 0) AS bytes
     FROM pending_deletes WHERE media_type = ?`,
    [mediaType],
  );
  const row = result.rows[0] as Record<string, unknown> | undefined;
  return {
    count: (row?.n as number) || 0,
    totalBytes: (row?.bytes as number) || 0,
  };
}

export function getKeptAggregate(mediaType: MediaType): {
  count: number;
  totalBytes: number;
} {
  const database = getDatabase();
  if (mediaType === 'photo') {
    const result = database.executeSync(
      `SELECT COUNT(*) AS n, COALESCE(SUM(COALESCE(c.file_size, 0)), 0) AS bytes
       FROM kept_assets k
       LEFT JOIN asset_catalog c ON c.id = k.id
       WHERE c.media_type = 'photo' OR c.media_type IS NULL`,
    );
    const row = result.rows[0] as Record<string, unknown> | undefined;
    return {
      count: (row?.n as number) || 0,
      totalBytes: (row?.bytes as number) || 0,
    };
  }
  const result = database.executeSync(
    `SELECT COUNT(*) AS n, COALESCE(SUM(c.file_size), 0) AS bytes
     FROM kept_assets k
     INNER JOIN asset_catalog c ON c.id = k.id
     WHERE c.media_type = 'video'`,
  );
  const row = result.rows[0] as Record<string, unknown> | undefined;
  return {
    count: (row?.n as number) || 0,
    totalBytes: (row?.bytes as number) || 0,
  };
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

export function getPendingDeleteIds(): ReadonlySet<string> {
  if (pendingIdsCache) return pendingIdsCache;
  const database = getDatabase();
  const result = database.executeSync('SELECT id FROM pending_deletes');
  const ids = new Set<string>();
  for (const row of result.rows) {
    ids.add(row.id as string);
  }
  pendingIdsCache = ids;
  return ids;
}

export function clearPendingDeletesByType(mediaType: MediaType): void {
  const database = getDatabase();
  database.executeSync('BEGIN');
  try {
    database.executeSync('DELETE FROM pending_deletes WHERE media_type = ?', [
      mediaType,
    ]);
    database.executeSync('DELETE FROM swipe_decisions WHERE media_type = ?', [
      mediaType,
    ]);
    database.executeSync('COMMIT');
  } catch (e) {
    database.executeSync('ROLLBACK');
    throw e;
  }
  pendingIdsCache = null;
}

/** Kept (0) + pending-delete photos (1) for training */
export function getLabeledPhotoSamples(): LabeledSample[] {
  const database = getDatabase();
  const samples: LabeledSample[] = [];

  const kept = database.executeSync(
    `SELECT k.id, COALESCE(c.uri, k.id) AS uri
     FROM kept_assets k
     LEFT JOIN asset_catalog c ON c.id = k.id
     WHERE c.media_type IS NULL OR c.media_type = 'photo'`,
  );
  for (const row of kept.rows) {
    samples.push({
      id: row.id as string,
      uri: row.uri as string,
      label: 0,
    });
  }

  const pending = database.executeSync(
    `SELECT id, uri FROM pending_deletes WHERE media_type = 'photo'`,
  );
  for (const row of pending.rows) {
    samples.push({
      id: row.id as string,
      uri: row.uri as string,
      label: 1,
    });
  }

  return samples;
}

export function getLabeledCounts(): {kept: number; pendingDelete: number} {
  const database = getDatabase();
  const kept = database.executeSync('SELECT COUNT(*) AS n FROM kept_assets');
  const pending = database.executeSync(
    `SELECT COUNT(*) AS n FROM pending_deletes WHERE media_type = 'photo'`,
  );
  return {
    kept: (kept.rows[0]?.n as number) || 0,
    pendingDelete: (pending.rows[0]?.n as number) || 0,
  };
}

// --- Asset catalog ---

function rowToAsset(row: Record<string, unknown>): MediaAsset {
  return {
    id: row.id as string,
    uri: row.uri as string,
    type: row.media_type as MediaType,
    filename: (row.filename as string) || 'unknown',
    width: (row.width as number) || 0,
    height: (row.height as number) || 0,
    fileSize: (row.file_size as number) || 0,
    creationDate: (row.creation_date as number) || 0,
    duration: row.duration != null ? (row.duration as number) : undefined,
    hasExif: Boolean(row.has_exif),
    isScreenshot: Boolean(row.is_screenshot),
    isWhatsApp: Boolean(row.is_whatsapp),
    isLocal:
      row.is_local == null ? undefined : Boolean(Number(row.is_local)),
  };
}

export function upsertCatalogAssets(assets: MediaAsset[]): void {
  if (assets.length === 0) return;
  const database = getDatabase();
  const now = Date.now();
  database.executeSync('BEGIN');
  try {
    for (const a of assets) {
      database.executeSync(
        `INSERT INTO asset_catalog
         (id, uri, media_type, filename, width, height, file_size, creation_date, duration, has_exif, is_screenshot, is_whatsapp, is_local, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET
           uri=excluded.uri,
           media_type=excluded.media_type,
           filename=excluded.filename,
           width=excluded.width,
           height=excluded.height,
           file_size=excluded.file_size,
           creation_date=excluded.creation_date,
           duration=excluded.duration,
           has_exif=excluded.has_exif,
           is_screenshot=excluded.is_screenshot,
           is_whatsapp=excluded.is_whatsapp,
           is_local=COALESCE(excluded.is_local, asset_catalog.is_local),
           updated_at=excluded.updated_at`,
        [
          a.id,
          a.uri,
          a.type,
          a.filename,
          a.width,
          a.height,
          a.fileSize,
          a.creationDate,
          a.duration ?? null,
          a.hasExif ? 1 : 0,
          a.isScreenshot ? 1 : 0,
          a.isWhatsApp ? 1 : 0,
          a.isLocal === undefined ? null : a.isLocal ? 1 : 0,
          now,
        ],
      );
    }
    database.executeSync('COMMIT');
  } catch (e) {
    database.executeSync('ROLLBACK');
    throw e;
  }
}

export function updateCatalogLocalFlags(
  flags: Array<{id: string; isLocal: boolean}>,
): void {
  if (flags.length === 0) return;
  const database = getDatabase();
  database.executeSync('BEGIN');
  try {
    for (const row of flags) {
      database.executeSync('UPDATE asset_catalog SET is_local = ? WHERE id = ?', [
        row.isLocal ? 1 : 0,
        row.id,
      ]);
    }
    database.executeSync('COMMIT');
  } catch (e) {
    database.executeSync('ROLLBACK');
    throw e;
  }
}

export function getCatalogAssets(
  mediaType: MediaType,
  dateFrom?: number,
  dateTo?: number,
): MediaAsset[] {
  const database = getDatabase();
  let sql = 'SELECT * FROM asset_catalog WHERE media_type = ?';
  const params: (string | number)[] = [mediaType];
  if (dateFrom != null) {
    sql += ' AND creation_date >= ?';
    params.push(dateFrom);
  }
  if (dateTo != null) {
    sql += ' AND creation_date <= ?';
    params.push(dateTo);
  }
  sql += ' ORDER BY creation_date DESC';
  const result = database.executeSync(sql, params);
  return result.rows.map((row: Record<string, unknown>) => rowToAsset(row));
}

export function getCatalogCount(mediaType: MediaType): number {
  const database = getDatabase();
  const result = database.executeSync(
    'SELECT COUNT(*) AS n FROM asset_catalog WHERE media_type = ?',
    [mediaType],
  );
  return (result.rows[0]?.n as number) || 0;
}

export function pruneCatalogToIds(mediaType: MediaType, ids: Set<string>): void {
  const database = getDatabase();
  const existing = database.executeSync(
    'SELECT id FROM asset_catalog WHERE media_type = ?',
    [mediaType],
  );
  database.executeSync('BEGIN');
  try {
    for (const row of existing.rows) {
      const id = row.id as string;
      if (!ids.has(id)) {
        database.executeSync('DELETE FROM asset_catalog WHERE id = ?', [id]);
      }
    }
    database.executeSync('COMMIT');
  } catch (e) {
    database.executeSync('ROLLBACK');
    throw e;
  }
}
