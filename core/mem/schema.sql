-- Iris Companion SQLite Schema
-- Version: 1

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

-- Core memories table
CREATE TABLE IF NOT EXISTS memories (
    id            TEXT PRIMARY KEY,
    type          TEXT NOT NULL,           -- profile | friend | file | story | fact | procedure
    key           TEXT,                    -- key for profile (e.g. 'name', 'city')
    text          TEXT NOT NULL,           -- human-readable text
    category      TEXT DEFAULT 'general',  -- preference | habit | bio | general
    source        TEXT DEFAULT 'explicit', -- explicit | inferred | extracted | system
    confidence    REAL DEFAULT 1.0,        -- 0.0 to 1.0
    importance    REAL DEFAULT 0.5,        -- 0.0 to 1.0
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    last_accessed TEXT,
    access_count  INTEGER DEFAULT 0,
    tags          TEXT DEFAULT '[]',       -- JSON array
    metadata      TEXT DEFAULT '{}',       -- JSON object for extra details
    entity_name   TEXT,                    -- name for friends or files
    entity_role   TEXT,                    -- role/relation (e.g. 'friend', 'sister')
    is_archived   INTEGER DEFAULT 0,
    is_deleted    INTEGER DEFAULT 0
);

-- Indexes for zero-latency lookups
CREATE INDEX IF NOT EXISTS idx_mem_type ON memories(type) WHERE is_deleted = 0;
CREATE INDEX IF NOT EXISTS idx_mem_key ON memories(type, key) WHERE is_deleted = 0;
CREATE INDEX IF NOT EXISTS idx_mem_entity ON memories(type, entity_name) WHERE is_deleted = 0;
CREATE INDEX IF NOT EXISTS idx_mem_updated ON memories(updated_at DESC);

-- Full-text search (FTS5) table
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    id UNINDEXED,
    text,
    category,
    entity_name,
    entity_role
);

-- Automatic FTS sync triggers
CREATE TRIGGER IF NOT EXISTS trg_mem_ai AFTER INSERT ON memories
WHEN new.is_deleted = 0
BEGIN
    INSERT INTO memories_fts(id, text, category, entity_name, entity_role)
    VALUES (new.id, new.text, new.category, new.entity_name, new.entity_role);
END;

CREATE TRIGGER IF NOT EXISTS trg_mem_ad AFTER DELETE ON memories
BEGIN
    DELETE FROM memories_fts WHERE id = old.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_mem_au AFTER UPDATE ON memories
BEGIN
    DELETE FROM memories_fts WHERE id = old.id;
    INSERT INTO memories_fts(id, text, category, entity_name, entity_role)
    SELECT new.id, new.text, new.category, new.entity_name, new.entity_role
    WHERE new.is_deleted = 0;
END;

-- Conversation history table
CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role       TEXT NOT NULL,           -- user | assistant
    content    TEXT NOT NULL,
    timestamp  TEXT NOT NULL,
    turn_index INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id, turn_index);

-- Episode summaries table
CREATE TABLE IF NOT EXISTS episodes (
    id             TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL,
    summary        TEXT NOT NULL,
    topics         TEXT DEFAULT '[]',
    emotional_tone TEXT,
    start_time     TEXT NOT NULL,
    end_time       TEXT NOT NULL,
    turn_count     INTEGER DEFAULT 0,
    importance     REAL DEFAULT 0.5
);
