"""
manager.py - SQLite-backed memory manager with FTS5 search and conversation persistence.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("memory_manager")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_id(prefix: str = "mem") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class MemoryManager:
    """
    Thread-safe SQLite manager for persistent memories, FTS5 search,
    and multi-turn conversation history.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            import config
            self.db_path = str(getattr(config, "MEMORY_DB_PATH", Path("d:/eng/iris/core/mem/iris_memory.db")))
        else:
            self.db_path = str(db_path)

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite WAL mode, pragmas, and schema."""
        with self._lock, self._conn:
            self._conn.execute("PRAGMA journal_mode = WAL;")
            self._conn.execute("PRAGMA foreign_keys = ON;")
            schema_file = Path(__file__).parent / "schema.sql"
            if schema_file.exists():
                schema_sql = schema_file.read_text(encoding="utf-8")
                self._conn.executescript(schema_sql)
            # Record initial schema version if empty
            cur = self._conn.execute("SELECT COUNT(*) FROM schema_version;")
            if cur.fetchone()[0] == 0:
                self._conn.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (1, ?);",
                    (_now_iso(),),
                )

    def close(self) -> None:
        """Close SQLite database connection."""
        with self._lock:
            self._conn.close()

    # ---------------------------------------------------------------- Profile
    def get_profile(self) -> Dict[str, str]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT key, text FROM memories WHERE type = 'profile' AND is_deleted = 0;"
            )
            return {row["key"]: row["text"] for row in cur.fetchall()}

    def set_profile(self, key: str, value: str) -> None:
        now = _now_iso()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "SELECT id FROM memories WHERE type = 'profile' AND key = ? AND is_deleted = 0;",
                (key,),
            )
            row = cur.fetchone()
            if row:
                self._conn.execute(
                    "UPDATE memories SET text = ?, updated_at = ? WHERE id = ?;",
                    (value, now, row["id"]),
                )
            else:
                mem_id = _new_id("prof")
                self._conn.execute(
                    """INSERT INTO memories (id, type, key, text, created_at, updated_at)
                       VALUES (?, 'profile', ?, ?, ?, ?);""",
                    (mem_id, key, value, now, now),
                )

    # ---------------------------------------------------------------- Friends
    def get_friends(self) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT entity_name, entity_role, metadata FROM memories WHERE type = 'friend' AND is_deleted = 0;"
            )
            friends = []
            for row in cur.fetchall():
                meta = json.loads(row["metadata"] or "{}")
                friends.append({
                    "name": row["entity_name"],
                    "relation": row["entity_role"] or "friend",
                    "note": meta.get("note", ""),
                })
            return friends

    def add_friend(self, name: str, relation: str = "friend", note: str = "") -> None:
        now = _now_iso()
        meta = json.dumps({"note": note})
        with self._lock, self._conn:
            # Soft delete old entry with same name if exists
            self._conn.execute(
                "UPDATE memories SET is_deleted = 1, updated_at = ? WHERE type = 'friend' AND LOWER(entity_name) = LOWER(?);",
                (now, name),
            )
            mem_id = _new_id("rel")
            self._conn.execute(
                """INSERT INTO memories (id, type, text, entity_name, entity_role, metadata, created_at, updated_at)
                   VALUES (?, 'friend', ?, ?, ?, ?, ?, ?);""",
                (mem_id, f"{name} ({relation}) - {note}".strip(), name, relation.lower(), meta, now, now),
            )

    def remove_friend(self, name: str) -> None:
        now = _now_iso()
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE memories SET is_deleted = 1, updated_at = ? WHERE type = 'friend' AND LOWER(entity_name) = LOWER(?);",
                (now, name),
            )

    # ---------------------------------------------------------------- Files
    def get_files(self) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT entity_name, metadata FROM memories WHERE type = 'file' AND is_deleted = 0;"
            )
            files = []
            for row in cur.fetchall():
                meta = json.loads(row["metadata"] or "{}")
                files.append({
                    "name": row["entity_name"],
                    "path": meta.get("path", ""),
                    "note": meta.get("note", ""),
                })
            return files

    def add_file(self, name: str, path: str = "", note: str = "") -> None:
        now = _now_iso()
        meta = json.dumps({"path": path, "note": note})
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE memories SET is_deleted = 1, updated_at = ? WHERE type = 'file' AND LOWER(entity_name) = LOWER(?);",
                (now, name),
            )
            mem_id = _new_id("file")
            self._conn.execute(
                """INSERT INTO memories (id, type, text, entity_name, metadata, created_at, updated_at)
                   VALUES (?, 'file', ?, ?, ?, ?, ?);""",
                (mem_id, f"{name}: {path} ({note})", name, meta, now, now),
            )

    # ---------------------------------------------------------------- Story
    def get_story(self) -> List[str]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT text FROM memories WHERE type = 'story' AND is_deleted = 0 ORDER BY rowid ASC;"
            )
            return [row["text"] for row in cur.fetchall()]

    def add_story_chunk(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return False
        now = _now_iso()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "SELECT id FROM memories WHERE type = 'story' AND text = ? AND is_deleted = 0;",
                (text,),
            )
            if cur.fetchone():
                return False
            mem_id = _new_id("story")
            self._conn.execute(
                """INSERT INTO memories (id, type, text, created_at, updated_at)
                   VALUES (?, 'story', ?, ?, ?);""",
                (mem_id, text, now, now),
            )
            return True

    # ---------------------------------------------------------------- Long-Term Facts
    def get_long_term(self) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                """SELECT id, text, category, created_at, updated_at, access_count, tags
                   FROM memories WHERE type = 'fact' AND is_deleted = 0 ORDER BY rowid ASC;"""
            )
            facts = []
            for row in cur.fetchall():
                facts.append({
                    "id": row["id"],
                    "text": row["text"],
                    "category": row["category"] or "general",
                    "date": row["created_at"][:10],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "access_count": row["access_count"],
                    "tags": json.loads(row["tags"] or "[]"),
                })
            return facts

    def add_long_term(self, text: str, tags: Optional[List[str]] = None, category: str = "general") -> Optional[str]:
        text = text.strip()
        if not text:
            return None
        now = _now_iso()
        tags_json = json.dumps(tags or [])
        with self._lock, self._conn:
            cur = self._conn.execute(
                "SELECT id FROM memories WHERE type = 'fact' AND text = ? AND is_deleted = 0;",
                (text,),
            )
            if cur.fetchone():
                return None
            mem_id = _new_id("fact")
            self._conn.execute(
                """INSERT INTO memories (id, type, text, category, tags, created_at, updated_at)
                   VALUES (?, 'fact', ?, ?, ?, ?, ?);""",
                (mem_id, text, category, tags_json, now, now),
            )
            return mem_id

    def upsert_fact(self, text: str, category: str = "general") -> Tuple[str, Dict[str, Any]]:
        """
        Deduplicates or updates facts based on subject overlap.
        Returns (action, item_dict) where action is 'ADD', 'UPDATE', or 'NONE'.
        """
        from .extractor import FactExtractor

        text = text.strip()
        if not text:
            return "NONE", {}

        with self._lock:
            existing = self.get_long_term()
            existing_texts = [it["text"] for it in existing]
            action, update_idx = FactExtractor.determine_action(text, existing_texts)

            now = _now_iso()
            if action == "NONE":
                matched = next((it for it in existing if it["text"].lower() == text.lower()), None)
                return "NONE", matched or {"text": text}

            if action == "UPDATE" and 0 <= update_idx < len(existing):
                target = existing[update_idx]
                target_id = target["id"]
                with self._conn:
                    self._conn.execute(
                        "UPDATE memories SET text = ?, category = ?, updated_at = ? WHERE id = ?;",
                        (text, category, now, target_id),
                    )
                target["text"] = text
                target["category"] = category
                target["updated_at"] = now
                return "UPDATE", target

            # Action == ADD
            mem_id = self.add_long_term(text, category=category)
            new_item = {
                "id": mem_id,
                "text": text,
                "category": category,
                "date": now[:10],
                "created_at": now,
                "updated_at": now,
                "access_count": 0,
                "tags": [],
            }
            return "ADD", new_item

    def forget(self, target: str) -> Optional[Dict[str, Any]]:
        """
        Soft-delete a memory matching target keyword or text.
        Returns the forgotten item dictionary if found, else None.
        """
        now = _now_iso()
        target_clean = target.strip()
        if not target_clean:
            return None

        with self._lock, self._conn:
            # 1. Try exact or substring match in text
            cur = self._conn.execute(
                """SELECT id, type, text, category, entity_name FROM memories
                   WHERE is_deleted = 0 AND LOWER(text) LIKE LOWER(?)
                   ORDER BY rowid DESC LIMIT 1;""",
                (f"%{target_clean}%",),
            )
            row = cur.fetchone()

            # 2. If not found, try FTS5 match
            if not row:
                fts_hits = self.fts_search(target_clean, limit=1)
                if fts_hits:
                    cur = self._conn.execute(
                        "SELECT id, type, text, category, entity_name FROM memories WHERE id = ?;",
                        (fts_hits[0]["id"],),
                    )
                    row = cur.fetchone()

            if row:
                mem_id = row["id"]
                self._conn.execute(
                    "UPDATE memories SET is_deleted = 1, updated_at = ? WHERE id = ?;",
                    (now, mem_id),
                )
                return dict(row)
        return None

    # ---------------------------------------------------------------- Episodes
    def add_episode(
        self,
        summary: str,
        topics: Optional[List[str]] = None,
        emotional_tone: Optional[str] = None,
        session_id: str = "default",
        turn_count: int = 0,
    ) -> str:
        """Record an episode summary in the episodes table."""
        now = _now_iso()
        ep_id = _new_id("ep")
        topics_json = json.dumps(topics or [])
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO episodes (id, session_id, summary, topics, emotional_tone, start_time, end_time, turn_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?);""",
                (ep_id, session_id, summary, topics_json, emotional_tone, now, now, turn_count),
            )
        return ep_id

    def get_recent_episodes(self, limit: int = 3) -> List[Dict[str, Any]]:
        """Get latest episode summaries."""
        with self._lock:
            cur = self._conn.execute(
                """SELECT id, session_id, summary, topics, emotional_tone, start_time, end_time, turn_count
                   FROM episodes ORDER BY rowid DESC LIMIT ?;""",
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]

    def search_episodes(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Search episode summaries by keyword."""
        clean = "".join(c if c.isalnum() or c.isspace() else " " for c in query).strip()
        if not clean:
            return self.get_recent_episodes(limit=limit)
        with self._lock:
            cur = self._conn.execute(
                """SELECT id, session_id, summary, topics, emotional_tone, start_time, end_time, turn_count
                   FROM episodes WHERE LOWER(summary) LIKE LOWER(?) ORDER BY rowid DESC LIMIT ?;""",
                (f"%{clean}%", limit),
            )
            return [dict(r) for r in cur.fetchall()]

    # ---------------------------------------------------------------- FTS & Hybrid Search
    def fts_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search across all memories using SQLite FTS5."""
        clean_q = "".join(c if c.isalnum() or c.isspace() else " " for c in query).strip()
        if not clean_q:
            return []
        match_query = " OR ".join(clean_q.split())
        with self._lock:
            try:
                cur = self._conn.execute(
                    """SELECT m.id, m.type, m.text, m.category, m.entity_name, m.entity_role
                       FROM memories_fts f
                       JOIN memories m ON f.id = m.id
                       WHERE memories_fts MATCH ? AND m.is_deleted = 0
                       LIMIT ?;""",
                    (match_query, limit),
                )
                return [dict(row) for row in cur.fetchall()]
            except sqlite3.OperationalError as e:
                log.debug("FTS query failed for '%s': %s", match_query, e)
                return []

    def hybrid_search(
        self,
        query: str,
        embedder: Optional[Any] = None,
        top_k: int = 3,
        min_score: float = 0.35,
    ) -> List[Tuple[float, str]]:
        """
        Hybrid retrieval combining vector similarity, FTS5 lexical hits, and recency.
        Returns list of (score, text) tuples.
        """
        candidates: Dict[str, Tuple[float, str]] = {}

        # 1. FTS5 Lexical Search (fast text matching)
        fts_results = self.fts_search(query, limit=top_k * 2)
        for r in fts_results:
            text = r["text"]
            candidates[text] = (0.75, text)

        # 2. Embedding Search (if embedder available)
        if embedder and hasattr(embedder, "embed"):
            all_facts = self.get_long_term()
            fact_texts = [f["text"] for f in all_facts]
            if fact_texts:
                try:
                    q_vec = embedder.embed([query], query=True)[0]
                    doc_vecs = embedder.embed(fact_texts)
                    import numpy as np
                    sims = np.dot(doc_vecs, q_vec)
                    for text, sim in zip(fact_texts, sims):
                        score = float(sim)
                        if score >= min_score:
                            if text in candidates:
                                candidates[text] = (max(candidates[text][0], score) + 0.15, text)
                            else:
                                candidates[text] = (score, text)
                except Exception as e:
                    log.debug("Embedding hybrid scoring failed: %s", e)

        ranked = sorted(candidates.values(), key=lambda x: x[0], reverse=True)
        return ranked[:top_k]

    # ---------------------------------------------------------------- Conversations
    def add_turn(self, role: str, content: str, session_id: str = "default") -> None:
        """Record conversational turn to persistent SQLite history."""
        now = _now_iso()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "SELECT COALESCE(MAX(turn_index), -1) + 1 FROM conversations WHERE session_id = ?;",
                (session_id,),
            )
            turn_idx = cur.fetchone()[0]
            self._conn.execute(
                """INSERT INTO conversations (session_id, role, content, timestamp, turn_index)
                   VALUES (?, ?, ?, ?, ?);""",
                (session_id, role, content, now, turn_idx),
            )

    def get_recent_conversation(self, limit: int = 12, session_id: Optional[str] = None) -> List[Dict[str, str]]:
        """Fetch recent conversation turns for context construction."""
        with self._lock:
            if session_id:
                cur = self._conn.execute(
                    """SELECT role, content FROM conversations
                       WHERE session_id = ? ORDER BY id DESC LIMIT ?;""",
                    (session_id, limit),
                )
            else:
                cur = self._conn.execute(
                    "SELECT role, content FROM conversations ORDER BY id DESC LIMIT ?;",
                    (limit,),
                )
            rows = cur.fetchall()
            return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    # ---------------------------------------------------------------- Compatibility
    def load_full_store(self) -> Dict[str, Any]:
        """Produce the exact dictionary structure expected by RuleMemory."""
        return {
            "version": 1,
            "identity": {
                "profile": self.get_profile(),
                "friends": self.get_friends(),
                "files": self.get_files(),
            },
            "story": self.get_story(),
            "long_term": self.get_long_term(),
        }

    def import_from_dict(self, data: Dict[str, Any]) -> None:
        """Batch load data from a memory_store dictionary into SQLite."""
        with self._lock, self._conn:
            # Profile
            prof = data.get("identity", {}).get("profile", {})
            for k, v in prof.items():
                if v:
                    self.set_profile(k, str(v))

            # Friends
            for f in data.get("identity", {}).get("friends", []):
                self.add_friend(f.get("name", ""), f.get("relation", "friend"), f.get("note", ""))

            # Files
            for fl in data.get("identity", {}).get("files", []):
                self.add_file(fl.get("name", ""), fl.get("path", ""), fl.get("note", ""))

            # Stories
            for s in data.get("story", []):
                self.add_story_chunk(s)

            # Long term facts
            for lt in data.get("long_term", []):
                if isinstance(lt, dict):
                    self.add_long_term(lt.get("text", ""), tags=lt.get("tags"), category=lt.get("category", "general"))
                elif isinstance(lt, str):
                    self.add_long_term(lt)
