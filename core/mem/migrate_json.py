"""
migrate_json.py - Migrates legacy memory_store.json into SQLite iris_memory.db.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path if run directly
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from core.mem.manager import MemoryManager

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("migrate_json")


def migrate(json_path: Path = config.MEMORY_STORE_PATH, db_path: Path = config.MEMORY_DB_PATH) -> bool:
    """Migrate JSON memory store to SQLite database."""
    if not json_path.exists():
        log.warning(f"Source JSON file does not exist: {json_path}")
        return False

    log.info(f"Loading data from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    log.info(f"Connecting to SQLite database: {db_path}...")
    mgr = MemoryManager(str(db_path))

    mgr.import_from_dict(data)

    stats = mgr.load_full_store()
    log.info("Migration complete!")
    log.info(f"  • Profile keys: {len(stats['identity']['profile'])}")
    log.info(f"  • Friends/Relations: {len(stats['identity']['friends'])}")
    log.info(f"  • Files: {len(stats['identity']['files'])}")
    log.info(f"  • Stories: {len(stats['story'])}")
    log.info(f"  • Long-term facts: {len(stats['long_term'])}")

    mgr.close()
    return True


if __name__ == "__main__":
    migrate()
