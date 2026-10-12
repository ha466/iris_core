"""
test_memory_sqlite.py - Runnable assert-based check for SQLite memory & conversation history.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from core.mem.manager import MemoryManager
from core.mem.memory import RuleMemory
from core.pipeline.companion import CompanionBrain


def test_manager_crud():
    print("Testing MemoryManager CRUD & FTS5...")
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "test_mem.db"
        mgr = MemoryManager(str(db_file))

        # Profile
        mgr.set_profile("name", "Alice")
        mgr.set_profile("city", "Kyoto")
        prof = mgr.get_profile()
        assert prof["name"] == "Alice"
        assert prof["city"] == "Kyoto"

        # Update profile
        mgr.set_profile("city", "Osaka")
        assert mgr.get_profile()["city"] == "Osaka"

        # Friends
        mgr.add_friend("Bob", "friend", "Loves hiking")
        mgr.add_friend("Charlie", "brother", "Twin brother")
        friends = mgr.get_friends()
        assert len(friends) == 2
        bob = next(f for f in friends if f["name"] == "Bob")
        assert bob["relation"] == "friend"
        assert bob["note"] == "Loves hiking"

        # Remove friend
        mgr.remove_friend("Bob")
        assert len(mgr.get_friends()) == 1

        # Files
        mgr.add_file("taxes", "~/Docs/taxes.pdf", "2025 tax return")
        files = mgr.get_files()
        assert len(files) == 1
        assert files[0]["name"] == "taxes"

        # Story
        mgr.add_story_chunk("Alice studied physics in college.")
        assert len(mgr.get_story()) == 1

        # Long term facts
        id1 = mgr.add_long_term("Drinks oolong tea every afternoon.")
        assert id1 is not None
        facts = mgr.get_long_term()
        assert len(facts) == 1
        assert "oolong tea" in facts[0]["text"]

        # FTS5 search
        fts_tea = mgr.fts_search("oolong")
        assert len(fts_tea) >= 1
        assert fts_tea[0]["id"] == id1

        fts_file = mgr.fts_search("taxes")
        assert len(fts_file) >= 1

        # Conversations
        mgr.add_turn("user", "Hello assistant", session_id="test_sess")
        mgr.add_turn("assistant", "Hello Alice!", session_id="test_sess")
        history = mgr.get_recent_conversation(limit=5, session_id="test_sess")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello assistant"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Hello Alice!"

        mgr.close()
        print("✓ MemoryManager CRUD & FTS5 verified.")


def test_companion_persistence_across_restart():
    print("Testing CompanionBrain conversation persistence across simulated restart...")
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "iris_mem.db"
        json_file = Path(tmpdir) / "store.json"

        # Session 1: User interacts with Iris
        mem1 = RuleMemory(store_path=str(json_file), db_path=str(db_file))
        brain1 = CompanionBrain()
        brain1.mem = mem1  # point to test db
        brain1.history = brain1.mem.db.get_recent_conversation(limit=12)

        # Set user name
        brain1.mem.set_profile("name", "TestUser")

        # Conversational turn 1 (direct hit)
        ans, _ = brain1.process_utterance("what is my name")
        assert ans is not None
        assert "TestUser" in ans

        # Conversational turn 2 (LLM generation turn)
        brain1.record_turn("I have an interview tomorrow.", "Good luck! You will do great.")

        # Simulate app shutdown and restart: Session 2
        print("Simulating application shutdown and restart...")
        mem2 = RuleMemory(store_path=str(json_file), db_path=str(db_file))
        brain2 = CompanionBrain()
        brain2.mem = mem2
        brain2.history = brain2.mem.db.get_recent_conversation(limit=12)

        # History should survive!
        assert len(brain2.history) == 4, f"Expected 4 turns in history, got {len(brain2.history)}"
        assert brain2.history[0]["role"] == "user"
        assert brain2.history[0]["content"] == "what is my name"
        assert brain2.history[2]["content"] == "I have an interview tomorrow."
        assert brain2.history[3]["content"] == "Good luck! You will do great."

        print("✓ Conversation history successfully survived application restart!")
        mem1.db.close()
        mem2.db.close()


if __name__ == "__main__":
    test_manager_crud()
    test_companion_persistence_across_restart()
    print("\n🎉 ALL SQLITE MEMORY TESTS PASSED!")
