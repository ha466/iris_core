"""
test_memory_intelligence.py - Self-check test suite for Phase 3 Memory Intelligence.
Verifies STT cleaning, natural-language commands (forget, update, recall, suppress),
episodic recall, and hybrid search.
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

from core.mem.extractor import FactExtractor, clean_stt_text
from core.mem.manager import MemoryManager
from core.mem.memory import RuleMemory
from core.pipeline.companion import CompanionBrain


def test_stt_cleaning():
    print("Testing STT artifact cleaning...")
    raw1 = "This i have two pets named Gojo and Limbo"
    cleaned1 = clean_stt_text(raw1)
    assert not cleaned1.lower().startswith("this i"), f"Failed to strip 'This i': {cleaned1}"

    raw2 = "Um uh well like I really love dark roast coffee"
    cleaned2 = clean_stt_text(raw2)
    assert cleaned2.startswith("I really love"), f"Failed to strip stutter/filler: {cleaned2}"

    facts = FactExtractor.extract_from_utterance("Um well I really enjoy rock climbing")
    assert len(facts) == 1
    assert "Prefers rock climbing." in facts[0].text
    print("✓ STT artifact cleaning verified.")


def test_natural_language_memory_commands():
    print("Testing natural-language memory commands (forget, update, recall, suppress)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "test_intel.db"
        json_file = Path(tmpdir) / "test_intel.json"

        mem = RuleMemory(store_path=str(json_file), db_path=str(db_file))
        brain = CompanionBrain()
        brain.mem = mem

        try:
            # 1. Setup profile and facts
            mem.set_profile("name", "Hari")
            mem.set_profile("city", "Tokyo")
            mem.add_long_term("I love dark roast coffee.")
            mem.add_long_term("Plays guitar on weekends.")

            # 2. UPDATE COMMAND: "I changed my city to Berlin"
            ans_up, _ = brain.process_utterance("I changed my city to Berlin")
            assert ans_up is not None
            assert "berlin" in ans_up.lower()
            assert mem.get_profile()["city"].lower() == "berlin"
            assert brain.mem.db.get_profile()["city"].lower() == "berlin"
            print(f"Update response: {ans_up}")

            # 3. RECALL ALL COMMAND: "what do you remember about me?"
            ans_rec, _ = brain.process_utterance("what all do you remember about me")
            assert ans_rec is not None
            assert "Hari" in ans_rec
            assert "Berlin" in ans_rec
            print(f"Recall all response: {ans_rec}")

            # 4. FORGET COMMAND: "forget that I love dark roast coffee"
            ans_fgt, _ = brain.process_utterance("forget dark roast coffee")
            assert ans_fgt is not None
            assert "forgotten" in ans_fgt.lower()
            # Ensure it is soft-deleted
            lt = brain.mem.db.get_long_term()
            assert not any("dark roast" in f["text"].lower() for f in lt), "Fact was not soft-deleted"
            print(f"Forget response: {ans_fgt}")

            # 5. SUPPRESS COMMAND: "keep this conversation private"
            assert not brain.suppress_session
            ans_sup, _ = brain.process_utterance("keep this conversation private")
            assert ans_sup is not None
            assert "off the record" in ans_sup.lower() or "private" in ans_sup.lower()
            assert brain.suppress_session, "Failed to activate session privacy"

            # Under suppress, record_turn should NOT save to DB
            init_conv_count = len(brain.mem.db.get_recent_conversation(limit=50))
            brain.record_turn("This is super confidential password.", "I will not remember it.")
            post_conv_count = len(brain.mem.db.get_recent_conversation(limit=50))
            assert init_conv_count == post_conv_count, "Turn was saved despite suppress_session=True"
            print("✓ Privacy suppression verified.")
        finally:
            mem.db.close()


def test_episodic_recall_and_hybrid():
    print("Testing episodic memory and hybrid search...")
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "test_ep.db"
        json_file = Path(tmpdir) / "test_ep.json"

        mem = RuleMemory(store_path=str(json_file), db_path=str(db_file))
        brain = CompanionBrain()
        brain.mem = mem

        try:
            # Insert an episode summary into SQLite
            mem.db.add_episode(
                summary="Reviewed IRIS companion architecture and designed SQLite memory foundation.",
                topics=["architecture", "sqlite"],
            )

            # Query episodic memory
            ans_ep, _ = brain.process_utterance("what did we talk about last time")
            assert ans_ep is not None
            assert "sqlite memory foundation" in ans_ep.lower()
            print(f"Episodic recall response: {ans_ep}")

            # Add facts and test hybrid search
            mem.add_long_term("Drinks green tea with jasmine in the evening.")
            mem.add_long_term("Prefers running at 6am.")

            hits = mem.hybrid_search("jasmine tea", top_k=2)
            assert len(hits) >= 1
            assert "green tea" in hits[0][1]
            print(f"Hybrid search hit: {hits[0][1]}")
            print("✓ Episodic recall and hybrid search verified.")
        finally:
            mem.db.close()


if __name__ == "__main__":
    test_stt_cleaning()
    test_natural_language_memory_commands()
    test_episodic_recall_and_hybrid()
    print("\n🎉 ALL PHASE 3 MEMORY INTELLIGENCE TESTS PASSED!")
