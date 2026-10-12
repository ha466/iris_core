"""
test_companion_intelligence.py - Verification for Phase 4 Companion Intelligence & Persona.
Tests emotional tone detection, adaptive response budgeting, and prompt construction.
"""
import os
import sys
import tempfile
from pathlib import Path

# Fix Windows console encoding for UTF-8 test reports
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from core.mem.memory import RuleMemory
from core.pipeline.companion import CompanionBrain


def test_emotional_signal_detection():
    print("Testing emotional signal detection...")
    brain = CompanionBrain

    # Frustrated
    res = brain.detect_emotion("Ugh, my script is broken and nothing works.")
    assert res is not None and res[0] == "frustrated", f"Expected frustrated, got {res}"

    # Stressed
    res = brain.detect_emotion("I have a huge deadline and I feel completely overwhelmed.")
    assert res is not None and res[0] == "stressed", f"Expected stressed, got {res}"

    # Excited
    res = brain.detect_emotion("Guess what! That is awesome news!")
    assert res is not None and res[0] == "excited", f"Expected excited, got {res}"

    # Sad
    res = brain.detect_emotion("I miss my hometown so much, feeling really lonely.")
    assert res is not None and res[0] == "sad", f"Expected sad, got {res}"

    # Curious
    res = brain.detect_emotion("I wonder why the sky appears blue at sunset.")
    assert res is not None and res[0] == "curious", f"Expected curious, got {res}"

    # Neutral
    res = brain.detect_emotion("What time is the meeting?")
    assert res is None, f"Expected None for neutral text, got {res}"

    print("✓ Emotional tone detection verified across all states.")


def test_adaptive_response_budget():
    print("Testing adaptive response budgeting...")
    brain = CompanionBrain

    # Quick greetings / courtesies -> 80 tokens
    assert brain.get_response_budget("hello Iris") == 80
    assert brain.get_response_budget("thanks") == 80
    assert brain.get_response_budget("good morning") == 80

    # Conversational turn -> 160 tokens
    assert brain.get_response_budget("How are things going today?") == 160
    assert brain.get_response_budget("What should we have for dinner?") == 160

    # Deep explanations / stories -> 384 tokens
    assert brain.get_response_budget("Can you explain how neural networks learn?") == 384
    assert brain.get_response_budget("Tell me a bedtime story about the stars.") == 384
    assert brain.get_response_budget("Can you elaborate on your theory in detail?") == 384

    print("✓ Adaptive token budgeting verified.")


def test_companion_prompt_and_emotional_injection():
    print("Testing companion prompt assembly and emotional context injection...")
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "test_intel.db"
        json_file = Path(tmpdir) / "test_intel.json"

        mem = RuleMemory(store_path=str(json_file), db_path=str(db_file))
        brain = CompanionBrain()
        brain.mem = mem

        try:
            mem.set_profile("name", "Hari")
            mem.set_profile("city", "Berlin")

            # Verify base prompt constraints
            assert "spoken voice constraints" in config.BASE_SYSTEM_PROMPT.lower()
            assert "never use markdown" in config.BASE_SYSTEM_PROMPT.lower()
            assert "customer service" in config.BASE_SYSTEM_PROMPT.lower()

            # Process an utterance with emotional weight
            _, msgs = brain.process_utterance("I am so exhausted from work today.")
            assert msgs is not None, "Expected LLM messages payload"
            sys_msg = msgs[0]["content"]

            # Verify identity anchor injection
            assert "Speaking with User: Hari" in sys_msg
            assert "Berlin" in sys_msg

            # Verify emotional cue injection
            assert "USER EMOTIONAL STATE: STRESSED" in sys_msg
            assert "Acknowledge the weight calmly" in sys_msg

            # Process neutral utterance
            _, msgs_neutral = brain.process_utterance("What is the speed of sound?")
            sys_msg_neutral = msgs_neutral[0]["content"]
            assert "USER EMOTIONAL STATE:" not in sys_msg_neutral

            print("✓ Identity anchor and emotional state prompt injection verified.")
        finally:
            mem.db.close()


if __name__ == "__main__":
    test_emotional_signal_detection()
    test_adaptive_response_budget()
    test_companion_prompt_and_emotional_injection()
    print("\n🎉 ALL PHASE 4 COMPANION INTELLIGENCE TESTS PASSED!")
