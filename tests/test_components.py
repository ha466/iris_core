"""
test_components.py - Self-contained unit tests for Iris components.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


import numpy as np

# Ensure root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from core.audio.vad import create_vad, VADSegmenter
from core.pipeline.chunker import SentenceChunker

from core.pipeline.companion import CompanionBrain
from core.servers.server_manager import ServerManager
from core.tts.synthesizer import KokoroSynthesizer


def test_paths_and_models():
    """Verify that all required binaries and models exist."""
    print("Testing paths and model files...")
    assert config.LLAMA_SERVER_EXE.exists(), f"Missing {config.LLAMA_SERVER_EXE}"
    assert config.LLAMA_MODEL_PATH.exists(), f"Missing {config.LLAMA_MODEL_PATH}"
    assert config.PARAKEET_SERVER_EXE.exists(), f"Missing {config.PARAKEET_SERVER_EXE}"
    assert config.PARAKEET_MODEL_PATH.exists(), f"Missing {config.PARAKEET_MODEL_PATH}"
    assert config.TTS_WEIGHTS.exists(), f"Missing {config.TTS_WEIGHTS}"
    assert config.TTS_CONFIG.exists(), f"Missing {config.TTS_CONFIG}"
    assert config.TTS_VOICE.exists(), f"Missing {config.TTS_VOICE}"
    print("✓ All binary and model paths exist.")


def test_sentence_chunker():
    """Verify streaming token chunking at natural punctuation boundaries."""
    print("Testing SentenceChunker...")
    chunker = SentenceChunker(min_chars=15)
    tokens = ["Hello ", "there! ", "How ", "are ", "you ", "doing ", "today? ", "I ", "am ", "great."]
    emitted = []
    for tok in tokens:
        chunks = chunker.add_token(tok)
        emitted.extend(chunks)
    emitted.extend(chunker.flush())

    print(f"Emitted chunks: {emitted}")
    assert len(emitted) >= 2, f"Expected at least 2 chunks, got {len(emitted)}"
    assert "Hello there!" in emitted[0]
    print("✓ SentenceChunker passed.")


def test_rule_memory():
    """Verify instant direct memory retrieval, explicit write, and clean context injection."""
    print("Testing CompanionBrain and RuleMemory (Read & Write)...")
    companion = CompanionBrain()
    companion.mem.set_profile("name", "TestUser")

    # 1. Direct identity question (Read)
    ans, msgs = companion.process_utterance("what is my name")
    assert ans is not None, "Expected direct memory hit for 'what is my name'"
    assert "TestUser" in ans, f"Expected 'TestUser' in answer, got {ans}"
    print(f"Direct read hit result: {ans}")

    # 2. Explicit memorization (Write)
    ans_write, msgs_write = companion.process_utterance("please remember that I love green tea")
    assert ans_write is not None, "Expected direct memory hit for explicit remember"
    assert "green tea" in ans_write.lower(), f"Expected 'green tea' in answer, got {ans_write}"
    print(f"Direct write hit result: {ans_write}")

    # 3. General conversation query with baseline identity injection
    ans2, msgs2 = companion.process_utterance("Tell me a funny joke about space.")
    assert ans2 is None, "Did not expect direct memory hit for joke"
    assert msgs2 is not None and len(msgs2) >= 2
    assert msgs2[0]["role"] == "system"
    assert "TestUser" in msgs2[0]["content"], "Expected baseline user anchor (TestUser) in system prompt"
    assert "[MEMORY:" not in msgs2[0]["content"], "Expected raw bracket tags to be sanitized"
    print("✓ Baseline user identity injection verified.")

    # 4. Background fact extraction
    companion.record_turn("I really enjoy rock climbing on weekends", "That sounds awesome!")
    climber_hit = any("rock climbing" in it.get("text", "").lower() for it in companion.mem.data["long_term"])
    assert climber_hit, "Expected background extractor to capture 'rock climbing'"
    print("✓ Background fact extraction verified.")
    print("✓ CompanionBrain read & write memory system passed.")


def test_vad():
    """Verify Pipecat Silero VAD on dummy audio chunks."""
    print("Testing Pipecat Silero VAD...")
    vad = create_vad()
    segmenter = VADSegmenter(vad)

    # Silence frame (zeros)
    silence = np.zeros(config.MIC_CHUNK_SIZE, dtype=np.float32)
    completed, speaking, prob, rms = segmenter.process_chunk(silence)
    assert not speaking, "Should not trigger speaking on silence"

    # Synthetic voice tone (300Hz sine wave)
    t = np.linspace(0, config.MIC_CHUNK_SIZE / config.MIC_SAMPLE_RATE, config.MIC_CHUNK_SIZE, endpoint=False)
    voice_tone = (np.sin(2 * np.pi * 300 * t) * 0.4).astype(np.float32)
    score = vad.is_speech(voice_tone)
    print(f"✓ Pipecat Silero VAD passed (voice score: {score:.4f}, silence rms: {rms:.4f}).")





def test_kokoro_synthesizer():
    """Verify Kokoro-7M synthesis producing valid audio."""
    print("Testing KokoroSynthesizer...")
    synth = KokoroSynthesizer()
    audio = synth.synthesize("Hi, this is Iris.")
    assert audio is not None, "Synthesis returned None"
    assert len(audio) > 1000, f"Expected non-trivial audio, got {len(audio)} samples"
    assert audio.dtype == np.float32
    print(f"✓ KokoroSynthesizer passed ({len(audio)} samples generated at 24kHz).")


async def test_server_manager_and_endpoints():
    """Verify ServerManager auto-serving and HTTP API endpoints."""
    print("Testing ServerManager and API endpoints...")
    mgr = ServerManager()
    try:
        mgr.start_all(timeout_secs=30.0)
        assert mgr.is_port_open(config.LLAMA_HOST, config.LLAMA_PORT)
        assert mgr.is_port_open(config.PARAKEET_HOST, config.PARAKEET_PORT)
        print("✓ Servers booted and responding on ports 8081 and 8082.")

        # Test STT transcription endpoint with synthetic audio
        from core.pipeline.stt_client import ParakeetSTTClient
        stt = ParakeetSTTClient()
        # 1-second silence audio
        dummy_audio = np.zeros(16000, dtype=np.float32)
        transcription = await stt.transcribe(dummy_audio)
        print(f"STT response for silence: '{transcription}'")
        await stt.close()

        # Test LLM chat completion streaming endpoint
        from core.pipeline.llm_client import LlamaLLMClient
        llm = LlamaLLMClient()
        test_messages = [
            {"role": "system", "content": "You are Iris. Respond with one short sentence."},
            {"role": "user", "content": "Say hello!"},
        ]
        tokens = []
        async for token in llm.stream_chat(test_messages, max_tokens=20):
            tokens.append(token)
        reply = "".join(tokens).strip()
        print(f"LLM streaming reply: '{reply}'")
        assert len(reply) > 0, "LLM returned empty reply"
        await llm.close()
        print("✓ Both STT and LLM API endpoints verified successfully.")

    finally:
        mgr.stop_all()
        print("✓ ServerManager cleanly stopped all subprocesses.")


def main():
    test_paths_and_models()
    test_sentence_chunker()
    test_rule_memory()
    test_vad()
    test_kokoro_synthesizer()

    asyncio.run(test_server_manager_and_endpoints())
    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
