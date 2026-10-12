# iris_core

**iris_core** is an on-device, hands-free personal AI companion voice system running 100% locally on your hardware with sub-second response times, zero cloud dependencies, persistent multi-session memory, and emotional intelligence.

---

## 🌟 Key Features

- **Local MoE Language Model**: Powered by **Ling-3.0-tiny Q4_0** (7.9B total / 1.3B active MoE parameters) via `llama-server` with `--jinja` template parsing and Q8 KV caching. Clean spoken dialogue generation with disabled thinking mode to avoid TTS delays.
- **Sub-Second Voice Pipeline**:
  - **STT**: Fast on-device transcription with **Parakeet 110M GGUF** via `parakeet-server` (~180ms latency).
  - **TTS**: High-fidelity speech synthesis with distilled **Kokoro-7M** (~0.095 RTF / 10.5x real-time on CPU).
  - **VAD**: **Pipecat Silero VAD** (ONNX runtime) with acoustic feedback gating and barge-in support.
- **Persistent SQLite Memory System**:
  - Hybrid retrieval combining **SQLite FTS5 full-text keyword matching** and **vector cosine similarity**.
  - Multi-session conversation history that survives restarts.
  - Episodic memory summaries and soft-delete fact lifecycle.
  - Natural-language memory commands: *"remember that..."*, *"forget that..."*, *"I changed my city to..."*, *"what do you remember about me?"*, *"keep this conversation private"*.
- **Emotional & Companion Intelligence**:
  - Real-time emotional tone detection (frustration, stress, joy, sadness, curiosity) dynamically adjusting conversational warmth.
  - Adaptive spoken token budgeting (80 tokens for greetings, 160 tokens for conversational dialogue, 384 tokens for deep explanations or stories) preventing TTS rambling.
  - Non-blocking background fact extraction.

---

## 📁 Repository Structure

```
iris_core/
├── setup.py                  # Automated downloader for server binaries and models
├── config.py                 # Centralized configuration (ports, models, sampling, prompts)
├── main.py                   # Main entry point with hands-free terminal interface
├── benchmark.py              # End-to-end latency and synthesis benchmark script
├── pyproject.toml            # Python project dependencies
├── core/
│   ├── audio/                # Audio device management & stream handling
│   ├── mem/                  # SQLite schema, FTS5 triggers, MemoryManager & RuleMemory
│   │   ├── schema.sql        # Database schema for memories, FTS, conversations, episodes
│   │   ├── manager.py        # SQLite ACID manager with WAL mode and hybrid search
│   │   ├── memory.py         # Fast rule-based routing & retrieval
│   │   ├── extractor.py      # STT cleaning & fact extraction
│   │   └── memory_store.json # Initial empty store template
│   ├── pipeline/             # Companion brain, LLM streaming, STT client, VAD segmenter
│   │   ├── companion.py      # CompanionBrain with emotional detection and adaptive budgeting
│   │   ├── llm_client.py     # SSE streaming client for llama-server
│   │   ├── stt_client.py     # Parakeet STT HTTP client
│   │   ├── orchestrator.py   # Asynchronous audio orchestrator & TTS worker
│   │   └── chunker.py        # Streaming sentence chunker
│   ├── servers/              # Server process manager (llama-server & parakeet-server)
│   └── tts/                  # Kokoro-7M synthesizer & model loaders
└── tests/                    # Comprehensive unit and integration test suites
    ├── test_companion_intelligence.py
    ├── test_memory_intelligence.py
    ├── test_memory_sqlite.py
    └── test_components.py
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.12 (or virtual environment)
- Windows x64 (or Linux/macOS with appropriate binaries)

### 2. Setup Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r pyproject.toml   # or uv sync
```

### 3. Automated Model & Binary Setup
Run the automated asset installer to download and configure `core/bin` server executables, `core/modal` speech/language models, and `core/tts` voice weights:

```powershell
# Automatically download and unpack all required models and binaries
python setup.py

# Or verify asset status without downloading:
python setup.py --check
```

### 4. Running the Companion
```powershell
python main.py
```

---

## 🧪 Testing & Benchmarks

Run the complete test suite:
```powershell
# Unit & integration tests
python tests\test_companion_intelligence.py
python tests\test_memory_intelligence.py
python tests\test_memory_sqlite.py
python tests\test_components.py

# End-to-end performance benchmark
python benchmark.py
```

### Benchmark Highlights (Local CPU)
| Pipeline Stage | Metric | Measured Result |
|---|---|---|
| **Kokoro-7M TTS** | Real-Time Factor (RTF) | **0.095** (10.5x real-time) |
| **Direct Memory Retrieval** | Time-To-First-Audio (TTFA) | **~457 ms** |
| **Memory Write (Explicit)** | Time-To-First-Audio (TTFA) | **~625 ms** |
| **Parakeet STT** | Latency | **~180 ms** |
