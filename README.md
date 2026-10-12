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
- **[uv](https://docs.astral.sh/uv/)** (Fast Python package and project manager)
- Windows x64 (or Linux/macOS with appropriate binaries)

If `uv` is not yet installed on Windows:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Environment Setup (Sync Dependencies)
Clone the repository and sync dependencies:
```powershell
git clone https://github.com/ha466/iris_core.git
cd iris_core

# Automatically creates .venv and synchronizes all dependencies from uv.lock
uv sync
```

### 3. Automated Model & Binary Setup
Run the automated installer with `uv run` to download and configure `core/bin` server executables (`llama-server.exe` & `parakeet-server.exe`), `core/modal` speech/language models (`Ling-3.0-tiny` & `Parakeet`), and `core/tts` voice weights:

```powershell
# Automatically download and unpack all required models and binaries
uv run setup.py

# Or verify asset status without downloading:
uv run setup.py --check
```

### 4. Running the Companion
Launch the hands-free voice companion:
```powershell
uv run main.py
```

---

## 🧪 Testing & Benchmarks

Run the complete test suite directly with `uv run`:
```powershell
# Unit & integration tests
uv run tests\test_companion_intelligence.py
uv run tests\test_memory_intelligence.py
uv run tests\test_memory_sqlite.py
uv run tests\test_components.py

# End-to-end performance benchmark
uv run benchmark.py
```

---

### 💻 Benchmark System Specifications

All benchmarks were measured on a consumer-grade workstation running 100% locally on CPU with zero cloud calls or discrete GPU acceleration:

| Component | Specification |
|---|---|
| **CPU** | Intel Core i7-7700 @ 3.60GHz (4 Cores, 8 Logical Processors) |
| **RAM** | 16 GB DDR4 (15.9 GB usable) |
| **Operating System** | Microsoft Windows 11 Pro (x64) |
| **Python Runtime** | Python 3.12 (Astral `uv` environment) |
| **Inference Engines** | `llama.cpp` (b10587, AVX2 / OpenMP) + `parakeet.cpp` (v0.5.0) |
| **Hardware Offload** | **100% CPU Inference** (Zero discrete GPU required) |

---

### 📊 Measured Pipeline Performance

#### 1. Kokoro-7M Distilled TTS (CPU Synthesis)
Measured across multiple utterance lengths at 24kHz audio output rate:

| Prompt Length | Characters | Latency | Audio Output | Real-Time Factor (RTF) | Synthesis Speed |
|---|---|---|---|---|---|
| **Short** | 48 chars | 333.9 ms | 3.62 s | **0.092** | **10.9x real-time** |
| **Medium** | 107 chars | 643.7 ms | 6.70 s | **0.096** | **10.4x real-time** |
| **Long** | 167 chars | 966.7 ms | 9.97 s | **0.097** | **10.3x real-time** |
| **Average** | — | — | — | **0.095** | **10.5x real-time** |

> *Note: RTF < 0.10 means Kokoro generates 10 seconds of high-fidelity spoken audio in less than 1 second on a standard 4-core CPU.*

#### 2. End-to-End Turn Latencies (`benchmark.py`)
Measured with synthetic 16kHz speech input through Parakeet STT, SQLite memory routing, Ling-3.0-tiny LLM streaming, sentence chunking, and concurrent Kokoro synthesis:

| Interaction Type | Utterance Sample | STT Latency | Memory Routing | TTFA (Time-To-First-Audio) | Total Turn Duration |
|---|---|---|---|---|---|
| **Direct Memory Read** | *"what is my name"* | 183.6 ms | **6.7 ms** | **457.4 ms** (Instant) | 457.4 ms |
| **Explicit Memory Write** | *"please remember that I love espresso"* | 219.9 ms | **11.4 ms** | **625.4 ms** (Instant) | 625.4 ms |
| **Personal Companion Chat** | *"how are you doing today, Iris?"* | 176.0 ms | 3.7 ms | ~7.7 s | 8.9 s |
| **Creative / Knowledge Query** | *"tell me one fun fact about space"* | 197.6 ms | 0.9 ms | ~8.7 s | 11.1 s |

#### 3. Memory & Storage Footprint
- **Memory Hot Path**: Instant regex rules & direct cache lookups execute in **< 1 ms**.
- **FTS5 & Hybrid Search**: SQLite full-text search combined with vector similarity completes in **< 10 ms**.
- **On-Disk Database**: SQLite database file with WAL logging requires **< 10 MB** for thousands of turns and persistent facts.
- **RAM Footprint**: ~5.8 GB total system RAM during full pipeline execution (including `llama-server`, `parakeet-server`, in-memory VAD, and TTS).
