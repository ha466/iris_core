"""
config.py - Centralized configuration for Iris Companion Voice Assistant
"""
from pathlib import Path
from dataclasses import dataclass

# Root directory
ROOT_DIR = Path(__file__).resolve().parent
CORE_DIR = ROOT_DIR / "core"
BIN_DIR = CORE_DIR / "bin"
MODELS_DIR = CORE_DIR / "modal"
TTS_DIR = CORE_DIR / "tts"
MEM_DIR = CORE_DIR / "mem"

# Server Binaries & Models
LLAMA_SERVER_EXE = BIN_DIR / "llama-b10587-bin-win-cpu-x64" / "llama-server.exe"
LLAMA_MODEL_PATH = MODELS_DIR / "Ling-3.0-tiny-Q4_0.gguf"
LLAMA_MODEL_NAME = "Ling-3.0-tiny"
LLAMA_HOST = "127.0.0.1"
LLAMA_PORT = 8081
LLAMA_BASE_URL = f"http://{LLAMA_HOST}:{LLAMA_PORT}/v1"
LLAMA_THREADS = 4
LLAMA_CTX_SIZE = 4096
LLAMA_TEMPERATURE = 1.0
LLAMA_TOP_P = 0.95
LLAMA_TOP_K = 20
LLAMA_MAX_TOKENS = 512
LLAMA_KV_CACHE_TYPE = "q8_0"
LLAMA_THINKING_ENABLED = False

PARAKEET_SERVER_EXE = BIN_DIR / "parakeet-v0.5.0-bin-win-cpu-x64" / "parakeet-server.exe"
PARAKEET_MODEL_PATH = MODELS_DIR / "tdt_ctc-110m-f16.gguf"
PARAKEET_HOST = "127.0.0.1"
PARAKEET_PORT = 8082
PARAKEET_BASE_URL = f"http://{PARAKEET_HOST}:{PARAKEET_PORT}/v1"
PARAKEET_THREADS = 4

# Audio Settings
MIC_DEVICE_INDEX = None           # None for system default, or integer ID from sounddevice
MIC_SAMPLE_RATE = 16000           # 16 kHz for VAD & Parakeet
MIC_CHANNELS = 1                  # Mono
MIC_CHUNK_SIZE = 512              # 32ms at 16kHz (Pipecat / Silero standard)
SPK_SAMPLE_RATE = 24000           # Kokoro-7M output rate
SPK_CHANNELS = 1

# Voice Activity Detection (Pipecat Silero VAD)
VAD_BACKEND = "pipecat"
VAD_CONFIDENCE_THRESHOLD = 0.45   # Speech trigger threshold (0.45-0.55 for Silero)
VAD_SILENCE_THRESHOLD = 0.20      # Silence threshold
VAD_SILENCE_DURATION = 0.55       # Seconds of silence to trigger end-of-turn
VAD_PRE_SPEECH_DURATION = 0.30    # Seconds of audio to keep before speech starts
VAD_MIN_SPEECH_DURATION = 0.25    # Discard bursts shorter than 250ms
VAD_MAX_SPEECH_DURATION = 15.0    # Maximum utterance duration in seconds

# Acoustic Feedback & Interruption Settings
ENABLE_BARGE_IN = False           # Set False when using open speakers to prevent feedback/echo loops;
                                  # Set True if using headphones.




# Text-To-Speech (Kokoro-7M-Distill) Settings
TTS_WEIGHTS = TTS_DIR / "kokoro_en_7m.pth"
TTS_CONFIG = TTS_DIR / "config.json"
TTS_VOICE = TTS_DIR / "af_msa.pt"
TTS_CHUNK_MIN_CHARS = 18          # Minimum chars before splitting on punctuation

# Memory Settings
MEMORY_STORE_PATH = MEM_DIR / "memory_store.json"
MEMORY_DB_PATH = MEM_DIR / "iris_memory.db"

# Companion Personality & Prompt
COMPANION_NAME = "Iris"
BASE_SYSTEM_PROMPT = (
    "You are Iris, a close personal companion and intelligent friend who genuinely enjoys conversation.\n\n"
    "PERSONALITY & TONE:\n"
    "- Warm, curious, perceptive, and candid — like a smart, trusted friend.\n"
    "- Honest about what you know and do not know. Never fake experiences or feelings you do not possess.\n"
    "- Grounded and supportive. You can be playful when relaxed and serious when needed.\n\n"
    "SPOKEN VOICE CONSTRAINTS (CRITICAL FOR TTS AUDIO):\n"
    "- Speak in natural, spoken English using contractions and conversational flow.\n"
    "- Default to 1-3 concise sentences. Only expand when asked for a story, deep explanation, or detail.\n"
    "- NEVER use markdown headers, bullet points, asterisks, emoji, action tags like *smiles*, or code blocks.\n"
    "- NEVER sound like corporate customer service. Never ask 'How can I assist you today?'.\n\n"
    "MEMORY & EMOTIONAL CONTINUITY:\n"
    "- When you know details about the user, weave them in naturally without reciting facts like a robot.\n"
    "- Adapt your tone empathetically to the user's emotional state without clinical diagnosing."
)
