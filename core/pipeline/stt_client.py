"""
stt_client.py - Async client for parakeet-server OpenAI transcription endpoint.
"""
from __future__ import annotations

import io
import logging
import wave
from typing import Optional

import httpx
import numpy as np

import config

log = logging.getLogger("stt_client")


def pcm_to_wav_bytes(pcm_float32: np.ndarray, sample_rate: int = config.MIC_SAMPLE_RATE) -> bytes:
    """Convert float32 [-1.0, 1.0] 16kHz audio array to in-memory WAV bytes."""
    # Normalize peak volume so Parakeet STT receives optimal clear amplitude
    peak = float(np.max(np.abs(pcm_float32))) if len(pcm_float32) > 0 else 0.0
    if peak > 0.005:
        pcm_float32 = (pcm_float32 / peak) * 0.90

    pcm_int16 = np.clip(pcm_float32 * 32767, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()



class ParakeetSTTClient:
    """Sends audio buffers to parakeet-server for fast speech recognition."""

    def __init__(self, base_url: str = config.PARAKEET_BASE_URL):
        self.endpoint = f"{base_url}/audio/transcriptions"
        self.client = httpx.AsyncClient(timeout=10.0)

    async def transcribe(self, pcm_audio: np.ndarray) -> str:
        """Transcribe in-memory PCM audio."""
        wav_bytes = pcm_to_wav_bytes(pcm_audio)
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {"model": "parakeet", "response_format": "json"}

        try:
            r = await self.client.post(self.endpoint, files=files, data=data)
            if r.status_code == 200:
                result = r.json()
                text = result.get("text", "").strip()
                return text
            else:
                log.error(f"STT error {r.status_code}: {r.text}")
                return ""
        except Exception as e:
            log.error(f"Failed to communicate with parakeet-server: {e}")
            return ""

    async def close(self) -> None:
        await self.client.aclose()
