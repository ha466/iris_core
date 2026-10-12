"""
synthesizer.py - High-performance Kokoro-7M streaming speech synthesizer.
"""
from __future__ import annotations

import logging
import re
import threading
from typing import Generator, List, Optional

import numpy as np
import torch

from .load_model import load

log = logging.getLogger("synthesizer")


class KokoroSynthesizer:
    """Wrapper around distilled Kokoro-7M for fast, on-device speech synthesis."""

    def __init__(self):
        log.info("Loading Kokoro-7M-Distill model and voice pack...")
        self.model, self.pipeline, self.voice = load()
        self._lock = threading.Lock()
        log.info("Kokoro-7M synthesizer ready.")

    @staticmethod
    def clean_text(text: str) -> str:
        """Strip markdown symbols, asterisks, brackets, and emojis for clean speech."""
        # Remove markdown bold/italics
        cleaned = re.sub(r"[*_~`#>]", " ", text)
        # Remove markdown links [text](url) -> text
        cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
        # Remove brackets/parentheses around actions like (laughs) or [sighs]
        cleaned = re.sub(r"\([^)]*\)", " ", cleaned)
        cleaned = re.sub(r"\[[^\]]*\]", " ", cleaned)
        # Normalize whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def synthesize(self, text: str) -> Optional[np.ndarray]:
        """
        Synthesize text into 24kHz float32 audio.
        Returns 1D numpy array or None if text is empty.
        """
        clean = self.clean_text(text)
        if not clean:
            return None

        with self._lock:
            audio_chunks = []
            try:
                for res in self.pipeline(clean, voice=self.voice):
                    # res is (graphemes, phonemes, audio_tensor)
                    audio_tensor = res[2]
                    if audio_tensor is not None:
                        if isinstance(audio_tensor, torch.Tensor):
                            audio_chunks.append(audio_tensor.cpu().numpy())
                        elif isinstance(audio_tensor, np.ndarray):
                            audio_chunks.append(audio_tensor)
            except Exception as e:
                log.error(f"Error synthesizing text '{clean[:30]}...': {e}")
                return None

        if not audio_chunks:
            return None

        return np.concatenate(audio_chunks).astype(np.float32)
