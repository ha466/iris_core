"""
vad.py - Voice Activity Detection using Pipecat Silero VAD.
"""
from __future__ import annotations

import collections
import logging
from typing import Deque, Optional, Tuple

import numpy as np

import config

log = logging.getLogger("vad")


# --------------------------------------------------------------------------- #
# Pipecat Silero VAD Engine
# --------------------------------------------------------------------------- #
class PipecatVAD:
    """Pipecat's official Silero VAD analyzer."""

    def __init__(self, confidence: float = getattr(config, "VAD_CONFIDENCE_THRESHOLD", 0.45)):
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.vad.vad_analyzer import VADParams

        self.params = VADParams(
            confidence=confidence,
            start_secs=config.VAD_PRE_SPEECH_DURATION,
            stop_secs=config.VAD_SILENCE_DURATION,
            min_volume=0.03,
        )
        self.analyzer = SileroVADAnalyzer(sample_rate=config.MIC_SAMPLE_RATE, params=self.params)
        self.analyzer.set_sample_rate(config.MIC_SAMPLE_RATE)

    def is_speech(self, chunk: np.ndarray) -> float:
        # Remove DC offset
        frame = chunk - np.mean(chunk)
        int16_bytes = np.clip(frame * 32767, -32768, 32767).astype(np.int16).tobytes()
        try:
            conf = self.analyzer.voice_confidence(int16_bytes)
            return float(conf[0] if isinstance(conf, (list, np.ndarray)) else conf)
        except Exception:
            return 0.0

    def reset_states(self) -> None:
        try:
            self.analyzer._model.reset_states()
        except Exception:
            pass


def create_vad(backend: Optional[str] = None) -> PipecatVAD:
    """Instantiate Pipecat Silero VAD."""
    return PipecatVAD()


# --------------------------------------------------------------------------- #
# Speech Segmenter
# --------------------------------------------------------------------------- #
class VADSegmenter:
    """
    Consumes continuous audio chunks and yields complete speech utterances
    using Pipecat VAD.
    """

    def __init__(self, vad: Optional[PipecatVAD] = None):
        self.vad = vad or create_vad()
        self.is_speaking = False
        self.current_utterance: list[np.ndarray] = []

        chunk_sec = config.MIC_CHUNK_SIZE / config.MIC_SAMPLE_RATE
        pre_frames = int(config.VAD_PRE_SPEECH_DURATION / chunk_sec)
        self.pre_speech_buffer: Deque[np.ndarray] = collections.deque(maxlen=max(1, pre_frames))

        self.silence_frames_required = int(config.VAD_SILENCE_DURATION / chunk_sec)
        self.max_frames = int(config.VAD_MAX_SPEECH_DURATION / chunk_sec)
        self.consecutive_silence_count = 0
        self.consecutive_speech_count = 0

    def process_chunk(self, chunk: np.ndarray) -> Tuple[Optional[np.ndarray], bool, float, float]:
        """
        Process a 512-sample (32ms at 16kHz) chunk.
        Returns:
            (utterance_audio_or_None, is_currently_speaking, speech_probability, rms_volume)
        """
        rms = float(np.sqrt(np.mean(chunk**2)))
        score = self.vad.is_speech(chunk)
        completed_utterance: Optional[np.ndarray] = None

        thresh = getattr(config, "VAD_CONFIDENCE_THRESHOLD", 0.45)
        speech_triggered = score >= thresh

        if speech_triggered:
            self.consecutive_speech_count += 1
            self.consecutive_silence_count = 0

            # Require 2 consecutive frames (~64ms) to enter speaking state (filters single-frame clicks)
            if not self.is_speaking and self.consecutive_speech_count >= 2:
                self.is_speaking = True
                self.current_utterance = list(self.pre_speech_buffer)

            if self.is_speaking:
                self.current_utterance.append(chunk)

        elif self.is_speaking:
            self.consecutive_speech_count = 0
            self.current_utterance.append(chunk)
            self.consecutive_silence_count += 1

            # Speech turn ended on silence or max utterance length reached
            if (self.consecutive_silence_count >= self.silence_frames_required) or (len(self.current_utterance) >= self.max_frames):
                self.is_speaking = False
                self.consecutive_silence_count = 0
                self.consecutive_speech_count = 0
                self.vad.reset_states()

                # Keep trailing 150ms comfort buffer
                chunk_sec = config.MIC_CHUNK_SIZE / config.MIC_SAMPLE_RATE
                keep_trailing = int(0.15 / chunk_sec)
                trim_count = max(0, self.silence_frames_required - keep_trailing)
                active_frames = self.current_utterance[:-trim_count] if (trim_count > 0 and len(self.current_utterance) > trim_count) else self.current_utterance

                total_samples = sum(len(f) for f in active_frames)
                min_samples = int(config.VAD_MIN_SPEECH_DURATION * config.MIC_SAMPLE_RATE)

                if total_samples >= min_samples:
                    completed_utterance = np.concatenate(active_frames)
                self.current_utterance = []
        else:
            self.consecutive_speech_count = 0
            self.pre_speech_buffer.append(chunk)

        return completed_utterance, self.is_speaking, float(score), rms

    def reset(self) -> None:
        self.is_speaking = False
        self.current_utterance = []
        self.consecutive_silence_count = 0
        self.consecutive_speech_count = 0
        self.pre_speech_buffer.clear()
        self.vad.reset_states()
