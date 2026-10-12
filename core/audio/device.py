"""
device.py - Audio input (microphone) and output (speaker) devices with barge-in support.
"""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

import config

log = logging.getLogger("audio_device")


class AudioDeviceManager:
    """
    Manages non-blocking microphone capture (16kHz) and speaker playback (24kHz).
    Provides instant cancellation for barge-in / speech interruptions.
    """

    def __init__(self):
        self.input_stream: Optional[sd.InputStream] = None
        self.output_stream: Optional[sd.OutputStream] = None

        # Thread-safe queues
        self.mic_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self.spk_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue()

        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._is_playing = False
        self._interrupt_event = threading.Event()

    def start_input(self, loop: asyncio.AbstractEventLoop) -> None:
        """Start non-blocking microphone stream."""
        self._loop = loop

        def mic_callback(indata, frames, time_info, status):
            if status:
                log.debug(f"Input stream status: {status}")
            # indata is shape (512, 1) float32 -> flatten to (512,)
            chunk = indata[:, 0].copy()
            if self._loop and not self._loop.is_closed():
                self._loop.call_soon_threadsafe(self.mic_queue.put_nowait, chunk)

        dev_idx = config.MIC_DEVICE_INDEX if config.MIC_DEVICE_INDEX is not None else sd.default.device[0]
        try:
            dev_info = sd.query_devices(dev_idx)
            self.device_name = dev_info["name"]
        except Exception:
            self.device_name = f"Device #{dev_idx}"

        self.input_stream = sd.InputStream(
            device=config.MIC_DEVICE_INDEX,
            samplerate=config.MIC_SAMPLE_RATE,
            channels=config.MIC_CHANNELS,
            dtype="float32",
            blocksize=config.MIC_CHUNK_SIZE,
            callback=mic_callback,
        )
        self.input_stream.start()
        log.info(f"Microphone stream started on [{self.device_name}] ({config.MIC_SAMPLE_RATE}Hz, blocksize={config.MIC_CHUNK_SIZE}).")


    def start_output(self) -> None:
        """Start non-blocking speaker stream."""
        self._interrupt_event.clear()
        self._current_audio: Optional[np.ndarray] = None
        self._current_idx: int = 0

        def spk_callback(outdata, frames, time_info, status):
            if status:
                log.debug(f"Output stream status: {status}")

            if self._interrupt_event.is_set():
                outdata.fill(0)
                self._is_playing = False
                return

            filled = 0
            while filled < frames:
                if self._interrupt_event.is_set():
                    outdata[filled:].fill(0)
                    self._is_playing = False
                    return

                if self._current_audio is None or self._current_idx >= len(self._current_audio):
                    try:
                        next_chunk = self.spk_queue.get_nowait()
                        if next_chunk is None:
                            # End of current speech turn
                            self._current_audio = None
                            self._is_playing = False
                            break
                        self._current_audio = next_chunk
                        self._current_idx = 0
                        self._is_playing = True
                    except queue.Empty:
                        self._current_audio = None
                        self._is_playing = False
                        break

                remaining_in_chunk = len(self._current_audio) - self._current_idx
                needed = frames - filled
                to_copy = min(remaining_in_chunk, needed)

                outdata[filled:filled + to_copy, 0] = self._current_audio[self._current_idx:self._current_idx + to_copy]
                filled += to_copy
                self._current_idx += to_copy

            if filled < frames:
                outdata[filled:].fill(0)

        self.output_stream = sd.OutputStream(
            samplerate=config.SPK_SAMPLE_RATE,
            channels=config.SPK_CHANNELS,
            dtype="float32",
            blocksize=1024,
            callback=spk_callback,
        )
        self.output_stream.start()
        log.info(f"Speaker stream started ({config.SPK_SAMPLE_RATE}Hz).")

    def play_audio(self, pcm_data: np.ndarray) -> None:
        """Enqueue 24kHz float32 audio for playback."""
        if not self._interrupt_event.is_set():
            self._is_playing = True
            self.spk_queue.put(pcm_data)

    def mark_speech_turn_end(self) -> None:
        """Signal that no more audio chunks are coming for current turn."""
        self.spk_queue.put(None)

    def interrupt(self) -> None:
        """Instantly stop active audio playback and purge pending audio (barge-in)."""
        self._interrupt_event.set()
        # Drain queue
        while not self.spk_queue.empty():
            try:
                self.spk_queue.get_nowait()
            except queue.Empty:
                break
        self._current_audio = None
        self._is_playing = False
        # Reset event for next playback
        self._interrupt_event.clear()

    @property
    def is_playing(self) -> bool:
        """Whether audio is actively being played through speakers."""
        return self._is_playing

    def close(self) -> None:
        """Stop and close streams."""
        if self.input_stream:
            try:
                self.input_stream.stop()
                self.input_stream.close()
            except Exception:
                pass
        if self.output_stream:
            try:
                self.interrupt()
                self.output_stream.stop()
                self.output_stream.close()
            except Exception:
                pass
        log.info("Audio streams closed.")
