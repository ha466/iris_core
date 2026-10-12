"""
orchestrator.py - End-to-end streaming async orchestrator for Iris companion.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Optional

import numpy as np

import config
from core.audio.device import AudioDeviceManager
from core.audio.vad import create_vad, VADSegmenter
from core.pipeline.chunker import SentenceChunker

from core.pipeline.companion import CompanionBrain
from core.pipeline.llm_client import LlamaLLMClient
from core.pipeline.stt_client import ParakeetSTTClient
from core.servers.server_manager import ServerManager
from core.tts.synthesizer import KokoroSynthesizer

log = logging.getLogger("orchestrator")


class VoiceOrchestrator:
    """
    Coordinates all audio, inference, memory, and synthesis components.
    Provides hands-free conversation with barge-in interruption.
    """

    def __init__(
        self,
        on_state_change: Optional[Callable[[str, Optional[str]], None]] = None,
        on_token: Optional[Callable[[str], None]] = None,
        on_audio_meter: Optional[Callable[[float, float, bool], None]] = None,
    ):
        self.on_state_change = on_state_change or (lambda state, detail: None)
        self.on_token = on_token or (lambda token: None)
        self.on_audio_meter = on_audio_meter or (lambda rms, prob, is_spk: None)

        # Components
        self.server_mgr = ServerManager()
        self.audio_mgr = AudioDeviceManager()
        self.vad = create_vad()
        self.segmenter = VADSegmenter(self.vad)

        self.stt_client = ParakeetSTTClient()
        self.llm_client = LlamaLLMClient()
        self.companion = CompanionBrain()
        self.tts = KokoroSynthesizer()

        self._running = False
        self._current_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Start servers and verify model readiness."""
        self.on_state_change("starting_servers", "Starting llama-server and parakeet-server...")
        self.server_mgr.start_all(timeout_secs=30.0)
        self.on_state_change("ready", "All systems online and ready.")

    async def run(self) -> None:
        """Run hands-free conversational loop."""
        loop = asyncio.get_running_loop()
        self.audio_mgr.start_input(loop)
        self.audio_mgr.start_output()

        self._running = True
        self.on_state_change("mic_device", getattr(self.audio_mgr, "device_name", "Default Microphone"))
        self.on_state_change("listening", None)
        log.info("Iris voice loop active. Listening...")

        last_playback_time = 0.0

        while self._running:
            try:
                # Read mic chunk (512 samples float32)
                chunk = await self.audio_mgr.mic_queue.get()

                # If companion is speaking through speakers
                if self.audio_mgr.is_playing:
                    last_playback_time = time.perf_counter()
                    if config.ENABLE_BARGE_IN and self.segmenter.is_speaking:
                        log.info("Barge-in detected! Interrupting companion speech.")
                        self.audio_mgr.interrupt()
                        if self._current_task and not self._current_task.done():
                            self._current_task.cancel()
                        self.on_state_change("interrupted", "User interrupted.")
                    else:
                        # Echo-Safe: Ignore microphone while companion is speaking
                        self.segmenter.reset()
                        continue

                # Echo Reverb Guard: drain room acoustics for 350ms after playback stops
                if time.perf_counter() - last_playback_time < 0.35:
                    self.segmenter.reset()
                    continue

                # Ignore mic while previous turn is thinking or generating
                if self._current_task and not self._current_task.done():
                    self.segmenter.reset()
                    continue

                # Process through VAD
                utterance, is_speaking, prob, rms = self.segmenter.process_chunk(chunk)

                # Send live level to UI meter
                self.on_audio_meter(rms, prob, is_speaking)

                if is_speaking:
                    self.on_state_change("speech_detected", None)

                if utterance is not None:
                    # Utterance complete -> process turn in background task
                    self._current_task = asyncio.create_task(self._process_turn(utterance))

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in main audio loop: {e}", exc_info=True)
                await asyncio.sleep(0.1)

    async def _process_turn(self, pcm_audio: np.ndarray) -> None:
        """Transcribe, consult memory, stream LLM, and synthesize speech."""
        t_start = time.perf_counter()
        try:
            # 1. Speech-to-Text
            dur_s = len(pcm_audio) / config.MIC_SAMPLE_RATE
            self.on_state_change("transcribing", f"{dur_s:.1f}s audio captured")
            t_stt0 = time.perf_counter()
            user_text = await self.stt_client.transcribe(pcm_audio)
            stt_ms = (time.perf_counter() - t_stt0) * 1000

            if not user_text or len(user_text.strip()) == 0:
                self.on_state_change("empty_speech", "Sound detected, but no clear words recognized.")
                await asyncio.sleep(0.8)
                self.on_state_change("listening", None)
                return

            self.on_state_change("heard", f"{user_text} ({stt_ms:.0f}ms)")
            log.info(f"User: '{user_text}' (STT latency: {stt_ms:.1f}ms)")

            # 2. Memory routing
            self.on_state_change("thinking", None)
            direct_answer, messages = self.companion.process_utterance(user_text)

            if direct_answer:
                # Instant response from structured memory facts
                self.on_state_change("memory_hit", direct_answer)
                self.on_state_change("speaking", direct_answer)
                audio = await asyncio.to_thread(self.tts.synthesize, direct_answer)
                if audio is not None:
                    self.audio_mgr.play_audio(audio)
                self.audio_mgr.mark_speech_turn_end()
                while self.audio_mgr.is_playing and self._running:
                    await asyncio.sleep(0.04)
                self.on_state_change("listening", None)
                return

            # 3. Decoupled LLM streaming + concurrent TTS synthesis worker
            chunker = SentenceChunker()
            full_reply = ""
            ttfa_recorded = False

            tts_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

            async def tts_worker():
                while True:
                    chunk = await tts_queue.get()
                    if chunk is None:
                        tts_queue.task_done()
                        break
                    try:
                        self.on_state_change("speaking", chunk)
                        audio = await asyncio.to_thread(self.tts.synthesize, chunk)
                        if audio is not None:
                            self.audio_mgr.play_audio(audio)
                    except Exception as err:
                        log.error(f"TTS synthesis error: {err}")
                    finally:
                        tts_queue.task_done()

            tts_task = asyncio.create_task(tts_worker())

            budget = self.companion.get_response_budget(user_text)
            try:
                async for token in self.llm_client.stream_chat(messages, max_tokens=budget):
                    if not ttfa_recorded:
                        ttfa_ms = (time.perf_counter() - t_start) * 1000
                        ttfa_recorded = True
                        log.info(f"Time to first token: {ttfa_ms:.1f}ms")

                    full_reply += token
                    self.on_token(token)

                    chunks = chunker.add_token(token)
                    for chunk in chunks:
                        await tts_queue.put(chunk)

                # Flush remaining buffer
                for chunk in chunker.flush():
                    await tts_queue.put(chunk)

                # Signal completion to TTS worker
                await tts_queue.put(None)
                await tts_task

            except asyncio.CancelledError:
                tts_task.cancel()
                raise

            self.audio_mgr.mark_speech_turn_end()
            self.companion.record_turn(user_text, full_reply.strip(), async_extract=True)
            log.info(f"Companion: '{full_reply.strip()}'")

            while self.audio_mgr.is_playing and self._running:
                await asyncio.sleep(0.04)

        except asyncio.CancelledError:
            log.info("Turn cancelled by barge-in or shutdown.")
        except Exception as e:
            log.error(f"Error processing turn: {e}", exc_info=True)
        finally:
            self.on_state_change("listening", None)

    async def shutdown(self) -> None:
        """Stop all streams, tasks, and servers."""
        self._running = False
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
        self.audio_mgr.close()
        await self.stt_client.close()
        await self.llm_client.close()
        self.server_mgr.stop_all()
        log.info("VoiceOrchestrator successfully shut down.")
