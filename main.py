"""
main.py - Interactive terminal entry point for Iris Voice Companion.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from typing import Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from colorama import Fore, Style, init

import sounddevice as sd

import config
from core.pipeline.orchestrator import VoiceOrchestrator

# Initialize Colorama for Windows terminal ANSI colors
init(autoreset=True)

logging.basicConfig(
    level=logging.WARNING,  # Keep terminal clean; orchestrator prints structured UI
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def get_mic_name() -> str:
    dev = config.MIC_DEVICE_INDEX if config.MIC_DEVICE_INDEX is not None else sd.default.device[0]
    try:
        return sd.query_devices(dev)["name"]
    except Exception:
        return f"Device #{dev}"


def print_banner():
    mic_name = get_mic_name()
    barge_desc = f"{Fore.GREEN}Active (Headphones Recommended){Style.RESET_ALL}" if config.ENABLE_BARGE_IN else f"{Fore.YELLOW}Disabled (Speaker Echo-Safe){Style.RESET_ALL}"
    bullet_barge = f"{Fore.YELLOW}• Speak anytime while Iris is talking to barge-in / interrupt.{Style.RESET_ALL}" if config.ENABLE_BARGE_IN else f"{Fore.YELLOW}• Turn-based with auto reverb guard (safe for open speakers).{Style.RESET_ALL}"
    banner = f"""
{Fore.MAGENTA}======================================================================
              🌸  {Fore.CYAN}{Style.BRIGHT}IRIS: PERSONAL VOICE COMPANION{Fore.MAGENTA}
      {Style.DIM}100% On-Device  •  Hands-Free  •  Low-Latency Streaming{Style.NORMAL}{Fore.MAGENTA}
======================================================================{Style.RESET_ALL}
 {Fore.GREEN}✓ STT:{Style.RESET_ALL} Parakeet-110M GGUF (:8082)
 {Fore.GREEN}✓ LLM:{Style.RESET_ALL} Ling-3.0-tiny Q4_0 MoE (:8081)
 {Fore.GREEN}✓ TTS:{Style.RESET_ALL} Kokoro-7M-Distill (24kHz, af_msa)
 {Fore.GREEN}✓ VAD:{Style.RESET_ALL} {Fore.CYAN}Pipecat Silero VAD (Active){Style.RESET_ALL}
 {Fore.GREEN}✓ MIC:{Style.RESET_ALL} {Fore.YELLOW}{mic_name}{Style.RESET_ALL}
 {Fore.GREEN}✓ BRG:{Style.RESET_ALL} {barge_desc}

 {Fore.GREEN}✓ MEM:{Style.RESET_ALL} RuleMemory Personal Store Enabled
{Fore.MAGENTA}----------------------------------------------------------------------{Style.RESET_ALL}
 {Fore.YELLOW}• Speak naturally into your microphone (hands-free).{Style.RESET_ALL}
 {bullet_barge}
 {Fore.YELLOW}• Press Ctrl+C to shut down cleanly.{Style.RESET_ALL}
{Fore.MAGENTA}======================================================================{Style.RESET_ALL}
"""
    print(banner)


class TerminalUI:
    """Manages clean, colored terminal state presentation with a live audio level meter."""

    def __init__(self):
        self._last_state = ""
        self._in_companion_stream = False
        self._last_meter_time = 0.0

    @staticmethod
    def _vol_bar(rms: float) -> str:
        if rms < 0.008:
            return "    "
        elif rms < 0.025:
            return "▂   "
        elif rms < 0.060:
            return "▂▄  "
        elif rms < 0.120:
            return "▂▄▆ "
        else:
            return "▂▄▆█"

    def on_meter(self, rms: float, prob: float, is_speaking: bool):
        """Update live audio volume and VAD confidence indicator (throttled to ~10Hz)."""
        now = time.time()
        if now - self._last_meter_time < 0.09:
            return
        self._last_meter_time = now

        if self._in_companion_stream or self._last_state in {"transcribing", "thinking"}:
            return

        bar = self._vol_bar(rms)
        if is_speaking:
            sys.stdout.write(f"\r{Fore.YELLOW}{Style.BRIGHT}[🗣️ USER SPEAKING | Vol: {bar} | VAD: {prob:.2f}]{Style.RESET_ALL}   ")
        else:
            sys.stdout.write(f"\r{Fore.CYAN}[🎙️ LISTENING     | Vol: {bar} | VAD: {prob:.2f}]{Style.RESET_ALL}   ")
        sys.stdout.flush()

    def on_state(self, state: str, detail: Optional[str]):
        if state == self._last_state and not detail:
            return
        self._last_state = state

        if state == "listening":
            if self._in_companion_stream:
                print("\n")
                self._in_companion_stream = False
            sys.stdout.write(f"\r{Fore.CYAN}[🎙️ LISTENING     | Vol:      | VAD: 0.00]{Style.RESET_ALL}   ")
            sys.stdout.flush()

        elif state == "speech_detected":
            pass  # Handled smoothly by on_meter

        elif state == "transcribing":
            info = f" ({detail})" if detail else ""
            sys.stdout.write(f"\r{Fore.MAGENTA}[⚡ TRANSCRIBING{info}...]{Style.RESET_ALL}                     \n")
            sys.stdout.flush()

        elif state == "empty_speech":
            sys.stdout.write(f"\r{Fore.YELLOW}[ℹ️ {detail}]{Style.RESET_ALL}                                 \n")
            sys.stdout.flush()

        elif state == "heard":
            print(f"\r{Fore.GREEN}{Style.BRIGHT}User >{Style.RESET_ALL} {detail}")

        elif state == "thinking":
            sys.stdout.write(f"{Fore.BLUE}[🧠 THINKING...]{Style.RESET_ALL} ")
            sys.stdout.flush()

        elif state == "memory_hit":
            print(f"\r{Fore.YELLOW}{Style.BRIGHT}[⚡ MEMORY HIT]{Style.RESET_ALL} Direct answer in ~0.2ms!")

        elif state == "speaking":
            pass

        elif state == "interrupted":
            print(f"\n{Fore.RED}[✋ BARGE-IN]{Style.RESET_ALL} Interrupted by user speech.")
            self._in_companion_stream = False

        elif state in {"starting_servers", "ready", "mic_device"}:
            if detail:
                print(f"{Fore.WHITE}[SYSTEM] {detail}{Style.RESET_ALL}")

    def on_token(self, token: str):
        if not self._in_companion_stream:
            self._in_companion_stream = True
            print(f"\r{Fore.CYAN}{Style.BRIGHT}Iris >{Style.RESET_ALL} ", end="", flush=True)
        sys.stdout.write(token)
        sys.stdout.flush()


async def main():
    print_banner()
    ui = TerminalUI()
    orchestrator = VoiceOrchestrator(
        on_state_change=ui.on_state,
        on_token=ui.on_token,
        on_audio_meter=ui.on_meter,
    )

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def handle_signal():
        print(f"\n{Fore.YELLOW}Shutting down Iris gracefully...{Style.RESET_ALL}")
        stop_event.set()

    if os.name != "nt":
        for s in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(s, handle_signal)

    try:
        await orchestrator.initialize()
        run_task = asyncio.create_task(orchestrator.run())

        while not stop_event.is_set():
            await asyncio.sleep(0.5)

        run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}KeyboardInterrupt received. Shutting down...{Style.RESET_ALL}")
    finally:
        await orchestrator.shutdown()
        print(f"{Fore.GREEN}Iris has stopped. Goodbye!{Style.RESET_ALL}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
