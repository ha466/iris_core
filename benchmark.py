"""
benchmark.py - Benchmark Kokoro-7M TTS and the complete end-to-end pipeline across 3 iterations.
"""
from __future__ import annotations

import asyncio
import io
import os
import sys
import time
import wave
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

import config
from core.pipeline.chunker import SentenceChunker
from core.pipeline.companion import CompanionBrain
from core.pipeline.llm_client import LlamaLLMClient
from core.pipeline.stt_client import ParakeetSTTClient, pcm_to_wav_bytes
from core.servers.server_manager import ServerManager
from core.tts.synthesizer import KokoroSynthesizer


def benchmark_tts():
    """Benchmark Kokoro-7M-Distill across 3 iterations with varying lengths."""
    print("=" * 70)
    print("📊 BENCHMARK PART 1: KOKORO-7M DISTILLED TTS (3 RUNS)")
    print("=" * 70)

    synth = KokoroSynthesizer()

    test_prompts = [
        ("Short", "Hello! I am Iris, your personal voice companion."),
        ("Medium", "It is so great to meet you today. I am running entirely locally on your hardware without any cloud latency."),
        ("Long", "I remember that you enjoy strong black coffee in the morning and love working on private AI tools. Whenever you want to chat or take a break, I am right here with you."),
    ]

    results = []

    # Warmup
    print("Warming up TTS engine...")
    synth.synthesize("Warmup check.")
    print("Warmup complete.\n")

    for i, (label, text) in enumerate(test_prompts, start=1):
        t0 = time.perf_counter()
        audio = synth.synthesize(text)
        dt = time.perf_counter() - t0

        assert audio is not None, f"Synthesis failed for run {i}"
        audio_sec = len(audio) / config.SPK_SAMPLE_RATE
        rtf = dt / audio_sec
        speedup = audio_sec / dt

        results.append({
            "run": i,
            "label": label,
            "chars": len(text),
            "audio_sec": audio_sec,
            "latency_ms": dt * 1000,
            "rtf": rtf,
            "speedup": speedup,
        })

        print(f"Run {i} [{label}] ({len(text)} chars):")
        print(f"  • Prompt: \"{text}\"")
        print(f"  • Latency:       {dt * 1000:.1f} ms")
        print(f"  • Audio Length:  {audio_sec:.2f} s ({len(audio)} samples @ 24kHz)")
        print(f"  • Real-Time Factor (RTF): {rtf:.3f} (Lower is better)")
        print(f"  • Synthesis Speed:        {speedup:.1f}x real-time\n")

        # Save last run as a sample audio file
        if i == len(test_prompts):
            out_path = Path("benchmark_tts_sample.wav")
            int16_pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
            with wave.open(str(out_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(config.SPK_SAMPLE_RATE)
                wf.writeframes(int16_pcm.tobytes())
            print(f"📁 Saved 3rd run audio sample to: {out_path.resolve()}\n")

    avg_rtf = sum(r["rtf"] for r in results) / len(results)
    avg_speedup = sum(r["speedup"] for r in results) / len(results)
    print(f"👉 TTS Summary: Average RTF = {avg_rtf:.3f} | Average Speed = {avg_speedup:.1f}x real-time")
    print("=" * 70)


async def benchmark_complete_pipeline():
    """Benchmark the complete end-to-end loop 3 times."""
    print("\n" + "=" * 70)
    print("📊 BENCHMARK PART 2: COMPLETE END-TO-END PIPELINE (3 RUNS)")
    print("=" * 70)

    server_mgr = ServerManager()
    server_mgr.start_all(timeout_secs=30.0)

    stt = ParakeetSTTClient()
    llm = LlamaLLMClient()
    companion = CompanionBrain()
    synth = KokoroSynthesizer()

    # Pre-generate synthetic audio clips for STT input
    # 1: Direct Memory Query
    # 2: Conversational companion query
    # 3: General knowledge / creative query
    test_utterances = [
        ("Run 1 (Direct Memory Read)", "what is my name"),
        ("Run 2 (Explicit Memory Write)", "Please remember that I love espresso."),
        ("Run 3 (Personal Chat)", "How are you doing today Iris?"),
        ("Run 4 (Creative Query)", "Tell me one fun fact about space."),
    ]

    print("Generating synthetic 16kHz speech clips for STT input...")
    test_wavs = []
    for label, text in test_utterances:
        raw_audio = synth.synthesize(text)
        # Resample from 24kHz to 16kHz for STT testing
        # 16000 / 24000 = 2 / 3
        indices = np.round(np.linspace(0, len(raw_audio) - 1, int(len(raw_audio) * 16000 / 24000))).astype(int)
        pcm_16k = raw_audio[indices]
        test_wavs.append((label, text, pcm_16k))
    print("Synthetic audio ready.\n")

    try:
        pipeline_results = []

        for i, (label, orig_text, pcm_audio) in enumerate(test_wavs, start=1):
            print(f"--- {label} ---")
            t_turn_start = time.perf_counter()

            # 1. STT
            t0 = time.perf_counter()
            transcript = await stt.transcribe(pcm_audio)
            stt_ms = (time.perf_counter() - t0) * 1000
            print(f"  1. Parakeet STT ({stt_ms:.1f}ms): \"{transcript}\"")

            # 2. Memory Routing
            t0 = time.perf_counter()
            direct_ans, messages = companion.process_utterance(transcript if transcript else orig_text)
            mem_ms = (time.perf_counter() - t0) * 1000

            first_audio_ms = None
            total_reply = ""

            if direct_ans:
                print(f"  2. Memory Hit ({mem_ms:.2f}ms): Direct answer -> \"{direct_ans}\"")
                # Direct synthesize
                t_tts0 = time.perf_counter()
                audio = synth.synthesize(direct_ans)
                tts_ms = (time.perf_counter() - t_tts0) * 1000
                first_audio_ms = (time.perf_counter() - t_turn_start) * 1000
                total_turn_ms = first_audio_ms
                total_reply = direct_ans
                print(f"  3. Kokoro TTS ({tts_ms:.1f}ms): Generated {len(audio)/24000:.2f}s of speech")
                print(f"  🚀 Time-To-First-Audio (TTFA): {first_audio_ms:.1f} ms (INSTANT MEMORY)\n")

            else:
                print(f"  2. Context Routing ({mem_ms:.2f}ms): LLM generation required")
                chunker = SentenceChunker()
                t_first_token = None
                chunks_synthesized = 0

                tts_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

                async def bench_tts_worker():
                    nonlocal first_audio_ms, chunks_synthesized
                    while True:
                        c = await tts_queue.get()
                        if c is None:
                            tts_queue.task_done()
                            break
                        chunks_synthesized += 1
                        t_s0 = time.perf_counter()
                        audio = synth.synthesize(c)
                        synth_ms = (time.perf_counter() - t_s0) * 1000
                        if first_audio_ms is None:
                            first_audio_ms = (time.perf_counter() - t_turn_start) * 1000
                            print(f"  3. First Sentence Chunker Emit: \"{c}\" (Synthesized in {synth_ms:.1f}ms)")
                            print(f"  🚀 Time-To-First-Audio (TTFA): {first_audio_ms:.1f} ms")
                        tts_queue.task_done()

                worker_task = asyncio.create_task(bench_tts_worker())

                budget = companion.get_response_budget(transcript if transcript else orig_text)
                async for token in llm.stream_chat(messages, max_tokens=budget):
                    if t_first_token is None:
                        t_first_token = (time.perf_counter() - t_turn_start) * 1000

                    total_reply += token
                    chunks = chunker.add_token(token)
                    for chunk in chunks:
                        await tts_queue.put(chunk)

                for chunk in chunker.flush():
                    await tts_queue.put(chunk)

                await tts_queue.put(None)
                await worker_task

                total_turn_ms = (time.perf_counter() - t_turn_start) * 1000
                print(f"  4. Full Reply: \"{total_reply.strip()}\"")
                print(f"  • Time-To-First-Token (TTFT): {t_first_token:.1f} ms")
                print(f"  • Total Turn Duration:        {total_turn_ms:.1f} ms\n")

            pipeline_results.append({
                "run": i,
                "label": label,
                "stt_ms": stt_ms,
                "ttfa_ms": first_audio_ms or 0,
                "total_ms": total_turn_ms,
            })

        print("=" * 70)
        print("🏆 COMPLETE PIPELINE BENCHMARK SUMMARY (3 RUNS):")
        print("=" * 70)
        for r in pipeline_results:
            print(f"  • {r['label']}: STT = {r['stt_ms']:.0f}ms | TTFA = {r['ttfa_ms']:.0f}ms | Total = {r['total_ms']:.0f}ms")
        avg_ttfa = sum(r["ttfa_ms"] for r in pipeline_results) / len(pipeline_results)
        print(f"\n👉 Average Time-To-First-Audio (TTFA): {avg_ttfa:.1f} ms")
        print("=" * 70)

    finally:
        await stt.close()
        await llm.close()
        server_mgr.stop_all()


async def main():
    benchmark_tts()
    await benchmark_complete_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
