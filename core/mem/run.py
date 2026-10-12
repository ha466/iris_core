#!/usr/bin/env python3
"""
run.py - demo, benchmark and integration example for memory.py

    python run.py --seed            # write sample data, then open the interactive prompt
    python run.py --bench           # run the built-in test utterances (pass/fail + latency)
    python run.py                   # interactive prompt (type what the STT would give you)
    python run.py --no-embed        # force keyword matching (no model download)

Interactive commands (everything else is treated as a user utterance):
    /set name Hari                  /friend Alex | best friend | plays guitar
    /file resume | ~/Docs/cv.pdf | latest CV      /story <text>
    /remember <text>                /stats        /quit
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Callable, Optional

try:
    from core.mem.memory import Embedder, RuleMemory
except ImportError:
    from memory import Embedder, RuleMemory

BASE_SYSTEM = "You are a friendly voice assistant. Keep answers short and spoken-style."


# --------------------------------------------------------------------------- #
# INTEGRATION EXAMPLE - this is the part you copy into your existing pipeline
# --------------------------------------------------------------------------- #
def respond(
    user_text: str,
    mem: RuleMemory,
    llm: Optional[Callable[[str, str], str]] = None,   # llm(system_prompt, user_text) -> reply
    tts: Callable[[str], None] = print,                # tts(text) -> plays audio
) -> str:
    """STT text in -> (memory rules) -> LLM / direct answer -> TTS."""
    hit = mem.retrieve(user_text)                      # None for ~everything that isn't a memory question

    if hit and hit.direct:                             # structured fact: answer instantly, skip the LLM
        reply = hit.answer
    else:
        system = BASE_SYSTEM + (("\n\n" + hit.context) if hit else "")
        reply = llm(system, user_text) if llm else "(your LLM answers here)"

    tts(reply)
    return reply


# --------------------------------------------------------------------------- #
# Sample data (replace with the real user's data)
# --------------------------------------------------------------------------- #
def seed(mem: RuleMemory) -> None:
    mem.set_profile("name", "Hari")
    mem.add_friend("Alex", "friend", "Met in college. Plays guitar.")
    mem.add_friend("Sam", "best friend", "Lives nearby and loves cricket.")
    mem.add_friend("Priya", "sister", "Younger sister.")
    mem.add_file("resume", "~/Documents/resume.pdf", "Latest CV")
    mem.add_file("invoice template", "~/Documents/invoice_template.xlsx", "Template for lab invoices")
    mem.add_file("project notes", "~/Documents/voice_assistant_notes.md", "Notes on the voice assistant project")
    mem.add_story(
        "Hari grew up taking apart old radios and got into computers as a teenager. "
        "He taught himself programming from online tutorials.\n\n"
        "Later he started building small local AI tools, and now works on private voice assistants "
        "that run entirely on his own hardware."
    )
    mem.add_long_term("Likes strong black coffee in the morning.")
    mem.add_long_term("Prefers short spoken answers over long explanations.")
    mem.add_long_term("Is learning to play the guitar.")
    mem.add_long_term("Wants to finish the voice assistant project by December.")


# --------------------------------------------------------------------------- #
# Benchmark / tests: (utterance, expected rule or None)
# --------------------------------------------------------------------------- #
CASES = [
    ("what is my name", "identity.name"),
    ("Uh, who is my, who is my, what is my name", "identity.name"),
    ("what's my name please", "identity.name"),
    ("who am I", "identity.name"),
    ("do you remember my name", "identity.name"),
    ("who is Hari", "entity"),
    ("who is Alex", "entity"),
    ("how many friends do I have", "friends.count"),
    ("who are my friends", "friends.list"),
    ("who is my best friend", "friends.relation"),
    ("what is my sister's name", "friends.relation"),
    ("how many files do I have", "files.count"),
    ("what files do I have", "files.list"),
    ("where is my resume file", "files.find"),
    ("where did I save the invoice template", "files.find_loose"),
    ("what do you know about me", "about_me"),
    ("tell me my story", "story.full"),
    ("how did I get started with computers", "story.search"),
    ("do you remember what I told you about coffee", "ltm.search"),
    ("what do I like to drink", "ltm.search"),
    ("what is my favorite color", "ltm.search"),
    ("what did we talk about last time", "ltm.recent"),
    # --- must NOT touch memory ---
    ("who is Elon Musk", None),
    ("who is my favorite singer", None),
    ("what is the capital of France", None),
    ("what time is it", None),
    ("play some jazz music", None),
    ("tell me a joke", None),
    ("what is my name in Spanish", None),
    ("remind me to call mom tomorrow", None),
    ("where did I put my keys", None),
    ("do you know what the weather is", None),
]


def bench(mem: RuleMemory) -> int:
    ok = 0
    print(f"{'utterance':<52} {'rule':<18} {'ms':>6}  result")
    print("-" * 92)
    for text, expected in CASES:
        t0 = time.perf_counter()
        hit = mem.retrieve(text)
        ms = (time.perf_counter() - t0) * 1000
        got = hit.rule if hit else None
        good = got == expected
        ok += good
        print(f"{text[:50]:<52} {str(got):<18} {ms:6.2f}  {'PASS' if good else 'FAIL (expected ' + str(expected) + ')'}")
    print("-" * 92)
    print(f"{ok}/{len(CASES)} passed")
    return 0 if ok == len(CASES) else 1


# --------------------------------------------------------------------------- #
# Interactive prompt
# --------------------------------------------------------------------------- #
def command(mem: RuleMemory, line: str) -> bool:
    cmd, _, rest = line[1:].partition(" ")
    parts = [p.strip() for p in rest.split("|")]
    if cmd == "set":
        k, _, v = rest.partition(" ")
        mem.set_profile(k.strip(), v.strip())
    elif cmd == "friend":
        mem.add_friend(parts[0], *(parts[1:3]))
    elif cmd == "file":
        mem.add_file(parts[0], *(parts[1:3]))
    elif cmd == "story":
        print(f"saved {mem.add_story(rest)} chunk(s)")
    elif cmd == "remember":
        mem.add_long_term(rest)
    elif cmd == "stats":
        print(mem.stats())
    elif cmd in ("quit", "exit"):
        return False
    else:
        print("unknown command")
    return True


def repl(mem: RuleMemory) -> None:
    print("Type an utterance (as the STT would transcribe it). /stats, /quit for commands.\n")
    while True:
        try:
            line = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line.startswith("/"):
            try:
                if not command(mem, line):
                    break
            except Exception as e:  # noqa: BLE001
                print("error:", e)
            continue
        hit = mem.retrieve(line)
        if hit is None:
            print("  [no memory rule matched -> memory NOT used]")
        else:
            mode = "DIRECT (skip LLM)" if hit.direct else "inject into LLM prompt"
            print(f"  [{hit.rule} | {mode} | found={hit.found} | {hit.latency_ms:.2f} ms"
                  f"{' | embeddings' if hit.used_embeddings else ''}]")
            if hit.answer:
                print(f"  answer : {hit.answer}")
            print("  context:\n" + "\n".join("    " + l for l in hit.context.splitlines()))
        print("  assistant> ", end="")
        respond(line, mem)
        print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default="memory_store.json")
    ap.add_argument("--seed", action="store_true", help="fill the store with sample data")
    ap.add_argument("--bench", action="store_true", help="run test utterances and exit")
    ap.add_argument("--no-embed", action="store_true", help="skip the embedding model (keyword matching only)")
    ap.add_argument("--backend", default="auto", choices=["auto", "fastembed", "sentence-transformers"])
    ap.add_argument("--assistant-name", action="append", default=[], help="wake word / assistant name (allowed at end of a query)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    embedder = Embedder("none" if args.no_embed else args.backend)
    mem = RuleMemory(args.store, embedder=embedder, assistant_names=tuple(args.assistant_name))
    if args.seed:
        seed(mem)
        print(f"seeded {args.store}")

    if not args.no_embed:
        t0 = time.perf_counter()
        mem.warmup(background=False)   # in your assistant: mem.warmup() runs in a background thread
        print(f"embedding model: {mem.embedder.tag} (loaded in {(time.perf_counter() - t0) * 1000:.0f} ms)")

    if args.bench:
        return bench(mem)
    repl(mem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
