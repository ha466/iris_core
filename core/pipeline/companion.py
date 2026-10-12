"""
companion.py - Companion persona brain with RuleMemory integration and prompt management.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import config
from core.mem.memory import RuleMemory
from core.mem.run import seed

log = logging.getLogger("companion")

EMOTIONAL_SIGNALS: Dict[str, Tuple[List[str], str]] = {
    "frustrated": (
        ["ugh", "annoying", "broken", "doesn't work", "hate", "tired of", "pissed", "so done", "stupid"],
        "Respond with calm patience, gentle validation, and steady reassurance.",
    ),
    "excited": (
        ["awesome", "amazing", "finally", "can't wait", "guess what", "so happy", "super excited", "celebrate", "great news", "huge news", "omg"],
        "Share their joy and enthusiasm warmly.",
    ),
    "sad": (
        ["miss", "lost", "heartbroken", "lonely", "sad", "depressed", "hurts", "grief", "crying"],
        "Be tender, compassionate, and attentive. Offer comfort without giving unsolicited lectures.",
    ),
    "stressed": (
        ["deadline", "overwhelmed", "too much", "can't keep up", "falling behind", "exhausted", "burnout", "freaking out", "anxious", "so much pressure"],
        "Acknowledge the weight calmly. Keep your response reassuring, steady, and uncluttered.",
    ),
    "curious": (
        ["how come", "what if", "wonder why", "fascinating", "curious about", "philosophical"],
        "Lean into the curiosity thoughtfully and explore the idea with them.",
    ),
}


class CompanionBrain:
    """
    Manages conversational memory, personality, and prompt construction.
    Routes queries to RuleMemory for instant direct answering or contextual injection.
    """

    def __init__(self, memory_path: Optional[str] = None):
        path = memory_path or str(config.MEMORY_STORE_PATH)
        self.mem = RuleMemory(path)

        # Warm up embeddings in the background
        self.mem.warmup(background=True)

        self.max_history_turns = 6
        self.suppress_session = False
        # Load persisted conversation history across restarts
        self.history: List[Dict[str, str]] = self.mem.db.get_recent_conversation(
            limit=self.max_history_turns * 2
        )

    @staticmethod
    def detect_emotion(text: str) -> Optional[Tuple[str, str]]:
        """
        Detects primary emotional tone from user utterance using fast keyword scanning.
        Returns (emotion_name, guidance_hint) or None.
        """
        cleaned = " " + re.sub(r"[^a-zA-Z0-9!']+", " ", text.lower()).strip() + " "
        for emotion, (keywords, hint) in EMOTIONAL_SIGNALS.items():
            for kw in keywords:
                if f" {kw} " in cleaned:
                    return emotion, hint
        return None

    @staticmethod
    def get_response_budget(user_text: str) -> int:
        """
        Dynamically adjusts max_tokens based on intent to optimize latency and spoken naturalness.
        """
        t = user_text.lower().strip()
        # Story or deep explanatory queries: allow larger headroom
        if any(w in t for w in ["explain", "story", "in detail", "elaborate", "deep dive", "how does", "why does"]):
            return 384
        # Quick greetings or confirmations: snappy response
        greetings = [
            "hi", "hello", "hey", "sup", "yo", "morning", "night", "thanks", "thank you",
            "bye", "good morning", "good evening", "good night", "good afternoon", "ok", "okay",
        ]
        if any(t == g or t.startswith(f"{g} ") or t.startswith(f"{g},") for g in greetings):
            return 80
        # Standard conversational turn: 1-3 spoken sentences
        return 160

    def process_utterance(self, user_text: str) -> Tuple[Optional[str], Optional[List[Dict[str, str]]]]:
        """
        Processes user text through memory rules.
        Returns:
            (direct_answer, None) if structured fact matched directly.
            (None, messages_payload) if LLM generation is required.
        """
        hit = self.mem.retrieve(user_text)

        if hit and hit.direct:
            log.info(f"RuleMemory direct hit: '{user_text}' -> '{hit.answer}'")
            if hit.data and hit.data.get("suppress"):
                self.suppress_session = True

            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": hit.answer})
            if not self.suppress_session:
                self.mem.db.add_turn("user", user_text)
                self.mem.db.add_turn("assistant", hit.answer)
            return hit.answer, None

        # Build system prompt with always-on user identity baseline
        system_content = config.BASE_SYSTEM_PROMPT
        user_anchor = self.mem.get_user_summary()
        if user_anchor:
            system_content += f"\nSpeaking with {user_anchor}."

        # Detect and inject emotional context cue
        emotion_match = self.detect_emotion(user_text)
        if emotion_match:
            emotion, hint = emotion_match
            log.info(f"Emotional cue detected: {emotion}")
            system_content += f"\n\n[USER EMOTIONAL STATE: {emotion.upper()} - {hint}]"

        # Inject memory context if available
        if hit and hit.context:
            formatted_ctx = self.mem.format_context_for_prompt(hit)
            if formatted_ctx:
                log.info(f"RuleMemory context injected for: '{user_text}'")
                system_content += f"\n\n{formatted_ctx}"
        else:
            # Hybrid search fallback (FTS5 + vector similarity)
            hybrid_hits = self.mem.hybrid_search(user_text, top_k=2, min_score=0.40)
            if hybrid_hits:
                log.info(f"Hybrid memory search injected for: '{user_text}' ({len(hybrid_hits)} hits)")
                lines = [f"- {text}" for _, text in hybrid_hits]
                system_content += (
                    "\n\nRelevant user facts:\n"
                    + "\n".join(lines)
                    + "\n(Incorporate naturally in spoken response if relevant; do not quote technical tags)."
                )

        messages = [{"role": "system", "content": system_content}]
        # Add recent conversation history within budget
        messages.extend(self.history[-self.max_history_turns * 2:])
        messages.append({"role": "user", "content": user_text})

        return None, messages

    def record_turn(self, user_text: str, assistant_reply: str, async_extract: bool = False) -> None:
        """Record completed conversational turn and extract background memories."""
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": assistant_reply})

        if not self.suppress_session:
            self.mem.db.add_turn("user", user_text)
            self.mem.db.add_turn("assistant", assistant_reply)
            # Extract declarative user facts in background
            try:
                if async_extract:
                    import threading
                    threading.Thread(
                        target=self.mem.background_extract,
                        args=(user_text, assistant_reply),
                        daemon=True,
                    ).start()
                else:
                    self.mem.background_extract(user_text, assistant_reply)
            except Exception as e:
                log.debug(f"Background extraction error: {e}")

        # Trim in-memory history
        if len(self.history) > self.max_history_turns * 2:
            self.history = self.history[-self.max_history_turns * 2:]
