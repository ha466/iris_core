"""
extractor.py - Autonomous fact extraction and conflict resolution for conversational memory.
Runs asynchronously in the background without blocking the voice response loop.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from .models import FactItem

log = logging.getLogger("memory_extractor")

# Ignore conversational overhead, questions, commands, and short filler
_IGNORE_STARTS = (
    "what", "who", "where", "when", "why", "how", "can you", "could you", "will you",
    "do you", "did you", "are you", "is it", "is there", "tell me", "explain",
    "hello", "hi", "hey", "good morning", "good evening", "good afternoon",
    "bye", "goodbye", "thanks", "thank you", "stop", "pause", "resume", "play",
    "yes", "no", "yeah", "nope", "okay", "ok", "sure", "alright", "maybe",
)

# Declarative patterns to extract facts from first-person conversational statements
_EXTRACTION_RULES = [
    # Preferences & likes
    (
        re.compile(r"\b(?:i(?:'m| am)? (?:really )?(?:like|love|enjoy|prefer|adore))\s+(?P<val>[a-z0-9 ]{3,60})", re.IGNORECASE),
        "Prefers {val}.",
        "preference"
    ),
    # Dislikes & hates
    (
        re.compile(r"\b(?:i (?:really )?(?:dislike|hate|can't stand|cannot stand))\s+(?P<val>[a-z0-9 ]{3,60})", re.IGNORECASE),
        "Dislikes {val}.",
        "preference"
    ),
    # Favorites
    (
        re.compile(r"\bmy fav(?:ou?rite)? (?P<item>[a-z ]{2,25}) is (?P<val>[a-z0-9 ]{2,50})", re.IGNORECASE),
        "Favorite {item} is {val}.",
        "preference"
    ),
    # Learning & hobbies
    (
        re.compile(r"\bi(?:'m| am)? (?:learning|practicing|studying|taking up)(?: to)?\s+(?P<val>[a-z0-9 ]{3,60})", re.IGNORECASE),
        "Is learning {val}.",
        "habit"
    ),
    # Projects & work goals
    (
        re.compile(r"\bi(?:'m| am)? (?:working on|building|developing|creating)\s+(?P<val>[a-z0-9 ]{3,60})", re.IGNORECASE),
        "Is working on {val}.",
        "goal"
    ),
    # Living location
    (
        re.compile(r"\bi (?:live in|stay in|moved to|relocated to)\s+(?P<val>[a-z ]{2,40})", re.IGNORECASE),
        "Lives in {val}.",
        "bio"
    ),
    # Pets & family
    (
        re.compile(r"\bmy (?P<rel>dog|cat|pet|brother|sister|wife|husband|partner)(?:'s)? name is (?P<name>[a-z]+)", re.IGNORECASE),
        "Has a {rel} named {name}.",
        "bio"
    ),
    # Routines & habits
    (
        re.compile(r"\bi usually (?P<val>[a-z0-9 ]{3,50}) in the (?:morning|evening|night|afternoon)", re.IGNORECASE),
        "Usually {val}.",
        "habit"
    ),
]


_STT_ARTIFACTS = re.compile(
    r"^(?:this\s+i\s+(?:have|am|was|had)?\s*|uh+\s*|um+\s*|well\s*|you\s+know\s*|like\s*)+",
    re.IGNORECASE,
)


def clean_stt_text(text: str) -> str:
    """Strip common STT transcription disfluencies and filler prefixes."""
    cleaned = _STT_ARTIFACTS.sub("", text.strip()).strip()
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


class FactExtractor:
    """Extracts atomic user facts and detects updates/duplicates."""

    @staticmethod
    def extract_from_utterance(text: str) -> List[FactItem]:
        """
        Extract candidate facts from user utterance using fast pattern heuristics.
        Returns a list of FactItem objects ready for upserting.
        """
        norm = clean_stt_text(text)
        lower = norm.lower()

        # Discard non-declarative utterances (questions, commands, greetings)
        if any(lower.startswith(prefix) for prefix in _IGNORE_STARTS):
            return []

        # If question mark present at end, skip
        if norm.endswith("?"):
            return []

        extracted: List[FactItem] = []

        for pattern, template, category in _EXTRACTION_RULES:
            match = pattern.search(norm)
            if match:
                groups = {k: v.strip().capitalize() if k == "name" else v.strip() for k, v in match.groupdict().items()}
                try:
                    fact_text = template.format(**groups)
                    # Normalize first letter capitalization
                    fact_text = fact_text[0].upper() + fact_text[1:]
                    if not fact_text.endswith("."):
                        fact_text += "."
                    extracted.append(FactItem(text=fact_text, category=category))
                except Exception as e:
                    log.debug(f"Template formatting error: {e}")

        return extracted

    @staticmethod
    def determine_action(new_fact: str, existing_facts: List[str]) -> Tuple[str, Optional[int]]:
        """
        Compares new_fact against existing facts to determine action:
        - ('NONE', None): already known / identical
        - ('UPDATE', index): supersedes an existing fact (e.g. new favorite color)
        - ('ADD', None): brand new fact
        """
        new_norm = clean_stt_text(new_fact)
        new_lower = new_norm.lower().strip()
        new_tokens = set(re.findall(r"[a-z0-9]+", new_lower))
        if not new_tokens:
            return "NONE", None

        for idx, exist in enumerate(existing_facts):
            exist_lower = clean_stt_text(exist).lower().strip()
            exist_tokens = set(re.findall(r"[a-z0-9]+", exist_lower))
            if not exist_tokens:
                continue

            # Exact match -> already exists
            if new_lower == exist_lower:
                return "NONE", None

            # High token overlap (> 80%) -> already covered
            overlap = len(new_tokens & exist_tokens)
            if overlap / max(len(new_tokens), len(exist_tokens)) >= 0.80:
                return "NONE", None

            # Attribute clashes:
            # 1. Location updates ("Lives in X" vs "Lives in Y")
            if new_lower.startswith("lives in ") and exist_lower.startswith("lives in "):
                return "UPDATE", idx

            # 2. Favorite item updates ("Favorite X is Y" vs "Favorite X is Z")
            if new_lower.startswith("favorite ") and exist_lower.startswith("favorite "):
                new_item = new_lower.split(" is ")[0]
                exist_item = exist_lower.split(" is ")[0]
                if new_item == exist_item:
                    return "UPDATE", idx

            # 3. Same subject/verb prefix (first 3 words match) -> UPDATE
            new_prefix = " ".join(new_lower.split()[:3])
            exist_prefix = " ".join(exist_lower.split()[:3])
            if new_prefix == exist_prefix and len(new_prefix) > 8:
                return "UPDATE", idx

        return "ADD", None
