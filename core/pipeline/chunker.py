"""
chunker.py - Low-latency streaming sentence and clause chunker for TTS overlap.
"""
from __future__ import annotations

import re
from typing import AsyncIterator, Iterator

import config

# Punctuation marks where speech can naturally pause
MAJOR_BREAKS = {".", "!", "?", "\n"}
MINOR_BREAKS = {",", ";", ":", "—", "-"}


class SentenceChunker:
    """
    Accumulates streaming tokens and yields speakable sentence/clause chunks.
    Enables starting TTS synthesis on the first sentence while subsequent tokens generate.
    """

    def __init__(self, min_chars: int = config.TTS_CHUNK_MIN_CHARS):
        self.min_chars = min_chars
        self.buffer = ""
        self._first_chunk = True

    def reset(self) -> None:
        """Reset chunker state for a new turn."""
        self.buffer = ""
        self._first_chunk = True

    def add_token(self, token: str) -> list[str]:
        """Add a token and return any completed chunks."""
        self.buffer += token
        chunks = []

        while True:
            # Look for major punctuation breaks (. ? ! \n)
            split_idx = -1
            match = re.search(r"[.!?\n]", self.buffer)
            if match:
                idx = match.end()
                # Check that we have enough characters or it's an exclamation/question
                if idx >= self.min_chars or match.group() in {"!", "?"} or (self._first_chunk and idx >= 10):
                    split_idx = idx

            # If no major break, look for minor breaks (, ; :)
            # For the VERY FIRST chunk, allow early emit on comma/colon at >= 8 chars (e.g. "Sure,", "Yes,")
            if split_idx == -1:
                match_minor = re.search(r"[,;:\u2014]", self.buffer)
                if match_minor:
                    idx_m = match_minor.end()
                    if self._first_chunk and idx_m >= 8:
                        split_idx = idx_m
                    elif len(self.buffer) >= self.min_chars * 2 and idx_m >= self.min_chars:
                        split_idx = idx_m

            if split_idx != -1:
                chunk = self.buffer[:split_idx].strip()
                self.buffer = self.buffer[split_idx:].lstrip()
                if chunk:
                    chunks.append(chunk)
                    self._first_chunk = False
            else:
                break

        return chunks

    def flush(self) -> list[str]:
        """Flush any remaining buffered text at end of stream."""
        remaining = self.buffer.strip()
        self.buffer = ""
        self._first_chunk = True
        if remaining:
            return [remaining]
        return []
