"""
models.py - Structured memory data models for Iris Companion.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


def now_iso() -> str:
    """Return current UTC ISO timestamp string."""
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def new_id(prefix: str = "mem") -> str:
    """Generate a short unique ID with prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class FactItem:
    """Represents an atomic long-term fact or preference about the user."""
    text: str
    id: str = field(default_factory=lambda: new_id("fact"))
    category: str = "general"          # preference | habit | goal | bio | general
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    access_count: int = 0
    confidence: float = 1.0
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FactItem:
        if isinstance(data, str):
            return cls(text=data)
        return cls(
            text=data.get("text", ""),
            id=data.get("id") or new_id("fact"),
            category=data.get("category", "general"),
            created_at=data.get("created_at") or data.get("date") or now_iso(),
            updated_at=data.get("updated_at") or now_iso(),
            access_count=data.get("access_count", 0),
            confidence=data.get("confidence", 1.0),
            tags=data.get("tags") or [],
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FriendItem:
    """Represents a person or relationship mentioned by the user."""
    name: str
    relation: str = "friend"
    note: str = ""
    id: str = field(default_factory=lambda: new_id("rel"))
    updated_at: str = field(default_factory=now_iso)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FriendItem:
        return cls(
            name=data.get("name", ""),
            relation=data.get("relation", "friend"),
            note=data.get("note", ""),
            id=data.get("id") or new_id("rel"),
            updated_at=data.get("updated_at") or now_iso(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FileItem:
    """Represents a personal file reference remembered for the user."""
    name: str
    path: str = ""
    note: str = ""
    id: str = field(default_factory=lambda: new_id("file"))
    updated_at: str = field(default_factory=now_iso)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FileItem:
        return cls(
            name=data.get("name", ""),
            path=data.get("path", ""),
            note=data.get("note", ""),
            id=data.get("id") or new_id("file"),
            updated_at=data.get("updated_at") or now_iso(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
