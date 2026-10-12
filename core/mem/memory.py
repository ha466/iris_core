"""
memory.py - Rules-based memory retrieval for voice assistants
=============================================================

A query only touches memory if one of the RULES below matches it.
Anything else ("play jazz", "what's the capital of France", "who is Elon Musk")
returns None -> zero memory work, zero added latency, nothing injected in the prompt.

Memory types
------------
  identity   structured facts: profile (name, age, ...), friends/people, personal files
  story      free-text story about the user (chunked)
  long_term  facts / preferences / things the user told the assistant over time

How retrieval works
-------------------
  1. normalize()    lowercase, expand contractions, drop STT fillers ("uh", "um")
  2. rules          regex rules; the LAST matching clause in the utterance wins, so
                    stutters like "who is my.. who is my.. what is my name" still work
  3. handler        - structured lookups (name, friend count, file path...) answer directly
                      from the JSON store: no embeddings, ~0.1 ms
                    - fuzzy lookups (story, long-term, file search) use the embedding model
                      only inside the memory type the rule selected

Embedding model (English-only, tiny, CPU-friendly)
--------------------------------------------------
  Snowflake/snowflake-arctic-embed-xs   22M params, 384-dim, Apache-2.0
  (MiniLM-L6 sized, MTEB retrieval NDCG@10 = 50.15 - best in its size class)
  Backends tried in order: fastembed (ONNX, no PyTorch) -> sentence-transformers
  -> pure-Python keyword matching (so nothing ever crashes the assistant).

  pip install numpy fastembed            # recommended (light)
  # or: pip install numpy sentence-transformers

Quick integration
-----------------
    from memory import RuleMemory
    mem = RuleMemory("memory_store.json")
    mem.warmup()                          # once at startup (background thread)

    hit = mem.retrieve(user_text)         # None => don't use memory at all
    if hit and hit.direct:
        speak(hit.answer)                 # skip the LLM entirely
    else:
        system_prompt += "\\n\\n" + hit.context if hit else ""
"""
from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Optional

import numpy as np

try:
    from core.mem.manager import MemoryManager
except ImportError:
    from manager import MemoryManager

log = logging.getLogger("rulememory")

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
EMBED_MODEL_ST = "Snowflake/snowflake-arctic-embed-xs"          # sentence-transformers id
EMBED_MODEL_FE = "snowflake/snowflake-arctic-embed-xs"          # fastembed id
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # model card

PROFILE_LABELS = {
    "name": "name",
    "nickname": "nickname",
    "age": "age",
    "birthday": "birthday",
    "city": "location",
    "job": "job",
    "email": "email address",
}

DEFAULT_RELATIONS = {
    "friend", "best friend", "close friend", "brother", "sister", "mom", "dad",
    "wife", "husband", "girlfriend", "boyfriend", "partner", "boss", "manager",
    "colleague", "roommate", "cousin", "uncle", "aunt", "son", "daughter",
    "grandmother", "grandfather", "teacher", "mentor", "neighbor", "fiance",
}
_SYN = {
    "mother": "mom", "mum": "mom", "mama": "mom", "mommy": "mom", "father": "dad",
    "papa": "dad", "daddy": "dad", "grandma": "grandmother", "grandpa": "grandfather",
    "neighbour": "neighbor", "coworker": "colleague", "fiancee": "fiance",
}

# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
_CONTRACTIONS = (
    ("what's", "what is"), ("who's", "who is"), ("where's", "where is"),
    ("how's", "how is"), ("when's", "when is"), ("that's", "that is"),
    ("what're", "what are"), ("who're", "who are"), ("i'm", "i am"),
    ("i've", "i have"), ("i'd", "i would"), ("i'll", "i will"),
    ("don't", "do not"), ("didn't", "did not"), ("doesn't", "does not"),
    ("can't", "cannot"), ("it's", "it is"), ("let's", "let us"),
)
_DISFLUENCY = re.compile(r"\b(?:u+h+|u+m+|e+r+m*|h+m+|a+h+|m{2,})\b")
_STOP = set(
    "a an the of to in on at for and or is are was were do does did i me my you your we our "
    "it this that what who where when how about with be have has had can could would should "
    "tell remember recall please file files document documents doc docs".split()
)


def normalize(text: str) -> str:
    """Lowercase, expand contractions, strip punctuation and speech fillers."""
    t = text.lower().replace("\u2019", "'").replace("\u2018", "'")
    for a, b in _CONTRACTIONS:
        t = t.replace(a, b)
    t = t.replace("'", "")
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = _DISFLUENCY.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def _tokens(text: str) -> set[str]:
    out = set()
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in _STOP or len(w) < 2:
            continue
        if len(w) > 3 and w.endswith("s"):
            w = w[:-1]
        out.add(w)
    return out


def _lex_score(query: str, doc: str) -> float:
    """Fraction of query keywords present in doc (cheap fallback / exact-name matcher)."""
    q = _tokens(query)
    if not q:
        return 0.0
    return len(q & _tokens(doc)) / len(q)


def _canon(rel: str) -> str:
    r = rel.lower().strip()
    return _SYN.get(r, r)


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _cap(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 3].rstrip() + "..."


def _ctx(label: str, lines: list[str]) -> str:
    return f"[MEMORY:{label}]\n" + "\n".join(f"- {l}" for l in lines)


def _chunk(text: str, max_chars: int = 350) -> list[str]:
    out: list[str] = []
    for para in re.split(r"\n\s*\n", text.strip()):
        para = " ".join(para.split())
        if not para:
            continue
        if len(para) <= max_chars:
            out.append(para)
            continue
        cur = ""
        for s in re.split(r"(?<=[.!?])\s+", para):
            if cur and len(cur) + 1 + len(s) > max_chars:
                out.append(cur)
                cur = s
            else:
                cur = (cur + " " + s).strip()
        if cur:
            out.append(cur)
    return out


# --------------------------------------------------------------------------- #
# Embedder (lazy, thread-safe, never raises into the conversation)
# --------------------------------------------------------------------------- #
class Embedder:
    """backend: 'auto' | 'fastembed' | 'sentence-transformers' | 'none'"""

    def __init__(self, backend: str = "auto", threads: int = 2):
        self.backend = backend
        self.threads = threads
        self.state = "failed" if backend == "none" else "idle"  # idle|loading|ready|failed
        self.tag = "none"
        self.error: Optional[str] = None
        self._model: Any = None
        self._kind = ""
        self._lock = threading.Lock()
        self._spawned = False

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    def load(self) -> bool:
        with self._lock:
            if self.state in ("ready", "failed"):
                return self.ready
            self.state = "loading"
            t0 = time.perf_counter()
            if self.backend in ("auto", "fastembed"):
                try:
                    from fastembed import TextEmbedding

                    self._model = TextEmbedding(model_name=EMBED_MODEL_FE, threads=self.threads)
                    self._kind, self.tag = "fastembed", f"fastembed:{EMBED_MODEL_FE}"
                except Exception as e:  # noqa: BLE001
                    self.error = f"fastembed: {e!r}"
            if self._model is None and self.backend in ("auto", "sentence-transformers"):
                try:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(EMBED_MODEL_ST, device="cpu")
                    self._kind, self.tag = "st", f"st:{EMBED_MODEL_ST}"
                except Exception as e:  # noqa: BLE001
                    self.error = f"{self.error or ''} | sentence-transformers: {e!r}"
            self.state = "ready" if self._model is not None else "failed"
            if self.ready:
                log.info("embedder ready: %s (%.0f ms)", self.tag, (time.perf_counter() - t0) * 1000)
            else:
                log.warning("no embedding backend available, using keyword matching. %s", self.error)
            return self.ready

    def load_async(self) -> None:
        if self.state != "idle" or self._spawned:
            return
        self._spawned = True
        threading.Thread(target=self.load, daemon=True, name="embedder-load").start()

    def embed(self, texts: list[str], query: bool = False) -> np.ndarray:
        if not self.ready:
            raise RuntimeError("embedder not ready")
        if query:
            texts = [QUERY_PREFIX + t for t in texts]
        if self._kind == "fastembed":
            vecs = np.array(list(self._model.embed(texts)), dtype=np.float32)
        else:
            vecs = np.asarray(
                self._model.encode(texts, normalize_embeddings=True, batch_size=32, convert_to_numpy=True),
                dtype=np.float32,
            )
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-9)


# --------------------------------------------------------------------------- #
# Result + rule containers
# --------------------------------------------------------------------------- #
@dataclass
class MemoryResult:
    found: bool                          # was the thing actually in memory?
    answer: str = ""                     # short speakable sentence (TTS-ready)
    context: str = ""                    # block to inject into the LLM prompt
    direct: Optional[bool] = None        # True => safe to speak `answer` without calling the LLM
    data: dict = field(default_factory=dict)
    rule: str = ""
    memory_type: str = ""
    used_embeddings: bool = False
    latency_ms: float = 0.0


@dataclass
class Rule:
    name: str
    mtype: str
    patterns: list[re.Pattern]
    handler: Callable[[re.Match, str], Optional[MemoryResult]]
    priority: int
    direct: bool = False


# --------------------------------------------------------------------------- #
# RuleMemory
# --------------------------------------------------------------------------- #
def _empty_store() -> dict:
    return {
        "version": 1,
        "identity": {"profile": {}, "friends": [], "files": []},
        "story": [],
        "long_term": [],
    }


class RuleMemory:
    def __init__(
        self,
        store_path: str = "memory_store.json",
        embedder: Optional[Embedder] = None,
        assistant_names: tuple[str, ...] = (),
        min_score: float = 0.38,          # cosine threshold for arctic-embed-xs (tune 0.30-0.50)
        min_score_lexical: float = 0.5,   # keyword-overlap threshold when no embedder
        top_k: int = 3,
        max_context_chars: int = 900,
        block_first_use: bool = False,    # True: wait for model load on first fuzzy query
        db_path: Optional[str] = None,
    ):
        self.store_path = store_path
        self.cache_path = os.path.splitext(store_path)[0] + ".emb.npz"
        self.embedder = embedder or Embedder("auto")
        self.min_score = min_score
        self.min_score_lexical = min_score_lexical
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.block_first_use = block_first_use
        self.enabled = True

        self._lock = threading.RLock()
        self._idx_lock = threading.Lock()
        self._vec: dict[str, np.ndarray] = {}
        self._memo: dict[str, tuple[list[str], np.ndarray]] = {}
        self._dirty = False

        db = db_path or (os.path.splitext(store_path)[0] + ".db" if not store_path.endswith("memory_store.json") else None)
        self.db = MemoryManager(db)
        self.data = self._load_store()
        self._load_cache()

        extra = "".join("|" + re.escape(n.lower()) for n in assistant_names)
        self._tail = (
            r"(?:\s+(?:please|again|now|today|sir|buddy|mate|then|friend|assistant|quickly|"
            r"once more|one more time|right now|by the way|real quick|for me" + extra + r"))*\s*$"
        )
        self.rules = self._build_rules()

    # ------------------------------------------------------------------ rules
    def _build_rules(self) -> list[Rule]:
        A = lambda s: re.compile(s + self._tail)  # anchored: must end the utterance (+ polite tail)
        F = re.compile                            # free: prefix match, rest of sentence is the topic
        R: list[Rule] = []

        # ---- identity: profile fields --------------------------------------------------
        prof = {
            "name": [
                r"\bwhat is my (?:full |first |last |real )?name",
                r"\bwho am i",
                r"\bdo you (?:know|remember|recall) my (?:full |first |real )?name",
                r"\bwhat (?:do|should|can) you call me",
                r"\bsay my name",
                r"\btell me my name",
            ],
            "nickname": [r"\bwhat is my (?:nick ?name|user ?name|handle)"],
            "age": [r"\bhow old am i", r"\bwhat is my age"],
            "birthday": [
                r"\bwhen is my (?:birthday|birth day|birth ?date)",
                r"\bwhen was i born",
                r"\bwhat is my (?:birthday|birth ?date|date of birth)",
            ],
            "city": [
                r"\bwhere do i (?:live|stay)",
                r"\bwhere am i from",
                r"\bwhat is my (?:city|hometown|home town|country|location)",
            ],
            "job": [
                r"\bwhat do i do for (?:a )?(?:work|living)",
                r"\bwhere do i work",
                r"\bwhat is my (?:job|profession|occupation|role|designation)",
            ],
            "email": [r"\bwhat is my e ?mail(?: address| id)?"],
        }
        for key, pats in prof.items():
            R.append(Rule(f"identity.{key}", "identity", [A(p) for p in pats],
                          functools.partial(self._h_profile, key=key), 10, direct=True))

        # ---- identity: friends / people -------------------------------------------------
        R.append(Rule("friends.count", "identity", [
            A(r"\bhow many (?:close |best |good )?friends(?: do i have| have i got| do i know| i have)?"),
            A(r"\b(?:number|count|total) of (?:my )?friends"),
        ], self._h_friends_count, 20, direct=True))
        R.append(Rule("friends.list", "identity", [
            A(r"\b(?:who are|list|name|show|tell me|what are|read out) (?:all )?my (?:close |best |good |all )?friends"),
            A(r"\bdo i have (?:any )?friends"),
        ], self._h_friends_list, 21, direct=True))
        R.append(Rule("friends.relation", "identity", [
            A(r"\bwho is my (?P<rel>[a-z ]{2,25}?)"),
            A(r"\bwhat is my (?P<rel>[a-z]{2,15}) name"),
        ], self._h_relation, 22, direct=True))
        R.append(Rule("entity", "identity", [
            A(r"\b(?:who is|who was|tell me about|what do you know about|do you know)\s+"
              r"(?!my\b|me\b|myself\b|the\b|a\b|an\b|your\b|you\b)(?P<who>[a-z]+(?: [a-z]+)?)"),
            A(r"\bwho is (?P<who>[a-z]+(?: [a-z]+)?) to me"),
        ], self._h_entity, 30, direct=True))

        # ---- identity: personal files ----------------------------------------------------
        R.append(Rule("files.count", "identity", [
            A(r"\bhow many (?:personal |saved )?(?:files|documents|docs)(?: do i have| have i saved| did i save| i have)?"),
        ], self._h_files_count, 40, direct=True))
        R.append(Rule("files.list", "identity", [
            A(r"\b(?:what|which) (?:personal |saved )?(?:files|documents|docs) do i have"),
            A(r"\b(?:list|show|read out|tell me) my (?:personal |saved )?(?:files|documents|docs)"),
            A(r"\bwhat are my (?:personal |saved )?(?:files|documents|docs)"),
            A(r"\bdo i have (?:any )?(?:personal )?(?:files|documents)"),
        ], self._h_files_list, 41, direct=True))
        R.append(Rule("files.find", "identity", [
            A(r"\bwhere (?:is|are) my (?P<q>[a-z0-9 ]{2,40}?) (?:file|files|document|documents|doc|docs|pdf|sheet|spreadsheet|notes)"),
            A(r"\b(?:find|open|get|show me|pull up) my (?P<q>[a-z0-9 ]{2,40}?) (?:file|document|doc|pdf|sheet|spreadsheet)"),
        ], functools.partial(self._h_file_find, strict=True), 42, direct=True))
        R.append(Rule("files.find_loose", "identity", [
            A(r"\bwhere did i (?:save|put|store|keep) (?:my |the )?(?P<q>[a-z0-9 ]{2,40}?)"),
        ], functools.partial(self._h_file_find, strict=False), 43, direct=True))

        # ---- story ------------------------------------------------------------------------
        R.append(Rule("about_me", "identity", [
            A(r"\bwhat do you (?:know|remember) about me"),
            A(r"\btell me about (?:myself|me)"),
        ], self._h_about_me, 45))
        R.append(Rule("story.full", "story", [
            A(r"\b(?:tell|say|narrate|read|recite) (?:me )?my (?:life )?story"),
            A(r"\bwhat is my (?:life )?story"),
            A(r"\bmy life story"),
            A(r"\bwhat is my background"),
            A(r"\btell me about my (?:background|journey|life|past)"),
        ], self._h_story_full, 50))
        R.append(Rule("story.search", "story", [
            F(r"\b(?:where|how|when|why) did i (?:grow up|start|begin|get started|learn|meet|move|end up)\b"),
            F(r"\btell me about my (?:childhood|school|college|family|upbringing|early life)"),
            F(r"\bwhat happened (?:in|during|when|after) my (?:childhood|school|college|life)"),
        ], self._h_story_search, 51))

        # ---- long-term --------------------------------------------------------------------
        R.append(Rule("ltm.recent", "long_term", [
            F(r"\blast time (?:we|i) (?:talked|spoke|chatted|discussed|met)"),
            F(r"\bwhat did we (?:talk|speak|chat|discuss) about"),
            F(r"\bwhat were we (?:talking|speaking|chatting) about"),
            F(r"\bwhere did we (?:leave off|stop)"),
            F(r"\bwhat did i (?:tell|say|mention) (?:to )?(?:you )?(?:last time|yesterday|earlier|before)"),
        ], self._h_ltm_recent, 55))
        R.append(Rule("ltm.search", "long_term", [
            F(r"\bdo you (?:remember|recall) (?:when|that|what|how|where|the time|me|my|our|i|we|about my|about me)\b"),
            F(r"\bwhat did i (?:tell|say|mention|share|ask)\b"),
            F(r"\b(?:have|did) i (?:ever )?(?:tell|told|mention|mentioned|say|said) (?:you|to you)\b"),
            F(r"\bremind me (?:what|about|of)\b"),
            F(r"\bwhat do i (?:like|love|hate|enjoy|prefer|want|need|usually|always|normally|dislike)\b"),
            F(r"\bwhat is my (?:favou?rite|fav)\b"),
            F(r"\bwhat are my (?:hobbies|interests|goals|plans|habits|preferences|favou?rites|dreams)\b"),
            F(r"\bwhat is my (?:hobby|interest|goal|plan|habit|preference|dream)\b"),
        ], self._h_ltm_search, 60))

        # ---- explicit write / memorization -----------------------------------------------
        R.append(Rule("ltm.remember", "long_term", [
            A(r"(?:^|\b(?:please|iris)\s+)\b(?:remember|don't forget|do not forget|note down|keep in mind)(?: that)? (?P<fact>[a-z0-9 ]{3,100})"),
        ], self._h_ltm_remember, 5, direct=True))
        R.append(Rule("identity.set", "identity", [
            A(r"\bmy (?P<key>name|nickname|birthday|city|job|email) is (?P<val>[a-z0-9 ]{2,50})"),
            A(r"\b(?:i(?:'ve| have)? changed my|update my) (?P<key>name|nickname|birthday|city|job|email) to (?P<val>[a-z0-9 ]{2,50})"),
            A(r"\bmy new (?P<key>name|nickname|birthday|city|job|email) is (?P<val>[a-z0-9 ]{2,50})"),
        ], self._h_set_profile, 6, direct=True))
        R.append(Rule("ltm.forget", "long_term", [
            A(r"\b(?:forget|delete|remove|erase|clear)(?: that| about| the| my)? (?P<target>[a-z0-9 ]{3,100})"),
            A(r"\bdon't remember (?:that|about|my)? (?P<target>[a-z0-9 ]{3,100})"),
        ], self._h_ltm_forget, 4, direct=True))
        R.append(Rule("ltm.suppress", "control", [
            A(r"\bdon't save this (?:conversation|turn|chat)?"),
            A(r"\bkeep this (?:conversation )?private"),
            A(r"\bthis is off the record"),
        ], self._h_suppress, 3, direct=True))
        R.append(Rule("ltm.recall_all", "long_term", [
            A(r"\bwhat (?:do you|all do you) remember (?:about me)?"),
            A(r"\bwhat do you have (?:saved|stored) (?:about me)?"),
            A(r"\bshow (?:my|all) memories"),
        ], self._h_recall_all, 15, direct=True))
        R.append(Rule("episode.recall", "episode", [
            A(r"\bwhat did we (?:talk|discuss|speak|chat) about (?:last time|yesterday|earlier|before)"),
            A(r"\bwhat was our last conversation about"),
        ], self._h_episode_recall, 50, direct=True))
        return R

    # --------------------------------------------------------------- retrieval
    def retrieve(self, text: str) -> Optional[MemoryResult]:
        """Return a MemoryResult if a rule fires, else None (=> do not use memory)."""
        if not self.enabled or not text:
            return None
        t0 = time.perf_counter()
        norm = normalize(text)
        if not norm:
            return None
        cands = []
        for rule in self.rules:
            for pat in rule.patterns:
                m = pat.search(norm)
                if m:
                    cands.append((m.start(), rule.priority, rule, m))
                    break
        if not cands:
            return None
        # Discard candidates that are strictly internal substrings of another candidate
        cands = [
            c for c in cands
            if not any(
                o[3].start() <= c[0] and o[3].end() >= c[3].end() and (o[3].start(), o[3].end()) != (c[0], c[3].end())
                for o in cands
            )
        ]
        # last clause in the utterance wins (handles stutters / restarts); ties -> priority
        cands.sort(key=lambda c: (-c[0], c[1]))
        for _, _, rule, m in cands:
            try:
                res = rule.handler(m, norm)
            except Exception:  # never crash a live conversation
                log.exception("memory rule %s failed", rule.name)
                continue
            if res is None:
                continue
            res.rule, res.memory_type = rule.name, rule.mtype
            if res.direct is None:
                res.direct = rule.direct
            res.latency_ms = (time.perf_counter() - t0) * 1000
            return res
        return None

    def context_for(self, text: str) -> Optional[str]:
        """Convenience: prompt-injection block, or None when memory shouldn't be used."""
        res = self.retrieve(text)
        return res.context if res else None

    def _profile(self) -> dict:
        return self.data["identity"]["profile"]

    def get_profile(self) -> dict:
        """Return the user profile dictionary."""
        return self._profile()

    def _friends_only(self) -> list[dict]:
        return [f for f in self.data["identity"]["friends"] if "friend" in _canon(f.get("relation", "friend"))]

    def _find_friend(self, name: str) -> Optional[dict]:
        n = name.lower().strip()
        for f in self.data["identity"]["friends"]:
            fn = f["name"].lower()
            if n == fn or n == fn.split()[0]:
                return f
        return None

    def _h_profile(self, m, norm, key):
        label, val = PROFILE_LABELS[key], self._profile().get(key)
        if val:
            return MemoryResult(True, f"Your {label} is {val}.",
                                _ctx("identity", [f"The user's {label} is {val}."]), data={key: val})
        return MemoryResult(False, f"I don't have your {label} saved yet.",
                            _ctx("identity", [f"The user's {label} is not saved. Do not guess it."]))

    def _h_entity(self, m, norm):
        who = m.group("who").strip()
        prof = self._profile()
        cands = [who] + ([who.split()[0]] if " " in who else [])
        for c in cands:
            for k in ("name", "nickname"):
                v = (prof.get(k) or "").lower().strip()
                if v and (c == v or c == v.split()[0]):
                    extra = f" You live in {prof['city']}." if prof.get("city") else ""
                    lines = [f"{PROFILE_LABELS[k]}: {val}" for k, val in prof.items() if val]
                    return MemoryResult(True, f"{prof.get(k)} is you.{extra}", _ctx("identity", lines),
                                        data={"person": "user", **prof})
            f = self._find_friend(c)
            if f:
                note = f" {f['note']}" if f.get("note") else ""
                rel = f.get("relation", "friend")
                return MemoryResult(True, f"{f['name']} is your {rel}.{note}",
                                    _ctx("identity", [f"{f['name']} - the user's {rel}. {f.get('note', '')}".strip()]),
                                    data=dict(f))
        return None  # unknown person -> memory not used

    def _known_relations(self) -> set[str]:
        return set(DEFAULT_RELATIONS) | {_canon(f.get("relation", "friend")) for f in self.data["identity"]["friends"]}

    def _h_relation(self, m, norm):
        rel = m.group("rel").strip()
        known = self._known_relations()
        cands = [rel, rel + " friend"] if not rel.endswith("friend") else [rel]
        key = None
        for r_cand in cands:
            key = next((c for c in (_canon(r_cand), _canon(r_cand[:-1]) if r_cand.endswith("s") else None) if c in known), None)
            if key is not None:
                break
        if key is None:
            return None  # "who is my favorite singer" etc. -> not a memory question
        people = self._friends_only() if key == "friend" else [
            f for f in self.data["identity"]["friends"] if _canon(f.get("relation", "friend")) == key]
        if not people:
            return MemoryResult(False, f"You haven't told me who your {key} is yet.",
                                _ctx("identity", [f"No {key} saved. Do not guess."]))
        names = [p["name"] for p in people]
        ans = f"Your {key} is {names[0]}." if len(names) == 1 else f"Your {key}s are {_join(names)}."
        lines = [f"{p['name']} ({p.get('relation', 'friend')}) {p.get('note', '')}".strip() for p in people]
        return MemoryResult(True, ans, _ctx("identity", lines), data={"people": people})

    def _h_friends_count(self, m, norm):
        fr = self._friends_only()
        n = len(fr)
        if n == 0:
            return MemoryResult(False, "You haven't told me about any friends yet.",
                                _ctx("identity", ["No friends saved. Do not guess a number."]), data={"count": 0})
        names = [f["name"] for f in fr]
        tail = f": {_join(names)}" if n <= 6 else ""
        return MemoryResult(True, f"You have {n} friend{'s' if n != 1 else ''}{tail}.",
                            _ctx("identity", [f"The user has {n} friends: {_join(names)}"]), data={"count": n})

    def _h_friends_list(self, m, norm):
        fr = self._friends_only()
        if not fr:
            return MemoryResult(False, "You haven't told me about any friends yet.",
                                _ctx("identity", ["No friends saved."]), data={"friends": []})
        lines = [f"{f['name']} ({f.get('relation', 'friend')}) {f.get('note', '')}".strip() for f in fr]
        return MemoryResult(True, f"Your friends are {_join([f['name'] for f in fr])}.",
                            _ctx("identity", lines), data={"friends": fr})

    def _h_files_count(self, m, norm):
        n = len(self.data["identity"]["files"])
        return MemoryResult(n > 0, f"You have {n} personal file{'s' if n != 1 else ''} saved." if n
                            else "You haven't saved any personal files yet.",
                            _ctx("identity", [f"The user has {n} personal files saved."]), data={"count": n})

    def _h_files_list(self, m, norm):
        files = self.data["identity"]["files"]
        if not files:
            return MemoryResult(False, "You haven't saved any personal files yet.",
                                _ctx("identity", ["No personal files saved."]))
        names = [f["name"] for f in files]
        shown = _join(names[:8]) + (f", and {len(names) - 8} more" if len(names) > 8 else "")
        lines = [f"{f['name']}: {f.get('path') or 'no path'} {f.get('note', '')}".strip() for f in files]
        return MemoryResult(True, f"Your files are {shown}.", _ctx("identity", lines), data={"files": files})

    def _h_file_find(self, m, norm, strict):
        q = m.group("q").strip()
        hits, used = self._search("files", q, k=1, lexical_first=True)
        if not hits:
            if not strict:
                return None  # "where did I save my keys" is not a file question
            return MemoryResult(False, f"I don't have a file called {q} saved.",
                                _ctx("identity", [f"No saved file matches '{q}'. Do not guess a path."]))
        _, (_, f) = hits[0]
        ans = f"Your {f['name']} is at {f['path']}." if f.get("path") else f"I have {f['name']} noted. {f.get('note', '')}".strip()
        return MemoryResult(True, ans, _ctx("identity", [f"{f['name']}: {f.get('path', '')} {f.get('note', '')}".strip()]),
                            data=dict(f), used_embeddings=used)

    def _h_about_me(self, m, norm):
        prof, lines = self._profile(), []
        lines += [f"{PROFILE_LABELS.get(k, k)}: {v}" for k, v in prof.items() if v]
        fr = self._friends_only()
        if fr:
            lines.append(f"{len(fr)} friends: {_join([f['name'] for f in fr])}")
        if self.data["identity"]["files"]:
            lines.append(f"{len(self.data['identity']['files'])} personal files saved")
        if self.data["story"]:
            lines.append("Story: " + _cap(self.data["story"][0], 300))
        if not lines:
            return MemoryResult(False, "I don't know much about you yet.",
                                _ctx("identity", ["Nothing saved about the user yet."]), direct=True)
        return MemoryResult(True, "", _cap(_ctx("identity", lines), self.max_context_chars))

    def _h_story_full(self, m, norm):
        chunks = self.data["story"]
        if not chunks:
            return MemoryResult(False, "You haven't told me your story yet.",
                                _ctx("story", ["No story saved. Do not invent one."]), direct=True)
        return MemoryResult(True, _cap(" ".join(chunks[:2]), 400),
                            _cap(_ctx("story", chunks), self.max_context_chars * 2), data={"story": chunks})

    def _h_story_search(self, m, norm):
        hits, used = self._search("story", norm, k=self.top_k)
        if not hits:
            return MemoryResult(False, "", _ctx("story", ["Nothing in the user's saved story covers this. Do not guess."]))
        lines = [p["text"] for _, (_, p) in hits]
        return MemoryResult(True, lines[0], _cap(_ctx("story", lines), self.max_context_chars),
                            data={"hits": lines}, used_embeddings=used)

    def _h_ltm_search(self, m, norm):
        hits, used = self._search("long_term", norm, k=self.top_k)
        if not hits:
            return MemoryResult(False, "", _ctx("long_term", ["Nothing saved about this. Say you don't remember; do not guess."]))
        lines = [f"{p['text']} (saved {p.get('date', '?')})" for _, (_, p) in hits]
        return MemoryResult(True, hits[0][1][1]["text"], _cap(_ctx("long_term", lines), self.max_context_chars),
                            data={"hits": [p for _, (_, p) in hits]}, used_embeddings=used)

    def _h_ltm_recent(self, m, norm):
        items = sorted(self.data["long_term"], key=lambda x: x.get("date", ""), reverse=True)[: self.top_k]
        if not items:
            return MemoryResult(False, "", _ctx("long_term", ["No earlier conversations saved."]))
        lines = [f"{p['text']} (saved {p.get('date', '?')})" for p in items]
        return MemoryResult(True, items[0]["text"], _cap(_ctx("long_term", lines), self.max_context_chars),
                            data={"hits": items})

    def _h_ltm_remember(self, m, norm):
        fact = m.group("fact").strip()
        for tail in ("please", "iris", "for me", "by the way", "real quick", "right now"):
            if fact.lower().endswith(" " + tail):
                fact = fact[:-len(tail)-1].rstrip()
        if not fact:
            return None
        fact_clean = fact[0].upper() + fact[1:]
        if not fact_clean.endswith("."):
            fact_clean += "."
        action, item = self.upsert_fact(fact_clean, category="preference")
        return MemoryResult(
            found=True,
            answer=f"Got it, I will remember that {fact}.",
            context=_ctx("long_term", [fact_clean]),
            direct=True,
            data=item,
        )

    def _h_set_profile(self, m, norm):
        k = m.group("key").strip()
        v = m.group("val").strip()
        for tail in ("please", "iris", "for me"):
            if v.lower().endswith(" " + tail):
                v = v[:-len(tail)-1].rstrip()
        if not v:
            return None
        v_clean = v.capitalize()
        self.set_profile(k, v_clean)
        label = PROFILE_LABELS.get(k, k)
        return MemoryResult(
            found=True,
            answer=f"Got it, I updated your {label} to {v_clean}.",
            context=_ctx("identity", [f"The user's {label} is {v_clean}."]),
            direct=True,
            data={k: v_clean},
        )

    def _h_ltm_forget(self, m, norm):
        target = m.group("target").strip()
        for tail in ("please", "iris", "for me", "by the way", "real quick", "right now"):
            if target.lower().endswith(" " + tail):
                target = target[:-len(tail)-1].rstrip()
        if not target:
            return None
        forgotten = self.db.forget(target)
        if forgotten:
            self.data = self.db.load_full_store()
            self._dirty = True
            self._memo.pop("long_term", None)
            self._save()
            return MemoryResult(
                found=True,
                answer=f"I have forgotten that: {forgotten.get('text', target)}.",
                context=_ctx("long_term", [f"User requested to forget {target}."]),
                direct=True,
                data=forgotten,
            )
        return MemoryResult(
            found=False,
            answer=f"I couldn't find any memory matching '{target}'.",
            context=_ctx("long_term", [f"No memory found matching {target}."]),
            direct=True,
        )

    def _h_suppress(self, m, norm):
        return MemoryResult(
            found=True,
            answer="Understood. This conversation will stay off the record and won't be saved.",
            context="",
            direct=True,
            data={"suppress": True},
        )

    def _h_recall_all(self, m, norm):
        prof = self.data.get("identity", {}).get("profile", {})
        facts = self.data.get("long_term", [])
        parts = []
        if prof.get("name"):
            parts.append(f"your name is {prof['name']}")
        if prof.get("city"):
            parts.append(f"you live in {prof['city']}")
        if prof.get("job"):
            parts.append(f"you work as a {prof['job']}")
        for f in facts[:3]:
            parts.append(f.get("text", "").rstrip("."))
        if parts:
            ans = "I remember that " + ", and ".join(parts[:4]) + "."
        else:
            ans = "I don't have many memories saved yet."
        return MemoryResult(
            found=True,
            answer=ans,
            context=_ctx("identity", [ans]),
            direct=True,
            data={"profile": prof, "facts_count": len(facts)},
        )

    def _h_episode_recall(self, m, norm):
        episodes = self.db.get_recent_episodes(limit=1)
        if episodes:
            summary = episodes[0]["summary"]
            return MemoryResult(
                found=True,
                answer=f"Last time, we talked about: {summary}",
                context=_ctx("episode", [f"Previous episode summary: {summary}"]),
                direct=True,
                data=episodes[0],
            )
        recent = self.db.get_recent_conversation(limit=4)
        if recent:
            user_msg = next((t["content"] for t in reversed(recent) if t["role"] == "user"), None)
            if user_msg:
                return MemoryResult(
                    found=True,
                    answer=f"Earlier, you asked about: {user_msg}",
                    context=_ctx("conversation", [user_msg]),
                    direct=True,
                )
        return MemoryResult(
            found=False,
            answer="I don't have a record of our previous conversation yet.",
            context=_ctx("episode", ["No previous conversation found."]),
            direct=True,
        )

    def hybrid_search(self, query: str, top_k: int = 3, min_score: float = 0.35) -> list[tuple[float, str]]:
        """Hybrid vector + FTS5 retrieval."""
        if hasattr(self, "db"):
            return self.db.hybrid_search(query, embedder=self.embedder, top_k=top_k, min_score=min_score)
        return self.semantic_search(query, top_k=top_k, min_score=min_score)

    # ------------------------------------------------------------------ search
    def _docs(self, mtype: str) -> list[tuple[str, Any]]:
        with self._lock:
            if mtype == "long_term":
                return [(it["text"], it) for it in self.data["long_term"]]
            if mtype == "story":
                return [(t, {"text": t}) for t in self.data["story"]]
            if mtype == "files":
                return [(f"{f['name']}. {f.get('note', '')}".strip(), f) for f in self.data["identity"]["files"]]
        return []

    def _search(self, mtype: str, query: str, k: Optional[int] = None, lexical_first: bool = False):
        """Returns ([(score, (text, payload))...], used_embeddings)."""
        k = k or self.top_k
        docs = self._docs(mtype)
        if not docs:
            return [], False
        texts = [d[0] for d in docs]
        lex = np.array([_lex_score(query, t) for t in texts], dtype=np.float32)

        def pick(scores: np.ndarray, thr: float):
            order = np.argsort(-scores)[:k]
            return [(float(scores[i]), docs[i]) for i in order if scores[i] >= thr]

        if lexical_first and lex.max() >= 0.5:  # exact-ish name match: no model needed
            return pick(lex, 0.5), False

        if self.embedder.state == "idle":
            self.embedder.load() if self.block_first_use else self.embedder.load_async()
        if self.embedder.ready:
            try:
                mat = self._matrix(mtype, texts)
                q = self.embedder.embed([query], query=True)[0]
                return pick(mat @ q, self.min_score), True
            except Exception:  # noqa: BLE001
                log.exception("embedding search failed; falling back to keywords")
        return pick(lex, self.min_score_lexical), False

    def _matrix(self, mtype: str, texts: list[str]) -> np.ndarray:
        with self._idx_lock:
            tag = self.embedder.tag
            keys = [hashlib.sha1(f"{tag}|{t}".encode()).hexdigest() for t in texts]
            memo = self._memo.get(mtype)
            if memo and memo[0] == keys:
                return memo[1]
            missing = [(k, t) for k, t in zip(keys, texts) if k not in self._vec]
            if missing:
                vecs = self.embedder.embed([t for _, t in missing])
                for (k, _), v in zip(missing, vecs):
                    self._vec[k] = v
                self._dirty = True
            mat = np.stack([self._vec[k] for k in keys])
            self._memo[mtype] = (keys, mat)
            self._flush_cache()
            return mat

    def warmup(self, background: bool = True) -> None:
        """Load the embedding model + pre-embed everything. Call once at startup."""
        def _w():
            t0 = time.perf_counter()
            if self.embedder.load():
                for mt in ("long_term", "story", "files"):
                    texts = [d[0] for d in self._docs(mt)]
                    if texts:
                        self._matrix(mt, texts)
                self.embedder.embed(["warmup"], query=True)
            log.info("memory warm in %.0f ms", (time.perf_counter() - t0) * 1000)
        if background:
            threading.Thread(target=_w, daemon=True, name="memory-warmup").start()
        else:
            _w()

    # ----------------------------------------------------------------- storage
    def _load_store(self) -> dict:
        base = _empty_store()
        if hasattr(self, "db"):
            db_data = self.db.load_full_store()
            if db_data["identity"]["profile"] or db_data["long_term"]:
                return db_data

        if not os.path.exists(self.store_path):
            return base
        try:
            with open(self.store_path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            base["identity"].update(loaded.get("identity", {}))
            for k in ("profile", "friends", "files"):
                base["identity"].setdefault(k, {} if k == "profile" else [])
            base["story"] = loaded.get("story", [])
            base["long_term"] = loaded.get("long_term", [])
            if hasattr(self, "db"):
                self.db.import_from_dict(base)
        except Exception:  # noqa: BLE001
            log.exception("corrupt memory store; backing up and starting fresh")
            try:
                os.replace(self.store_path, self.store_path + ".bad")
            except OSError:
                pass
        return base

    def _save(self) -> None:
        d = os.path.dirname(os.path.abspath(self.store_path))
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, self.store_path)  # atomic: a crash never leaves a half-written store

    def _load_cache(self) -> None:
        if not os.path.exists(self.cache_path):
            return
        try:
            z = np.load(self.cache_path, allow_pickle=False)
            keys, vecs = z["keys"].tolist(), z["vecs"]
            if len(keys) == len(vecs):
                self._vec = dict(zip(keys, vecs))
        except Exception:  # noqa: BLE001
            log.warning("embedding cache unreadable; it will be rebuilt")

    def _flush_cache(self) -> None:
        if not self._dirty or not self._vec:
            return
        try:
            keys = list(self._vec)
            tmp = self.cache_path + ".tmp.npz"
            np.savez(tmp, keys=np.array(keys), vecs=np.stack([self._vec[k] for k in keys]).astype(np.float32))
            os.replace(tmp, self.cache_path)
            self._dirty = False
        except Exception:  # noqa: BLE001
            log.warning("could not write embedding cache")

    # ------------------------------------------------------------------ writers
    def set_profile(self, key: str, value: str) -> None:
        if key not in PROFILE_LABELS:
            raise ValueError(f"unknown profile key {key!r}; use one of {sorted(PROFILE_LABELS)}")
        with self._lock:
            self._profile()[key] = value
            if hasattr(self, "db"):
                self.db.set_profile(key, value)
            self._save()

    def add_friend(self, name: str, relation: str = "friend", note: str = "") -> None:
        """People list. relation containing 'friend' counts as a friend; others (sister, boss) don't."""
        with self._lock:
            people = self.data["identity"]["friends"]
            people[:] = [p for p in people if p["name"].lower() != name.lower()]
            people.append({"name": name, "relation": relation.lower(), "note": note})
            if hasattr(self, "db"):
                self.db.add_friend(name, relation, note)
            self._save()

    def remove_friend(self, name: str) -> None:
        with self._lock:
            p = self.data["identity"]["friends"]
            p[:] = [x for x in p if x["name"].lower() != name.lower()]
            if hasattr(self, "db"):
                self.db.remove_friend(name)
            self._save()

    def add_file(self, name: str, path: str = "", note: str = "") -> None:
        with self._lock:
            files = self.data["identity"]["files"]
            files[:] = [f for f in files if f["name"].lower() != name.lower()]
            files.append({"name": name, "path": path, "note": note})
            if hasattr(self, "db"):
                self.db.add_file(name, path, note)
            self._save()

    def add_story(self, text: str) -> int:
        chunks = _chunk(text)
        with self._lock:
            for c in chunks:
                if c not in self.data["story"]:
                    self.data["story"].append(c)
                    if hasattr(self, "db"):
                        self.db.add_story_chunk(c)
            self._save()
        return len(chunks)

    def add_long_term(self, text: str, tags: Optional[list[str]] = None) -> None:
        text = text.strip()
        with self._lock:
            if any(i.get("text") == text for i in self.data["long_term"]):
                return
            today_str = date.today().isoformat()
            mem_id = self.db.add_long_term(text, tags=tags) if hasattr(self, "db") else None
            self.data["long_term"].append({
                "id": mem_id or f"fact_{uuid.uuid4().hex[:8]}",
                "text": text,
                "date": today_str,
                "created_at": today_str,
                "updated_at": today_str,
                "access_count": 0,
                "tags": tags or [],
            })
            self._dirty = True
            self._memo.pop("long_term", None)
            self._save()

    def upsert_fact(self, text: str, category: str = "general") -> tuple[str, dict]:
        """
        Upsert a fact into long_term with deduplication and update tracking.
        Returns (action, item_dict) where action is 'ADD', 'UPDATE', or 'NONE'.
        """
        from .extractor import FactExtractor
        text = text.strip()
        if not text:
            return "NONE", {}

        with self._lock:
            if hasattr(self, "db"):
                action, item = self.db.upsert_fact(text, category=category)
                if action in ("ADD", "UPDATE"):
                    self.data["long_term"] = self.db.get_long_term()
                    self._dirty = True
                    self._memo.pop("long_term", None)
                    self._save()
                return action, item

            existing_texts = [it.get("text", "") for it in self.data["long_term"]]
            action, update_idx = FactExtractor.determine_action(text, existing_texts)

            if action == "NONE":
                matched = next((it for it in self.data["long_term"] if it.get("text", "").lower() == text.lower()), None)
                return "NONE", matched or {"text": text}

            today_str = date.today().isoformat()
            if action == "UPDATE" and update_idx is not None:
                old_item = self.data["long_term"][update_idx]
                old_item["text"] = text
                old_item["updated_at"] = today_str
                old_item["category"] = category
                self._dirty = True
                self._memo.pop("long_term", None)
                self._save()
                return "UPDATE", old_item

            # ADD new fact
            new_item = {
                "id": f"fact_{uuid.uuid4().hex[:8]}",
                "text": text,
                "category": category,
                "date": today_str,
                "created_at": today_str,
                "updated_at": today_str,
                "access_count": 0,
                "tags": [],
            }
            self.data["long_term"].append(new_item)
            self._dirty = True
            self._memo.pop("long_term", None)
            self._save()
            return "ADD", new_item

    def fts_search(self, query: str, limit: int = 5) -> list[dict]:
        """Full-text search powered by SQLite FTS5."""
        if hasattr(self, "db"):
            return self.db.fts_search(query, limit=limit)
        return []

    def get_user_summary(self) -> str:
        """Return compact 1-line user anchor for prompt baseline."""
        prof = self.data.get("identity", {}).get("profile", {})
        name = prof.get("name")
        if not name:
            return ""
        city = prof.get("city")
        if city:
            return f"User: {name} (Location: {city})"
        return f"User: {name}"

    def semantic_search(self, query: str, top_k: int = 2, min_score: float = 0.42) -> list[tuple[float, str]]:
        """
        Search long_term facts by vector similarity.
        Returns [(score, text), ...].
        """
        results: list[tuple[float, str]] = []
        hits, _ = self._search("long_term", query, k=top_k)
        for score, (text, payload) in hits:
            if score >= min_score:
                results.append((float(score), text))
        return results

    def format_context_for_prompt(self, hit: Optional[MemoryResult]) -> str:
        """
        Convert raw [MEMORY:...] context blocks into clean spoken companion prompt bullets.
        """
        if not hit or not hit.found:
            return ""
        lines = []
        for raw_line in hit.context.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("[MEMORY:"):
                continue
            if line.startswith("- "):
                line = line[2:]
            lines.append(f"- {line}")
        if not lines:
            return ""
        return (
            "Relevant facts about the user:\n"
            + "\n".join(lines)
            + "\n(Incorporate these facts naturally in conversation; do not recite technical labels)."
        )

    def background_extract(self, user_text: str, assistant_reply: str = "") -> list[dict]:
        """
        Extract facts from user utterance and upsert them in the background.
        """
        from .extractor import FactExtractor
        extracted = FactExtractor.extract_from_utterance(user_text)
        results = []
        for fact in extracted:
            action, item = self.upsert_fact(fact.text, category=fact.category)
            if action in ("ADD", "UPDATE"):
                log.info(f"Background memory {action}: '{fact.text}'")
                results.append(item)
        return results

    def stats(self) -> dict:
        d = self.data
        return {
            "profile_fields": len([v for v in d["identity"]["profile"].values() if v]),
            "people": len(d["identity"]["friends"]),
            "friends": len(self._friends_only()),
            "files": len(d["identity"]["files"]),
            "story_chunks": len(d["story"]),
            "long_term": len(d["long_term"]),
            "embedder": self.embedder.state,
            "embedder_tag": self.embedder.tag,
        }
