"""Persistent inverted indexes with the familiar :mod:`whoosh.index` API."""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .fields import Schema

_TOKEN_RE = re.compile(r"\w+(?:\.?\w+)*", re.UNICODE)
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "for",
    "from",
    "have",
    "if",
    "in",
    "is",
    "it",
    "may",
    "not",
    "of",
    "on",
    "or",
    "tbd",
    "that",
    "the",
    "this",
    "to",
    "us",
    "we",
    "when",
    "will",
    "with",
    "yet",
    "you",
    "your",
}


def _analyze(field, value) -> list[tuple[str, int]]:
    if field.kind == "stored":
        return []
    if field.kind in ("id", "numeric"):
        return [(str(value), 0)]
    text = " ".join(map(str, value)) if isinstance(value, (list, tuple)) else str(value)
    if getattr(field, "analyzer", None) is not None:
        result = []
        for i, token in enumerate(field.analyzer(text)):
            result.append((str(getattr(token, "text", token)), getattr(token, "pos", i)))
        return result
    result = []
    for match in _TOKEN_RE.finditer(text):
        term = match.group(0).lower()
        if len(term) >= 2 and term not in _STOP_WORDS:
            result.append((term, len(result)))
    return result


@dataclass
class Posting:
    docs: np.ndarray
    frequencies: np.ndarray
    positions: dict[int, tuple[int, ...]]


class Index:
    def __init__(self, dirname: str, schema: Schema, docs: list[dict], indexname="MAIN"):
        self.dirname = os.path.abspath(dirname)
        self.indexname = indexname
        self.schema = schema
        self._docs = docs
        self._lock = threading.RLock()
        self._generation = 0
        self._build_postings()

    @property
    def _path(self):
        return os.path.join(self.dirname, f"{self.indexname}.mwhoosh.json")

    def _build_postings(self):
        raw = defaultdict(lambda: defaultdict(list))
        lengths = {
            name: np.zeros(len(self._docs), dtype=np.int64)
            for name, field in self.schema._fields.items()
            if field.kind != "stored"
        }
        for docnum, document in enumerate(self._docs):
            for fieldname, field in self.schema._fields.items():
                if fieldname not in document or field.kind == "stored":
                    continue
                tokens = _analyze(field, document[fieldname])
                lengths[fieldname][docnum] = len(tokens)
                for term, position in tokens:
                    raw[(fieldname, term)][docnum].append(position)
        self._postings = {}
        for key, doc_positions in raw.items():
            docs = np.fromiter(sorted(doc_positions), dtype=np.int64)
            frequencies = np.fromiter(
                (len(doc_positions[int(doc)]) for doc in docs), dtype=np.float64
            )
            self._postings[key] = Posting(
                docs,
                frequencies,
                {doc: tuple(pos) for doc, pos in doc_positions.items()},
            )
        self._lengths = lengths
        self._average_lengths = {
            fieldname: float(values.sum() / max(len(values), 1))
            for fieldname, values in lengths.items()
        }
        self._terms = defaultdict(list)
        for fieldname, term in self._postings:
            self._terms[fieldname].append(term)
        for terms in self._terms.values():
            terms.sort()

    def _save(self):
        os.makedirs(self.dirname, exist_ok=True)
        payload = {"schema": self.schema.descriptor(), "documents": self._docs}
        fd, temporary = tempfile.mkstemp(
            prefix=f".{self.indexname}.", suffix=".tmp", dir=self.dirname
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def searcher(self, weighting=None, **kwargs):
        from .searching import Searcher

        return Searcher(self, weighting=weighting, **kwargs)

    def writer(
        self,
        procs=1,
        limitmb=128,
        blocklimit=128,
        timeout=0.0,
        delay=0.1,
        _lk=True,
        **kwargs,
    ):
        from .writing import IndexWriter

        return IndexWriter(self)

    def reader(self, reuse=None):
        return self.searcher()

    def doc_count(self):
        return len(self._docs)

    def doc_count_all(self):
        return len(self._docs)

    def is_empty(self):
        return not self._docs

    def latest_generation(self):
        return self._generation

    def refresh(self):
        fresh = open_dir(self.dirname, self.indexname)
        self._docs = fresh._docs
        self.schema = fresh.schema
        self._build_postings()
        return self

    def close(self):
        pass


def create_in(dirname, schema, indexname="MAIN", indexclass=None):
    os.makedirs(dirname, exist_ok=True)
    index = Index(dirname, schema, [], indexname)
    index._save()
    return index


def open_dir(dirname, indexname="MAIN", readonly=False, schema=None, **kwargs):
    path = os.path.join(os.path.abspath(dirname), f"{indexname}.mwhoosh.json")
    if not os.path.exists(path):
        raise OSError(f"index {indexname!r} does not exist in {dirname!r}")
    with open(path, encoding="utf-8") as stream:
        payload = json.load(stream)
    loaded_schema = schema or Schema.from_descriptor(payload["schema"])
    return Index(dirname, loaded_schema, payload["documents"], indexname)


def exists_in(dirname, indexname="MAIN"):
    return os.path.exists(
        os.path.join(os.path.abspath(dirname), f"{indexname}.mwhoosh.json")
    )
