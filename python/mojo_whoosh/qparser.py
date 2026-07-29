"""A compact parser for fielded terms, phrases, wildcards, and Boolean syntax."""

from __future__ import annotations

import re

from . import query
from .index import _analyze

_LEX = re.compile(
    r'(?:\w+:)?"[^"]*"|\(|\)|\bAND\b|\bOR\b|\bNOT\b|[^\s()]+', re.I
)


class QueryParser:
    def __init__(self, fieldname, schema=None, plugins=None, group=None):
        self.fieldname = fieldname
        self.schema = schema
        self.plugins = plugins
        self.group = group

    def parse(self, input, normalize=True, debug=False):
        tokens = _LEX.findall(input)
        self._tokens = tokens
        self._position = 0
        if not tokens:
            return query.NullQuery
        return self._parse_or()

    def _peek(self):
        return (
            self._tokens[self._position]
            if self._position < len(self._tokens)
            else None
        )

    def _take(self):
        token = self._peek()
        self._position += 1
        return token

    def _parse_or(self):
        children = [self._parse_and()]
        while self._peek() and self._peek().upper() == "OR":
            self._take()
            children.append(self._parse_and())
        return children[0] if len(children) == 1 else query.Or(children)

    def _parse_and(self):
        children = [self._parse_unary()]
        while self._peek() is not None:
            token = self._peek().upper()
            if token in ("OR", ")"):
                break
            if token == "AND":
                self._take()
            children.append(self._parse_unary())
        return children[0] if len(children) == 1 else query.And(children)

    def _parse_unary(self):
        if self._peek() and self._peek().upper() == "NOT":
            self._take()
            return query.Not(self._parse_unary())
        if self._peek() == "(":
            self._take()
            result = self._parse_or()
            if self._peek() == ")":
                self._take()
            return result
        return self._term(self._take())

    def _term(self, token):
        fieldname = self.fieldname
        if ":" in token and not token.startswith('"'):
            candidate, token = token.split(":", 1)
            if self.schema is None or candidate in self.schema:
                fieldname = candidate
        if token == "*":
            return query.Every(fieldname)
        if token.startswith('"') and token.endswith('"'):
            words = self._words(fieldname, token[1:-1])
            if not words:
                return query.NullQuery
            return query.Phrase(fieldname, words)
        if "*" in token or "?" in token:
            normalized = token.lower() if self._is_text(fieldname) else token
            if normalized.endswith("*") and normalized.count("*") == 1 and "?" not in normalized:
                return query.Prefix(fieldname, normalized[:-1])
            return query.Wildcard(fieldname, normalized)
        words = self._words(fieldname, token)
        if not words:
            return query.NullQuery
        terms = [query.Term(fieldname, word) for word in words]
        return terms[0] if len(terms) == 1 else query.And(terms)

    def _is_text(self, fieldname):
        return self.schema is None or self.schema[fieldname].kind == "text"

    def _words(self, fieldname, text):
        if self.schema is None:
            return [word.lower() for word in re.findall(r"\w+(?:\.?\w+)*", text)]
        return [term for term, _ in _analyze(self.schema[fieldname], text)]


class MultifieldParser(QueryParser):
    def __init__(self, fieldnames, schema, fieldboosts=None, **kwargs):
        super().__init__(fieldnames[0], schema, **kwargs)
        self.fieldnames = list(fieldnames)
        self.fieldboosts = fieldboosts or {}

    def _term(self, token):
        if ":" in token:
            return super()._term(token)
        children = []
        for fieldname in self.fieldnames:
            original = self.fieldname
            self.fieldname = fieldname
            child = super()._term(token)
            self.fieldname = original
            child.boost = self.fieldboosts.get(fieldname, 1.0)
            children.append(child)
        return query.Or(children)
