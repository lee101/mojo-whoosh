"""Transactional index writer."""

from __future__ import annotations

from .index import _analyze


class IndexWriter:
    def __init__(self, index):
        self.index = index
        self.schema = index.schema
        self._documents = [dict(document) for document in index._docs]
        self._closed = False

    def _check_open(self):
        if self._closed:
            raise RuntimeError("writer is closed")

    def add_document(self, **fields):
        self._check_open()
        unknown = set(fields).difference(self.schema.names())
        if unknown:
            raise KeyError(f"unknown field(s): {', '.join(sorted(unknown))}")
        self._documents.append(dict(fields))

    def update_document(self, **fields):
        self._check_open()
        unique = [
            name
            for name, value in fields.items()
            if name in self.schema and self.schema[name].unique
        ]
        if not unique:
            raise IndexError("None of the fields in the document are unique")
        self._documents = [
            document
            for document in self._documents
            if not any(document.get(name) == fields[name] for name in unique)
        ]
        self.add_document(**fields)

    def delete_document(self, docnum, delete=True):
        self._check_open()
        if docnum < 0 or docnum >= len(self._documents):
            raise IndexError(docnum)
        del self._documents[docnum]

    def delete_by_term(self, fieldname, text, searcher=None):
        self._check_open()
        if fieldname not in self.schema:
            raise KeyError(fieldname)
        field = self.schema[fieldname]
        needle = str(text)
        before = len(self._documents)
        self._documents = [
            document
            for document in self._documents
            if fieldname not in document
            or all(
                term != needle
                for term, _ in _analyze(field, document[fieldname])
            )
        ]
        return before - len(self._documents)

    def commit(self, mergetype=None, optimize=False, merge=True):
        self._check_open()
        with self.index._lock:
            previous_documents = self.index._docs
            self.index._docs = self._documents
            self.index._build_postings()
            try:
                self.index._save()
            except BaseException:
                self.index._docs = previous_documents
                self.index._build_postings()
                raise
            self.index._generation += 1
        self._closed = True

    def cancel(self):
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.commit()
        else:
            self.cancel()
