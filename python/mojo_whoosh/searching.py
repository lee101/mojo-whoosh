"""Query evaluation and result objects."""

from __future__ import annotations

import fnmatch
import operator
import time
from collections import Counter
from collections.abc import Sequence

import numpy as np

from . import query, scoring
from .kernels import (
    _bm25_top_k_arrays,
    accumulate_bm25,
    accumulate_frequency,
    accumulate_tfidf,
    difference,
    intersect,
    score_bm25,
    top_k,
    union,
)


class Hit(dict):
    def __init__(self, searcher, docnum, score, rank, matched_terms=()):
        super().__init__(searcher.stored_fields(docnum))
        self.searcher = searcher
        self.docnum = int(docnum)
        self.score = float(score)
        self.rank = rank
        self._matched_terms = tuple(matched_terms)

    def fields(self):
        return dict(self)

    def matched_terms(self):
        return list(self._matched_terms)


class Results(Sequence):
    def __init__(
        self,
        searcher,
        query_object,
        docids,
        scores,
        runtime=0.0,
        total=None,
        terms=False,
    ):
        self.searcher = searcher
        self.q = query_object
        self.runtime = runtime
        self._docids = np.asarray(docids, dtype=np.int64)
        self._scores = np.asarray(scores, dtype=np.float64)
        self._total = len(docids) if total is None else int(total)
        self._terms = terms

    def __len__(self):
        return len(self._docids)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [
                self._hit(i)
                for i in range(*index.indices(len(self._docids)))
            ]
        if index < 0:
            index += len(self._docids)
        return self._hit(index)

    def _hit(self, rank):
        docnum = int(self._docids[rank])
        matched = self.searcher._matched_terms(self.q, docnum) if self._terms else ()
        return Hit(self.searcher, docnum, self._scores[rank], rank, matched)

    def scored_length(self):
        return len(self)

    def estimated_length(self):
        return self._total

    def estimated_min_length(self):
        return self._total

    def has_exact_length(self):
        return True

    def is_empty(self):
        return self._total == 0

    def fields(self, n):
        return self[n].fields()

    def docnum(self, n):
        return int(self._docids[n])

    def score(self, n):
        return float(self._scores[n])

    def items(self):
        return list(self)


class Searcher:
    def __init__(
        self,
        index,
        weighting=None,
        closereader=True,
        fromindex=None,
        parent=None,
    ):
        self.ix = index
        self.schema = index.schema
        self.weighting = weighting or scoring.BM25F()
        self._closed = False
        self._all = np.arange(len(index._docs), dtype=np.int64)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self._closed = True

    def _posting(self, fieldname, term):
        return self.ix._postings.get((fieldname, str(term)))

    def _expanded(self, query_object):
        if isinstance(query_object, query.Prefix):
            return [
                term
                for term in self.ix._terms.get(query_object.fieldname, ())
                if term.startswith(str(query_object.text))
            ]
        if isinstance(query_object, query.Wildcard):
            return [
                term
                for term in self.ix._terms.get(query_object.fieldname, ())
                if fnmatch.fnmatchcase(term, str(query_object.text))
            ]
        if isinstance(query_object, query.TermRange):
            terms = []
            for term in self.ix._terms.get(query_object.fieldname, ()):
                lower = query_object.start is None or (
                    term > str(query_object.start)
                    if query_object.startexcl
                    else term >= str(query_object.start)
                )
                upper = query_object.end is None or (
                    term < str(query_object.end)
                    if query_object.endexcl
                    else term <= str(query_object.end)
                )
                if lower and upper:
                    terms.append(term)
            return terms
        return []

    def _docs_for(self, query_object):
        if query_object is query.NullQuery:
            return np.empty(0, dtype=np.int64)
        if isinstance(query_object, query.Term):
            posting = self._posting(query_object.fieldname, query_object.text)
            return posting.docs if posting else np.empty(0, dtype=np.int64)
        if isinstance(query_object, query.Every):
            if query_object.fieldname is None:
                return self._all
            lists = [
                posting.docs
                for (fieldname, _), posting in self.ix._postings.items()
                if fieldname == query_object.fieldname
            ]
            result = np.empty(0, dtype=np.int64)
            for docs in lists:
                result = union(result, docs)
            return result
        if isinstance(query_object, query.And):
            if not query_object.subqueries:
                return np.empty(0, dtype=np.int64)
            result = self._docs_for(query_object.subqueries[0])
            for child in query_object.subqueries[1:]:
                if isinstance(child, query.Not):
                    result = difference(result, self._docs_for(child.query))
                else:
                    result = intersect(result, self._docs_for(child))
                if not result.size:
                    break
            return result
        if isinstance(query_object, query.Or):
            lists = [self._docs_for(child) for child in query_object.subqueries]
            required = int(query_object.minmatch or 1)
            if required <= 1:
                result = np.empty(0, dtype=np.int64)
                for docs in lists:
                    result = union(result, docs)
                return result
            counts = Counter(int(doc) for docs in lists for doc in docs)
            return np.fromiter(
                sorted(doc for doc, count in counts.items() if count >= required),
                dtype=np.int64,
            )
        if isinstance(query_object, query.Not):
            return difference(self._all, self._docs_for(query_object.query))
        if isinstance(query_object, query.AndNot):
            return difference(
                self._docs_for(query_object.positive),
                self._docs_for(query_object.negative),
            )
        if isinstance(query_object, query.Phrase):
            postings = [
                self._posting(query_object.fieldname, word)
                for word in query_object.words
            ]
            if not postings or any(posting is None for posting in postings):
                return np.empty(0, dtype=np.int64)
            candidates = postings[0].docs
            for posting in postings[1:]:
                candidates = intersect(candidates, posting.docs)
            accepted = [
                int(doc)
                for doc in candidates
                if self._phrase_matches(postings, int(doc), query_object.slop)
            ]
            return np.asarray(accepted, dtype=np.int64)
        if isinstance(
            query_object, (query.Prefix, query.Wildcard, query.TermRange)
        ):
            result = np.empty(0, dtype=np.int64)
            for term in self._expanded(query_object):
                result = union(
                    result, self._posting(query_object.fieldname, term).docs
                )
            return result
        raise TypeError(f"unsupported query type: {type(query_object).__name__}")

    @staticmethod
    def _phrase_matches(postings, docnum, slop):
        positions = [posting.positions[docnum] for posting in postings]
        frontier = set(positions[0])
        for next_positions in positions[1:]:
            frontier = {
                nxt
                for previous in frontier
                for nxt in next_positions
                if 0 < nxt - previous <= slop
            }
            if not frontier:
                return False
        return True

    def _scoring_terms(self, query_object, multiplier=1.0):
        boost = multiplier * float(getattr(query_object, "boost", 1.0))
        if isinstance(query_object, query.Term):
            return [(query_object.fieldname, str(query_object.text), boost)]
        if isinstance(query_object, (query.And, query.Or)):
            return [
                term
                for child in query_object.subqueries
                if not isinstance(child, query.Not)
                for term in self._scoring_terms(child, boost)
            ]
        if isinstance(query_object, query.AndNot):
            return self._scoring_terms(query_object.positive, boost)
        if isinstance(query_object, query.Phrase):
            return [
                (query_object.fieldname, str(word), boost)
                for word in query_object.words
            ]
        if isinstance(
            query_object, (query.Prefix, query.Wildcard, query.TermRange)
        ) and not query_object.constantscore:
            return [
                (query_object.fieldname, term, boost)
                for term in self._expanded(query_object)
            ]
        return []

    def _model_for(self, fieldname):
        if isinstance(self.weighting, scoring.MultiWeighting):
            return self.weighting.weighting(fieldname)
        return self.weighting

    def _score(self, query_object, candidates):
        scores = np.zeros(len(self.ix._docs), dtype=np.float64)
        terms = self._scoring_terms(query_object)
        for fieldname, term, boost in terms:
            posting = self._posting(fieldname, term)
            if posting is None:
                continue
            field = self.schema[fieldname]
            term_boost = boost * field.field_boost
            model = self._model_for(fieldname)
            if isinstance(model, scoring.BM25F):
                lengths = self.ix._lengths[fieldname]
                accumulate_bm25(
                    posting.docs,
                    posting.frequencies,
                    lengths,
                    scores,
                    avg_length=self.ix._average_lengths[fieldname],
                    B=model.field_B(fieldname),
                    K1=model.K1,
                    boost=term_boost,
                )
            elif isinstance(model, scoring.TF_IDF):
                accumulate_tfidf(
                    posting.docs, posting.frequencies, scores, boost=term_boost
                )
            else:
                accumulate_frequency(
                    posting.docs, posting.frequencies, scores, boost=term_boost
                )
        if not terms and candidates.size:
            scores[candidates] = float(getattr(query_object, "boost", 1.0))
        return scores

    def _score_bm25_term(self, query_object):
        posting = self._posting(query_object.fieldname, query_object.text)
        if posting is None:
            return np.empty(0, dtype=np.float64)
        fieldname = query_object.fieldname
        model = self._model_for(fieldname)
        return score_bm25(
            posting.docs,
            posting.frequencies,
            self.ix._lengths[fieldname],
            avg_length=self.ix._average_lengths[fieldname],
            B=model.field_B(fieldname),
            K1=model.K1,
            boost=query_object.boost * self.schema[fieldname].field_boost,
        )

    def _top_bm25_term(self, query_object, limit):
        posting = self._posting(query_object.fieldname, query_object.text)
        if posting is None:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        fieldname = query_object.fieldname
        model = self._model_for(fieldname)
        return _bm25_top_k_arrays(
            posting.docs,
            posting.frequencies,
            self.ix._lengths[fieldname],
            limit,
            avg_length=self.ix._average_lengths[fieldname],
            B=model.field_B(fieldname),
            K1=model.K1,
            boost=query_object.boost * self.schema[fieldname].field_boost,
        )

    def search(
        self,
        q,
        limit=10,
        sortedby=None,
        reverse=False,
        groupedby=None,
        collapse=None,
        collapse_limit=1,
        terms=False,
        maptype=None,
        optimize=True,
        scored=True,
        filter=None,
        mask=None,
    ):
        started = time.perf_counter()
        candidates = self._docs_for(q)
        if filter is not None:
            candidates = intersect(candidates, self._docs_for(filter))
        if mask is not None:
            candidates = difference(candidates, self._docs_for(mask))
        direct_bm25 = (
            scored
            and filter is None
            and mask is None
            and isinstance(q, query.Term)
            and isinstance(self._model_for(q.fieldname), scoring.BM25F)
        )
        fused_bm25 = direct_bm25 and sortedby is None and limit is not None
        if fused_bm25:
            try:
                requested = operator.index(limit)
            except TypeError as error:
                raise TypeError("limit must be an integer") from error
            docids, result_scores = self._top_bm25_term(q, requested)
            if reverse:
                docids, result_scores = docids[::-1], result_scores[::-1]
        elif direct_bm25:
            aligned = self._score_bm25_term(q)
        elif scored:
            aligned = self._score(q, candidates)[candidates]
        else:
            aligned = np.zeros(candidates.size, dtype=np.float64)
        total = candidates.size
        if fused_bm25:
            pass
        elif sortedby is not None:
            fieldname = str(sortedby)
            order = sorted(
                range(total),
                key=lambda i: self.ix._docs[int(candidates[i])].get(fieldname),
                reverse=reverse,
            )
            if limit is not None:
                order = order[:limit]
            docids = candidates[order]
            result_scores = aligned[order]
        elif limit is None:
            order = np.lexsort((candidates, -aligned))
            docids, result_scores = candidates[order], aligned[order]
        else:
            docids, result_scores = top_k(candidates, aligned, limit)
            if reverse:
                docids, result_scores = docids[::-1], result_scores[::-1]
        return Results(
            self,
            q,
            docids,
            result_scores,
            runtime=time.perf_counter() - started,
            total=total,
            terms=terms,
        )

    def stored_fields(self, docnum):
        document = self.ix._docs[int(docnum)]
        return {
            name: document[name]
            for name in self.schema.stored_names()
            if name in document
        }

    def all_stored_fields(self):
        for docnum in range(len(self.ix._docs)):
            yield self.stored_fields(docnum)

    def document(self, **kw):
        for docnum in self.document_numbers(**kw):
            return self.stored_fields(docnum)
        return None

    def documents(self, **kw):
        for docnum in self.document_numbers(**kw):
            yield self.stored_fields(docnum)

    def document_number(self, **kw):
        for docnum in self.document_numbers(**kw):
            return docnum
        return None

    def document_numbers(self, **kw):
        for docnum, document in enumerate(self.ix._docs):
            if all(str(document.get(name)) == str(value) for name, value in kw.items()):
                yield docnum

    def doc_count_all(self):
        return len(self.ix._docs)

    def doc_count(self):
        return len(self.ix._docs)

    def field_length(self, fieldname):
        return int(self.ix._lengths[fieldname].sum())

    def avg_field_length(self, fieldname, default=None):
        lengths = self.ix._lengths[fieldname]
        if not lengths.size:
            return default
        return float(lengths.sum() / lengths.size)

    def doc_field_length(self, docnum, fieldname, default=0):
        lengths = self.ix._lengths.get(fieldname)
        return default if lengths is None else int(lengths[int(docnum)])

    def frequency(self, fieldname, text):
        posting = self._posting(fieldname, text)
        return 0.0 if posting is None else float(posting.frequencies.sum())

    def doc_frequency(self, fieldname, text):
        posting = self._posting(fieldname, text)
        return 0 if posting is None else int(posting.docs.size)

    def lexicon(self, fieldname):
        for term in self.ix._terms.get(fieldname, ()):
            yield term.encode("utf-8")

    def expand_prefix(self, fieldname, prefix):
        text = prefix.decode() if isinstance(prefix, bytes) else str(prefix)
        for term in self.ix._terms.get(fieldname, ()):
            if term.startswith(text):
                yield term.encode("utf-8")

    def _matched_terms(self, query_object, docnum):
        return [
            (fieldname, term.encode("utf-8"))
            for fieldname, term, _ in self._scoring_terms(query_object)
            if (posting := self._posting(fieldname, term)) is not None
            and docnum in posting.positions
        ]
