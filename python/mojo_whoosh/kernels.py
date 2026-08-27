"""Public NumPy wrappers over Mojo's posting-list and ranking kernels."""

from __future__ import annotations

import operator

import numpy as np

from ._lib import addr, lib

BM25_PARALLEL_THRESHOLD = 4_194_304


def _vector(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _i64(values, name: str = "values") -> np.ndarray:
    array = _vector(values, name)
    if array.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integers")
    if array.size:
        info = np.iinfo(np.int64)
        if array.min() < info.min or array.max() > info.max:
            raise OverflowError(f"{name} contains a value outside the int64 range")
    return np.ascontiguousarray(array, dtype=np.int64)


def _f64(values, name: str = "values") -> np.ndarray:
    array = _vector(values, name)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numbers")
    if array.dtype.kind in "iu" and array.size:
        exact_limit = 1 << 53
        if array.min() < -exact_limit or array.max() > exact_limit:
            raise OverflowError(
                f"{name} contains an integer that float64 cannot represent exactly"
            )
    if array.dtype.kind == "f" and array.dtype.itemsize > np.dtype(np.float64).itemsize:
        raise TypeError(f"{name} cannot be narrowed to float64")
    return np.ascontiguousarray(array, dtype=np.float64)


def _aligned(docs: np.ndarray, freqs: np.ndarray) -> None:
    if docs.size != freqs.size:
        raise ValueError("docids and frequencies must have the same length")


def _validate_docids(docs: np.ndarray, ndocs: int) -> None:
    if docs.size and (docs.min() < 0 or docs.max() >= ndocs):
        raise ValueError("docids must index the supplied document array")


def intersect(a, b) -> np.ndarray:
    """Return the intersection of two sorted, unique int64 posting lists."""
    left, right = _i64(a, "a"), _i64(b, "b")
    if not left.size or not right.size:
        return np.empty(0, dtype=np.int64)
    result = np.empty(min(left.size, right.size), dtype=np.int64)
    n = lib().mw_intersect(
        addr(left), left.size, addr(right), right.size, addr(result)
    )
    return result[:n]


def union(a, b) -> np.ndarray:
    """Return the union of two sorted, unique int64 posting lists."""
    left, right = _i64(a, "a"), _i64(b, "b")
    if not left.size:
        return right.copy()
    if not right.size:
        return left.copy()
    result = np.empty(left.size + right.size, dtype=np.int64)
    n = lib().mw_union(addr(left), left.size, addr(right), right.size, addr(result))
    return result[:n]


def difference(a, b) -> np.ndarray:
    """Return members of sorted posting list *a* which are absent from *b*."""
    left, right = _i64(a, "a"), _i64(b, "b")
    if not left.size or not right.size:
        return left.copy()
    result = np.empty(left.size, dtype=np.int64)
    n = lib().mw_difference(
        addr(left), left.size, addr(right), right.size, addr(result)
    )
    return result[:n]


def accumulate_bm25(
    docids,
    frequencies,
    field_lengths,
    scores,
    *,
    avg_length: float,
    B: float = 0.75,
    K1: float = 1.2,
    boost: float = 1.0,
) -> np.ndarray:
    """Add one term's Whoosh BM25F contribution into a dense score array."""
    docs = _i64(docids, "docids")
    freqs = _f64(frequencies, "frequencies")
    lengths = _i64(field_lengths, "field_lengths")
    target = _f64(scores, "scores")
    _aligned(docs, freqs)
    if lengths.size < target.size:
        raise ValueError("field_lengths must cover every score document")
    _validate_docids(docs, target.size)
    if docs.size:
        lib().mw_bm25_accumulate(
            addr(docs),
            addr(freqs),
            addr(lengths),
            docs.size,
            target.size,
            avg_length,
            B,
            K1,
            boost,
            addr(target),
        )
    return target


def score_bm25(
    docids,
    frequencies,
    field_lengths,
    *,
    avg_length: float,
    B: float = 0.75,
    K1: float = 1.2,
    boost: float = 1.0,
) -> np.ndarray:
    """Return BM25F scores aligned with one term's posting list."""
    docs = _i64(docids, "docids")
    freqs = _f64(frequencies, "frequencies")
    lengths = _i64(field_lengths, "field_lengths")
    _aligned(docs, freqs)
    _validate_docids(docs, lengths.size)
    if not docs.size:
        return np.empty(0, dtype=np.float64)
    values = np.empty(docs.size, dtype=np.float64)
    lib().mw_bm25_scores(
        addr(docs),
        addr(freqs),
        addr(lengths),
        docs.size,
        lengths.size,
        avg_length,
        B,
        K1,
        boost,
        addr(values),
    )
    return values


def _bm25_top_k_arrays(
    docs: np.ndarray,
    freqs: np.ndarray,
    lengths: np.ndarray,
    limit: int,
    *,
    avg_length: float,
    B: float,
    K1: float,
    boost: float,
) -> tuple[np.ndarray, np.ndarray]:
    k = min(max(limit, 0), docs.size)
    if not k:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    result_docs = np.empty(k, dtype=np.int64)
    result_scores = np.empty(k, dtype=np.float64)
    n = lib().mw_bm25_topk(
        addr(docs),
        addr(freqs),
        addr(lengths),
        docs.size,
        lengths.size,
        avg_length,
        B,
        K1,
        boost,
        k,
        addr(result_docs),
        addr(result_scores),
    )
    return result_docs[:n], result_scores[:n]


def bm25_top_k(
    docids,
    frequencies,
    field_lengths,
    limit: int,
    *,
    avg_length: float,
    B: float = 0.75,
    K1: float = 1.2,
    boost: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Score one term and return its best documents without a dense score buffer."""
    docs = _i64(docids, "docids")
    freqs = _f64(frequencies, "frequencies")
    lengths = _i64(field_lengths, "field_lengths")
    _aligned(docs, freqs)
    _validate_docids(docs, lengths.size)
    try:
        requested = operator.index(limit)
    except TypeError as error:
        raise TypeError("limit must be an integer") from error
    return _bm25_top_k_arrays(
        docs,
        freqs,
        lengths,
        requested,
        avg_length=avg_length,
        B=B,
        K1=K1,
        boost=boost,
    )


def accumulate_tfidf(docids, frequencies, scores, *, boost: float = 1.0):
    docs = _i64(docids, "docids")
    freqs = _f64(frequencies, "frequencies")
    target = _f64(scores, "scores")
    _aligned(docs, freqs)
    _validate_docids(docs, target.size)
    if docs.size:
        lib().mw_tfidf_accumulate(
            addr(docs), addr(freqs), docs.size, target.size, boost, addr(target)
        )
    return target


def accumulate_frequency(docids, frequencies, scores, *, boost: float = 1.0):
    docs = _i64(docids, "docids")
    freqs = _f64(frequencies, "frequencies")
    target = _f64(scores, "scores")
    _aligned(docs, freqs)
    _validate_docids(docs, target.size)
    if docs.size:
        lib().mw_frequency_accumulate(
            addr(docs), addr(freqs), docs.size, target.size, boost, addr(target)
        )
    return target


def top_k(docids, scores, limit: int) -> tuple[np.ndarray, np.ndarray]:
    """Rank aligned scores descending, breaking ties by ascending document ID."""
    docs = _i64(docids, "docids")
    values = _f64(scores, "scores")
    if docs.size != values.size:
        raise ValueError("docids and scores must have the same length")
    try:
        requested = operator.index(limit)
    except TypeError as error:
        raise TypeError("limit must be an integer") from error
    k = min(max(requested, 0), docs.size)
    if not k:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    result_docs = np.empty(k, dtype=np.int64)
    result_scores = np.empty(k, dtype=np.float64)
    n = lib().mw_topk(
        addr(docs),
        addr(values),
        docs.size,
        k,
        addr(result_docs),
        addr(result_scores),
    )
    return result_docs[:n], result_scores[:n]
