"""Sorted posting-list, scoring, and ranking kernels exposed through a C ABI."""

from max.algorithm import parallelize
from std.math import log
from std.sys import simd_width_of

comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()
comptime BM25_PARALLEL_THRESHOLD = 4_194_304


def _worse(score_a: Float64, doc_a: Int64, score_b: Float64, doc_b: Int64) -> Bool:
    return score_a < score_b or (score_a == score_b and doc_a > doc_b)


def _swap_ranked(docs: IPtr, scores: FPtr, a: Int, b: Int):
    var doc = docs[a]
    var score = scores[a]
    docs[a] = docs[b]
    scores[a] = scores[b]
    docs[b] = doc
    scores[b] = score


def _sift_down(docs: IPtr, scores: FPtr, start: Int, size: Int):
    var parent = start
    while True:
        var left = parent * 2 + 1
        if left >= size:
            break
        var child = left
        var right = left + 1
        if right < size and _worse(
            scores[right], docs[right], scores[left], docs[left]
        ):
            child = right
        if not _worse(
            scores[child], docs[child], scores[parent], docs[parent]
        ):
            break
        _swap_ranked(docs, scores, parent, child)
        parent = child


def _topk_insert(
    docs: IPtr,
    scores: FPtr,
    used: Int,
    capacity: Int,
    doc: Int64,
    score: Float64,
) -> Int:
    if used < capacity:
        docs[used] = doc
        scores[used] = score
        var child = used
        var new_used = used + 1
        while child > 0:
            var parent = (child - 1) // 2
            if not _worse(
                scores[child], docs[child], scores[parent], docs[parent]
            ):
                break
            _swap_ranked(docs, scores, child, parent)
            child = parent
        return new_used
    if score > scores[0] or (score == scores[0] and doc < docs[0]):
        docs[0] = doc
        scores[0] = score
        _sift_down(docs, scores, 0, used)
    return used


def _bm25_scores_range(
    docs: IPtr,
    freqs: FPtr,
    lengths: IPtr,
    dst: FPtr,
    start: Int,
    stop: Int,
    safe_avg: Float64,
    b_param: Float64,
    k1: Float64,
    scale: Float64,
):
    var simd_stop = stop - (stop - start) % W
    for i in range(start, simd_stop, W):
        var doc_vec = docs.load[width=W](i)
        var weights = freqs.load[width=W](i)
        var field_lengths = lengths.gather[width=W](doc_vec).cast[DType.float64]()
        var norms = 1.0 - b_param + b_param * field_lengths / safe_avg
        var values = scale * weights / (weights + k1 * norms)
        dst.store(i, values)
    for i in range(simd_stop, stop):
        var doc = Int(docs[i])
        var weight = freqs[i]
        var norm = 1.0 - b_param + b_param * Float64(lengths[doc]) / safe_avg
        dst[i] = scale * weight / (weight + k1 * norm)


@export("mw_intersect")
def mw_intersect(
    a_addr: Int, na: Int, b_addr: Int, nb: Int, dst_addr: Int
) abi("C") -> Int:
    var a = IPtr(unsafe_from_address=a_addr)
    var b = IPtr(unsafe_from_address=b_addr)
    var dst = IPtr(unsafe_from_address=dst_addr)
    var i = 0
    var j = 0
    var n = 0
    while i < na and j < nb:
        if a[i] < b[j]:
            i += 1
        elif b[j] < a[i]:
            j += 1
        else:
            dst[n] = a[i]
            n += 1
            i += 1
            j += 1
    return n


@export("mw_union")
def mw_union(
    a_addr: Int, na: Int, b_addr: Int, nb: Int, dst_addr: Int
) abi("C") -> Int:
    var a = IPtr(unsafe_from_address=a_addr)
    var b = IPtr(unsafe_from_address=b_addr)
    var dst = IPtr(unsafe_from_address=dst_addr)
    var i = 0
    var j = 0
    var n = 0
    while i < na and j < nb:
        if a[i] < b[j]:
            dst[n] = a[i]
            i += 1
        elif b[j] < a[i]:
            dst[n] = b[j]
            j += 1
        else:
            dst[n] = a[i]
            i += 1
            j += 1
        n += 1
    while i < na:
        dst[n] = a[i]
        i += 1
        n += 1
    while j < nb:
        dst[n] = b[j]
        j += 1
        n += 1
    return n


@export("mw_difference")
def mw_difference(
    a_addr: Int, na: Int, b_addr: Int, nb: Int, dst_addr: Int
) abi("C") -> Int:
    var a = IPtr(unsafe_from_address=a_addr)
    var b = IPtr(unsafe_from_address=b_addr)
    var dst = IPtr(unsafe_from_address=dst_addr)
    var i = 0
    var j = 0
    var n = 0
    while i < na and j < nb:
        if a[i] < b[j]:
            dst[n] = a[i]
            n += 1
            i += 1
        elif b[j] < a[i]:
            j += 1
        else:
            i += 1
            j += 1
    while i < na:
        dst[n] = a[i]
        n += 1
        i += 1
    return n


@export("mw_bm25_accumulate")
def mw_bm25_accumulate(
    docs_addr: Int,
    freqs_addr: Int,
    lengths_addr: Int,
    npost: Int,
    ndocs: Int,
    avg_length: Float64,
    b_param: Float64,
    k1: Float64,
    boost: Float64,
    scores_addr: Int,
) abi("C"):
    var docs = IPtr(unsafe_from_address=docs_addr)
    var freqs = FPtr(unsafe_from_address=freqs_addr)
    var lengths = IPtr(unsafe_from_address=lengths_addr)
    var scores = FPtr(unsafe_from_address=scores_addr)
    var denom_docs = Float64(npost + 1)
    var idf = log(Float64(ndocs) / denom_docs) + 1.0
    var safe_avg = avg_length if avg_length > 0.0 else 1.0
    for i in range(npost):
        var doc = Int(docs[i])
        if doc >= 0 and doc < ndocs:
            var weight = freqs[i]
            var norm = 1.0 - b_param + b_param * Float64(lengths[doc]) / safe_avg
            scores[doc] += boost * idf * weight * (k1 + 1.0) / (weight + k1 * norm)


@export("mw_bm25_scores")
def mw_bm25_scores(
    docs_addr: Int,
    freqs_addr: Int,
    lengths_addr: Int,
    npost: Int,
    ndocs: Int,
    avg_length: Float64,
    b_param: Float64,
    k1: Float64,
    boost: Float64,
    dst_addr: Int,
) abi("C"):
    var docs = IPtr(unsafe_from_address=docs_addr)
    var freqs = FPtr(unsafe_from_address=freqs_addr)
    var lengths = IPtr(unsafe_from_address=lengths_addr)
    var dst = FPtr(unsafe_from_address=dst_addr)
    var denom_docs = Float64(npost + 1)
    var idf = log(Float64(ndocs) / denom_docs) + 1.0
    var safe_avg = avg_length if avg_length > 0.0 else 1.0
    var scale = boost * idf * (k1 + 1.0)
    if npost >= BM25_PARALLEL_THRESHOLD:
        var workers = 16

        @parameter
        @__copy_capture(
            docs,
            freqs,
            lengths,
            dst,
            npost,
            workers,
            safe_avg,
            b_param,
            k1,
            scale,
        )
        def work(worker: Int):
            var start = npost * worker // workers
            var stop = npost * (worker + 1) // workers
            _bm25_scores_range(
                docs,
                freqs,
                lengths,
                dst,
                start,
                stop,
                safe_avg,
                b_param,
                k1,
                scale,
            )

        parallelize[work](workers, workers)
    else:
        _bm25_scores_range(
            docs,
            freqs,
            lengths,
            dst,
            0,
            npost,
            safe_avg,
            b_param,
            k1,
            scale,
        )


@export("mw_bm25_topk")
def mw_bm25_topk(
    docs_addr: Int,
    freqs_addr: Int,
    lengths_addr: Int,
    npost: Int,
    ndocs: Int,
    avg_length: Float64,
    b_param: Float64,
    k1: Float64,
    boost: Float64,
    k: Int,
    dst_docs_addr: Int,
    dst_scores_addr: Int,
) abi("C") -> Int:
    var docs = IPtr(unsafe_from_address=docs_addr)
    var freqs = FPtr(unsafe_from_address=freqs_addr)
    var lengths = IPtr(unsafe_from_address=lengths_addr)
    var dst_docs = IPtr(unsafe_from_address=dst_docs_addr)
    var dst_scores = FPtr(unsafe_from_address=dst_scores_addr)
    var idf = log(Float64(ndocs) / Float64(npost + 1)) + 1.0
    var safe_avg = avg_length if avg_length > 0.0 else 1.0
    var scale = boost * idf * (k1 + 1.0)
    var used = 0
    var simd_stop = npost - npost % W
    for i in range(0, simd_stop, W):
        var doc_vec = docs.load[width=W](i)
        var weights = freqs.load[width=W](i)
        var field_lengths = lengths.gather[width=W](doc_vec).cast[DType.float64]()
        var norms = 1.0 - b_param + b_param * field_lengths / safe_avg
        var values = scale * weights / (weights + k1 * norms)
        comptime for lane in range(W):
            used = _topk_insert(
                dst_docs,
                dst_scores,
                used,
                k,
                doc_vec[lane],
                values[lane],
            )
    for i in range(simd_stop, npost):
        var doc = Int(docs[i])
        var weight = freqs[i]
        var norm = 1.0 - b_param + b_param * Float64(lengths[doc]) / safe_avg
        var value = scale * weight / (weight + k1 * norm)
        used = _topk_insert(dst_docs, dst_scores, used, k, docs[i], value)
    var size = used
    while size > 1:
        _swap_ranked(dst_docs, dst_scores, 0, size - 1)
        size -= 1
        _sift_down(dst_docs, dst_scores, 0, size)
    return used


@export("mw_tfidf_accumulate")
def mw_tfidf_accumulate(
    docs_addr: Int,
    freqs_addr: Int,
    npost: Int,
    ndocs: Int,
    boost: Float64,
    scores_addr: Int,
) abi("C"):
    var docs = IPtr(unsafe_from_address=docs_addr)
    var freqs = FPtr(unsafe_from_address=freqs_addr)
    var scores = FPtr(unsafe_from_address=scores_addr)
    var idf = log(Float64(ndocs) / Float64(npost + 1)) + 1.0
    for i in range(npost):
        var doc = Int(docs[i])
        if doc >= 0 and doc < ndocs:
            scores[doc] += boost * freqs[i] * idf


@export("mw_frequency_accumulate")
def mw_frequency_accumulate(
    docs_addr: Int,
    freqs_addr: Int,
    npost: Int,
    ndocs: Int,
    boost: Float64,
    scores_addr: Int,
) abi("C"):
    var docs = IPtr(unsafe_from_address=docs_addr)
    var freqs = FPtr(unsafe_from_address=freqs_addr)
    var scores = FPtr(unsafe_from_address=scores_addr)
    for i in range(npost):
        var doc = Int(docs[i])
        if doc >= 0 and doc < ndocs:
            scores[doc] += boost * freqs[i]


@export("mw_topk")
def mw_topk(
    docs_addr: Int,
    scores_addr: Int,
    n: Int,
    k: Int,
    dst_docs_addr: Int,
    dst_scores_addr: Int,
) abi("C") -> Int:
    var docs = IPtr(unsafe_from_address=docs_addr)
    var scores = FPtr(unsafe_from_address=scores_addr)
    var dst_docs = IPtr(unsafe_from_address=dst_docs_addr)
    var dst_scores = FPtr(unsafe_from_address=dst_scores_addr)
    var used = 0
    for i in range(n):
        used = _topk_insert(dst_docs, dst_scores, used, k, docs[i], scores[i])
    var size = used
    while size > 1:
        _swap_ranked(dst_docs, dst_scores, 0, size - 1)
        size -= 1
        _sift_down(dst_docs, dst_scores, 0, size)
    return used
