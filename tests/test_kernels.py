import numpy as np
import pytest

from mojo_whoosh import kernels


def test_intersection_matches_numpy():
    a = np.arange(0, 200_000, 3, dtype=np.int64)
    b = np.arange(0, 200_000, 5, dtype=np.int64)
    assert np.array_equal(kernels.intersect(a, b), np.intersect1d(a, b))


def test_union_matches_numpy():
    a = np.arange(0, 100_000, 7, dtype=np.int64)
    b = np.arange(0, 100_000, 11, dtype=np.int64)
    assert np.array_equal(kernels.union(a, b), np.union1d(a, b))


def test_difference_matches_numpy():
    a = np.arange(0, 100_000, 3, dtype=np.int64)
    b = np.arange(0, 100_000, 12, dtype=np.int64)
    assert np.array_equal(kernels.difference(a, b), np.setdiff1d(a, b))


def test_empty_posting_operations():
    empty = np.empty(0, dtype=np.int64)
    values = np.array([1, 4, 8], dtype=np.int64)
    assert not kernels.intersect(empty, values).size
    assert np.array_equal(kernels.union(empty, values), values)
    assert np.array_equal(kernels.difference(values, empty), values)


def test_bm25_matches_whoosh_formula():
    docs = np.array([0, 2, 4], dtype=np.int64)
    freq = np.array([1.0, 3.0, 2.0])
    lengths = np.array([3, 5, 8, 2, 6], dtype=np.int64)
    scores = kernels.accumulate_bm25(
        docs,
        freq,
        lengths,
        np.zeros(5),
        avg_length=lengths.mean(),
        B=0.75,
        K1=1.2,
        boost=1.5,
    )
    idf = np.log(5 / (3 + 1)) + 1
    expected = np.zeros(5)
    for doc, weight in zip(docs, freq):
        norm = 1 - 0.75 + 0.75 * lengths[doc] / lengths.mean()
        expected[doc] = 1.5 * idf * weight * 2.2 / (weight + 1.2 * norm)
    assert scores == pytest.approx(expected)


def test_bm25_simd_tail_matches_accumulator():
    docs = np.arange(17, dtype=np.int64)
    frequencies = np.linspace(1.0, 4.0, docs.size)
    lengths = np.arange(2, 2 + docs.size, dtype=np.int64)
    expected = kernels.accumulate_bm25(
        docs,
        frequencies,
        lengths,
        np.zeros(docs.size),
        avg_length=lengths.mean(),
        B=0.4,
        K1=1.8,
        boost=1.7,
    )
    actual = kernels.score_bm25(
        docs,
        frequencies,
        lengths,
        avg_length=lengths.mean(),
        B=0.4,
        K1=1.8,
        boost=1.7,
    )
    assert actual == pytest.approx(expected, rel=1e-12)


def test_bm25_parallel_threshold_matches_formula():
    count = kernels.BM25_PARALLEL_THRESHOLD + 1
    docs = np.arange(count, dtype=np.int64)
    frequencies = np.linspace(1.0, 5.0, count)
    lengths = docs % 31 + 1
    actual = kernels.score_bm25(
        docs,
        frequencies,
        lengths,
        avg_length=16.0,
        B=0.75,
        K1=1.2,
        boost=1.3,
    )
    idf = np.log(count / (count + 1)) + 1
    norms = 0.25 + 0.75 * lengths / 16.0
    expected = 1.3 * idf * frequencies * 2.2 / (frequencies + 1.2 * norms)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)


def test_tfidf_and_frequency_accumulation():
    docs = np.array([0, 2], dtype=np.int64)
    freq = np.array([2.0, 4.0])
    tfidf = kernels.accumulate_tfidf(docs, freq, np.zeros(5), boost=2.0)
    expected = np.zeros(5)
    expected[docs] = 2 * freq * (np.log(5 / 3) + 1)
    assert tfidf == pytest.approx(expected)
    frequency = kernels.accumulate_frequency(docs, freq, np.zeros(5), boost=0.5)
    assert frequency.tolist() == [1.0, 0.0, 2.0, 0.0, 0.0]


def test_top_k_orders_scores_and_ties():
    docs = np.array([9, 3, 7, 1, 5])
    scores = np.array([0.2, 1.0, 0.7, 1.0, 0.4])
    got_docs, got_scores = kernels.top_k(docs, scores, 3)
    assert got_docs.tolist() == [1, 3, 7]
    assert got_scores.tolist() == [1.0, 1.0, 0.7]


def test_top_k_validates_aligned_arrays():
    with pytest.raises(ValueError):
        kernels.top_k([1, 2], [1.0], 1)


@pytest.mark.parametrize(
    "function",
    [kernels.accumulate_bm25, kernels.accumulate_tfidf, kernels.accumulate_frequency],
)
def test_accumulators_validate_aligned_arrays(function):
    kwargs = {"avg_length": 1.0} if function is kernels.accumulate_bm25 else {}
    lengths = ([1, 1],) if function is kernels.accumulate_bm25 else ()
    with pytest.raises(ValueError, match="same length"):
        function([0, 1], [1.0], *lengths, np.zeros(2), **kwargs)


def test_bm25_rejects_short_lengths_and_out_of_range_docids():
    with pytest.raises(ValueError, match="cover"):
        kernels.accumulate_bm25(
            [0], [1.0], [1], np.zeros(2), avg_length=1.0
        )
    with pytest.raises(ValueError, match="docids"):
        kernels.score_bm25([1], [1.0], [1], avg_length=1.0)


@pytest.mark.parametrize(
    "function,args",
    [
        (kernels.intersect, ([0.5], [1])),
        (kernels.union, ([0], [[1]])),
        (kernels.top_k, ([0.5], [1.0], 1)),
    ],
)
def test_integer_inputs_reject_narrowing_and_bad_dimensions(function, args):
    with pytest.raises((TypeError, ValueError)):
        function(*args)


def test_uint64_docids_reject_overflow():
    values = np.array([np.iinfo(np.uint64).max], dtype=np.uint64)
    with pytest.raises(OverflowError):
        kernels.intersect(values, np.array([1], dtype=np.int64))


def test_float64_inputs_reject_integer_precision_loss():
    frequencies = np.array([(1 << 53) + 1], dtype=np.uint64)
    with pytest.raises(OverflowError, match="represent exactly"):
        kernels.accumulate_frequency([0], frequencies, [0.0])


def test_noncontiguous_inputs_are_copied_safely_for_ffi():
    base = np.arange(40, dtype=np.int64)
    assert kernels.intersect(base[::2], base[::3]).tolist() == [0, 6, 12, 18, 24, 30, 36]


def test_top_k_rejects_noninteger_limit():
    with pytest.raises(TypeError, match="integer"):
        kernels.top_k([1], [1.0], 1.5)
