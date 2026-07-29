"""Benchmarks against Whoosh 2.7.4 on identical postings and documents."""

from __future__ import annotations

import math
import os
import platform
import tempfile
import time

import numpy as np
from whoosh import fields as whoosh_fields
from whoosh import index as whoosh_index
from whoosh import matching
from whoosh import query as whoosh_query

from mojo_whoosh import fields, index, kernels, query


def best_time(function, repeats=5):
    best = math.inf
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - started)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def matcher_intersection(a, b):
    matcher = matching.IntersectionMatcher(
        matching.ListMatcher(a, all_weights=1.0),
        matching.ListMatcher(b, all_weights=1.0),
    )
    return list(matcher.all_ids())


def matcher_union(a, b):
    matcher = matching.UnionMatcher(
        matching.ListMatcher(a, all_weights=1.0),
        matching.ListMatcher(b, all_weights=1.0),
    )
    return list(matcher.all_ids())


def build_indexes(root, count=50_000):
    mojo_path = os.path.join(root, "mojo")
    whoosh_path = os.path.join(root, "whoosh")
    os.makedirs(mojo_path)
    os.makedirs(whoosh_path)
    mojo_schema = fields.Schema(
        id=fields.ID(stored=True, unique=True), body=fields.TEXT(stored=False)
    )
    upstream_schema = whoosh_fields.Schema(
        id=whoosh_fields.ID(stored=True, unique=True),
        body=whoosh_fields.TEXT(stored=False),
    )
    ours = index.create_in(mojo_path, mojo_schema)
    theirs = whoosh_index.create_in(whoosh_path, upstream_schema)

    documents = []
    for i in range(count):
        terms = ["common", f"doc{i}"]
        if i % 2 == 0:
            terms.append("alpha")
        if i % 3 == 0:
            terms.append("beta")
        if i % 5 == 0:
            terms.append("gamma")
        documents.append({"id": str(i), "body": " ".join(terms)})

    started = time.perf_counter()
    with ours.writer() as writer:
        for document in documents:
            writer.add_document(**document)
    mojo_build = time.perf_counter() - started

    started = time.perf_counter()
    with theirs.writer() as writer:
        for document in documents:
            writer.add_document(**document)
    upstream_build = time.perf_counter() - started
    return ours, theirs, mojo_build, upstream_build


def main():
    rows = []
    a = np.arange(0, 2_000_000, 3, dtype=np.int64)
    b = np.arange(0, 2_000_000, 5, dtype=np.int64)
    a_list, b_list = a.tolist(), b.tolist()
    kernels.intersect(a, b)

    mojo_time = best_time(lambda: kernels.intersect(a, b))
    upstream_time = best_time(lambda: matcher_intersection(a_list, b_list), 3)
    assert kernels.intersect(a, b).tolist() == matcher_intersection(a_list, b_list)
    rows.append(("posting intersection (667k x 400k)", mojo_time, upstream_time))

    mojo_time = best_time(lambda: kernels.union(a, b))
    upstream_time = best_time(lambda: matcher_union(a_list, b_list), 3)
    assert kernels.union(a, b).tolist() == matcher_union(a_list, b_list)
    rows.append(("posting union (667k x 400k)", mojo_time, upstream_time))

    with tempfile.TemporaryDirectory() as root:
        ours, theirs, mojo_build, upstream_build = build_indexes(root)
        rows.append(("index 50k documents", mojo_build, upstream_build))
        with ours.searcher() as mojo_searcher, theirs.searcher() as upstream_searcher:
            cases = [
                (
                    "BM25 term query, limit 20",
                    query.Term("body", "alpha"),
                    whoosh_query.Term("body", "alpha"),
                ),
                (
                    "AND query, limit 20",
                    query.And(
                        [query.Term("body", "alpha"), query.Term("body", "beta")]
                    ),
                    whoosh_query.And(
                        [
                            whoosh_query.Term("body", "alpha"),
                            whoosh_query.Term("body", "beta"),
                        ]
                    ),
                ),
                (
                    "OR query, limit 20",
                    query.Or(
                        [query.Term("body", "beta"), query.Term("body", "gamma")]
                    ),
                    whoosh_query.Or(
                        [
                            whoosh_query.Term("body", "beta"),
                            whoosh_query.Term("body", "gamma"),
                        ]
                    ),
                ),
            ]
            for name, mojo_query, upstream_query in cases:
                mojo_searcher.search(mojo_query, limit=20)
                upstream_searcher.search(upstream_query, limit=20)
                mojo_time = best_time(
                    lambda q=mojo_query: mojo_searcher.search(q, limit=20)
                )
                upstream_time = best_time(
                    lambda q=upstream_query: upstream_searcher.search(q, limit=20)
                )
                rows.append((name, mojo_time, upstream_time))

    print(f"Machine: {cpu_name()}; {platform.system()} {platform.release()}")
    print()
    print("| Case | mojo-whoosh | Whoosh 2.7.4 | Upstream / Mojo |")
    print("|---|---:|---:|---:|")
    for name, mojo_time, upstream_time in rows:
        ratio = upstream_time / mojo_time
        verdict = "faster" if ratio >= 1 else "slower"
        print(
            f"| {name} | {mojo_time * 1000:.3f} ms | "
            f"{upstream_time * 1000:.3f} ms | {ratio:.2f}x {verdict} |"
        )


if __name__ == "__main__":
    main()
