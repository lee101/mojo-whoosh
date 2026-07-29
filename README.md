# mojo-whoosh

`mojo-whoosh` is a focused Mojo implementation of the compute-heavy subset of
[Whoosh](https://whoosh.readthedocs.io/): sorted posting-list evaluation,
BM25F/TF-IDF/frequency scoring, and top-K ranking. A Python layer provides a
small persistent inverted index with selected, familiar `whoosh.fields`,
`whoosh.index`, `whoosh.query`, `whoosh.qparser`, and `whoosh.scoring` names
and common call signatures.

The package is imported as `mojo_whoosh`, so an application can keep its
Whoosh-shaped code while making the implementation choice explicit. Whoosh
2.7.4 is only needed for the parity tests and benchmarks, not at runtime.

## Covered subset

- `Schema`, `TEXT`, `ID`, `STORED`, and exact-value `NUMERIC` fields
- `create_in`, `open_dir`, `exists_in`, atomic commits, `add_document`,
  `update_document`, and deletion
- `Term`, `And`, `Or`, `Not`, `AndNot`, `Phrase`, `Prefix`, `Wildcard`,
  `TermRange`, and `Every`
- `QueryParser` and `MultifieldParser` for terms, fielded terms, phrases,
  wildcards, parentheses, and Boolean operators
- `BM25F`, `TF_IDF`, `Frequency`, and `MultiWeighting`
- ranked and field-sorted results, limits, filters, masks, matched terms,
  stored-field lookup, lexicons, and field/document statistics
- direct NumPy APIs for posting intersection, union, difference, score
  accumulation, and top-K ranking

This is not a reader for Whoosh's on-disk format. It does not cover Whoosh's
facets/grouping, highlighting, spelling and fuzzy queries, numeric range
encoding, advanced analyzers, pluggable storage backends, multiprocessing
writers, or collector customization. The current transparent JSON index
format stores the schema and source documents and rebuilds derived postings
when opened; it is useful for compact local indexes and reproducible
workloads, not as a replacement for Whoosh's mature large-index storage
engine.

## Install

The repository pins the tested Mojo nightly and all development dependencies:

```bash
pixi install
pixi run build
pixi run test
```

The build creates `dist/libmojo-whoosh.so`. `PYTHONPATH=python` is supplied by
the Pixi environment.

## Usage

```python
from mojo_whoosh import fields, index, qparser

schema = fields.Schema(
    path=fields.ID(stored=True, unique=True),
    content=fields.TEXT(stored=True),
)
ix = index.create_in("example-index", schema)

with ix.writer() as writer:
    writer.add_document(path="/guide", content="fast inverted index queries")
    writer.add_document(path="/api", content="query evaluation and ranking")

parser = qparser.QueryParser("content", schema=ix.schema)
with ix.searcher() as searcher:
    results = searcher.search(parser.parse("query OR inverted"))
    print([(hit["path"], hit.score) for hit in results])
```

Run it inside the environment with `pixi run python example.py`.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-136-generic. Kernel and query times are the best of repeated warm
runs; indexing is one measured build per implementation. Query cases use the
same 50,000 generated documents and a result limit of 20; posting cases compare
directly with Whoosh's `IntersectionMatcher` and `UnionMatcher`. A ratio above
1 means Mojo was faster.

| Case | mojo-whoosh | Whoosh 2.7.4 | Upstream / Mojo |
|---|---:|---:|---:|
| posting intersection (667k x 400k) | 6.376 ms | 129.865 ms | 20.37x faster |
| posting union (667k x 400k) | 3.976 ms | 1531.671 ms | 385.27x faster |
| index 50k documents | 3032.536 ms | 14153.553 ms | 4.67x faster |
| BM25 term query, limit 20 | 0.273 ms | 1.061 ms | 3.89x faster |
| AND query, limit 20 | 1.039 ms | 258.517 ms | 248.93x faster |
| OR query, limit 20 | 1.006 ms | 130.698 ms | 129.94x faster |

These are measurements from one machine, not universal claims. Run
`pixi run bench` on the target machine before making deployment decisions.

No GPU path is included.

## How it works

The Mojo compilation unit receives C-contiguous buffers through a C ABI.
`ctypes` passes each buffer as an integer address; Mojo reconstructs an
`UnsafePointer[..., AnyOrigin[mut=True]]` inside the exported function.
Posting document IDs and field lengths are `int64`, while term frequencies
and scores are `float64`. Sorted two-pointer scans implement Boolean set
operations. Single-term BM25 uses SIMD frequency loads, indexed field-length
gathers, contiguous score stores, and a scalar remainder loop. Large inputs
are split across CPU workers. Multi-term scoring accumulates into a dense
document array, and a bounded heap kernel ranks the requested top K with
deterministic document-ID tie-breaking.

The Python layer tokenizes fields, records term frequency and positions,
dispatches Boolean work to the shared library, and exposes Whoosh-shaped
objects. Index commits write schema and source documents through an atomic
rename. All Mojo exports live in one compilation unit because shared-library
build startup dominates compilation time for this toolchain.
