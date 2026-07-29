import numpy as np
import pytest

from mojo_whoosh import fields, index, qparser, query, scoring

from conftest import result_ids

from whoosh import qparser as whoosh_qparser
from whoosh import query as whoosh_query
from whoosh import scoring as whoosh_scoring


def paired_search(indexes, ours_query, their_query, **kwargs):
    ours, theirs = indexes
    ours_weighting = kwargs.pop("ours_weighting", None)
    their_weighting = kwargs.pop("their_weighting", None)
    a = ours.searcher(**({"weighting": ours_weighting} if ours_weighting else {}))
    b = theirs.searcher(
        **({"weighting": their_weighting} if their_weighting else {})
    )
    return a.search(ours_query, **kwargs), b.search(their_query, **kwargs)


def test_term_results_and_bm25_scores_match(indexes):
    ours, theirs = paired_search(
        indexes,
        query.Term("content", "rendering"),
        whoosh_query.Term("content", "rendering"),
    )
    assert result_ids(ours) == result_ids(theirs)
    assert [hit.score for hit in ours] == pytest.approx(
        [hit.score for hit in theirs], rel=1e-9
    )


def test_and_query_matches(indexes):
    ours, theirs = paired_search(
        indexes,
        query.And(
            [query.Term("content", "fast"), query.Term("content", "pipeline")]
        ),
        whoosh_query.And(
            [
                whoosh_query.Term("content", "fast"),
                whoosh_query.Term("content", "pipeline"),
            ]
        ),
    )
    assert result_ids(ours) == result_ids(theirs)
    assert [hit.score for hit in ours] == pytest.approx(
        [hit.score for hit in theirs]
    )


def test_or_query_matches(indexes):
    ours, theirs = paired_search(
        indexes,
        query.Or(
            [query.Term("content", "inverted"), query.Term("content", "vector")]
        ),
        whoosh_query.Or(
            [
                whoosh_query.Term("content", "inverted"),
                whoosh_query.Term("content", "vector"),
            ]
        ),
    )
    assert result_ids(ours) == result_ids(theirs)


def test_and_not_matches(indexes):
    ours, theirs = paired_search(
        indexes,
        query.And(
            [query.Term("content", "rendering"), query.Not(query.Term("content", "vector"))]
        ),
        whoosh_query.And(
            [
                whoosh_query.Term("content", "rendering"),
                whoosh_query.Not(whoosh_query.Term("content", "vector")),
            ]
        ),
    )
    assert result_ids(ours) == result_ids(theirs)


def test_phrase_query_matches(indexes):
    ours, theirs = paired_search(
        indexes,
        query.Phrase("content", ["fast", "search"]),
        whoosh_query.Phrase("content", ["fast", "search"]),
    )
    assert result_ids(ours) == result_ids(theirs) == ["f"]
    assert [hit.score for hit in ours] == pytest.approx(
        [hit.score for hit in theirs]
    )


@pytest.mark.parametrize(
    ("ours_query", "their_query"),
    [
        (
            query.Prefix("content", "rend"),
            whoosh_query.Prefix("content", "rend"),
        ),
        (
            query.Wildcard("content", "rende?ing"),
            whoosh_query.Wildcard("content", "rende?ing"),
        ),
        (
            query.TermRange("content", "image", "inverted"),
            whoosh_query.TermRange("content", "image", "inverted"),
        ),
    ],
)
def test_multiterm_queries_match(indexes, ours_query, their_query):
    ours, theirs = paired_search(indexes, ours_query, their_query)
    assert sorted(result_ids(ours)) == sorted(result_ids(theirs))


def test_every_matches(indexes):
    ours, theirs = paired_search(
        indexes, query.Every(), whoosh_query.Every(), limit=None
    )
    assert result_ids(ours) == result_ids(theirs)


def test_frequency_scores_match(indexes):
    ours, theirs = paired_search(
        indexes,
        query.Term("content", "rendering"),
        whoosh_query.Term("content", "rendering"),
        ours_weighting=scoring.Frequency(),
        their_weighting=whoosh_scoring.Frequency(),
    )
    assert result_ids(ours) == result_ids(theirs)
    assert [hit.score for hit in ours] == [hit.score for hit in theirs]


def test_tfidf_scores_match(indexes):
    ours, theirs = paired_search(
        indexes,
        query.Term("content", "rendering", boost=1.7),
        whoosh_query.Term("content", "rendering", boost=1.7),
        ours_weighting=scoring.TF_IDF(),
        their_weighting=whoosh_scoring.TF_IDF(),
    )
    assert result_ids(ours) == result_ids(theirs)
    assert [hit.score for hit in ours] == pytest.approx(
        [hit.score for hit in theirs]
    )


def test_custom_bm25_scores_match(indexes):
    ours, theirs = paired_search(
        indexes,
        query.Term("content", "rendering"),
        whoosh_query.Term("content", "rendering"),
        ours_weighting=scoring.BM25F(B=0.4, K1=1.8),
        their_weighting=whoosh_scoring.BM25F(B=0.4, K1=1.8),
    )
    assert [hit.score for hit in ours] == pytest.approx(
        [hit.score for hit in theirs]
    )


def test_query_parser_behavior_matches(indexes):
    ours_index, theirs_index = indexes
    expressions = [
        "rendering pipeline",
        "rendering OR inverted",
        '"fast search"',
        'title:"vector rendering"',
        "rend* AND NOT vector",
        "title:rendering OR content:inverted",
        "(rendering OR inverted) AND query",
    ]
    with ours_index.searcher() as a, theirs_index.searcher() as b:
        for expression in expressions:
            aq = qparser.QueryParser("content", ours_index.schema).parse(expression)
            bq = whoosh_qparser.QueryParser("content", theirs_index.schema).parse(
                expression
            )
            assert sorted(result_ids(a.search(aq, limit=None))) == sorted(
                result_ids(b.search(bq, limit=None))
            )


def test_limit_and_estimated_length_match(indexes):
    ours, theirs = paired_search(
        indexes,
        query.Term("content", "rendering"),
        whoosh_query.Term("content", "rendering"),
        limit=2,
    )
    assert result_ids(ours) == result_ids(theirs)
    assert ours.scored_length() == theirs.scored_length() == 2
    assert ours.estimated_length() == theirs.estimated_length() == 3


def test_sorted_field_matches(indexes):
    ours, theirs = paired_search(
        indexes, query.Every(), whoosh_query.Every(), limit=None, sortedby="category"
    )
    assert result_ids(ours) == result_ids(theirs)


def test_filter_and_mask_match(indexes):
    our_index, their_index = indexes
    with our_index.searcher() as searcher:
        ours = searcher.search(
            query.Every(),
            filter=query.Term("category", "guide"),
            mask=query.Term("content", "inverted"),
        )
    with their_index.searcher() as searcher:
        their_results = searcher.search(
            whoosh_query.Every(),
            filter=whoosh_query.Term("category", "guide"),
            mask=whoosh_query.Term("content", "inverted"),
        )
        their_ids = result_ids(their_results)
    assert result_ids(ours) == their_ids


def test_matched_terms_match(indexes):
    ours, theirs = paired_search(
        indexes,
        query.Or([query.Term("content", "fast"), query.Term("content", "search")]),
        whoosh_query.Or(
            [
                whoosh_query.Term("content", "fast"),
                whoosh_query.Term("content", "search"),
            ]
        ),
        terms=True,
    )
    assert result_ids(ours) == result_ids(theirs)
    assert [set(hit.matched_terms()) for hit in ours] == [
        set(hit.matched_terms()) for hit in theirs
    ]


def test_searcher_statistics_match(indexes):
    ours_index, theirs_index = indexes
    with ours_index.searcher() as ours, theirs_index.searcher() as theirs:
        assert ours.doc_count_all() == theirs.doc_count_all()
        assert ours.field_length("content") == theirs.field_length("content")
        assert ours.avg_field_length("content") == pytest.approx(
            theirs.avg_field_length("content")
        )
        assert ours.frequency("content", "rendering") == theirs.frequency(
            "content", b"rendering"
        )
        assert ours.doc_frequency("content", "rendering") == theirs.doc_frequency(
            "content", b"rendering"
        )


def test_lexicon_and_prefix_expansion_match(indexes):
    ours_index, theirs_index = indexes
    with ours_index.searcher() as ours, theirs_index.searcher() as theirs:
        assert list(ours.lexicon("content")) == list(theirs.lexicon("content"))
        assert list(ours.expand_prefix("content", "rend")) == list(
            theirs.reader().expand_prefix("content", b"rend")
        )


def test_document_lookup_matches(indexes):
    ours_index, theirs_index = indexes
    with ours_index.searcher() as ours, theirs_index.searcher() as theirs:
        assert ours.document(id="c") == theirs.document(id="c")
        assert list(ours.documents(category="code")) == list(
            theirs.documents(category="code")
        )


def test_index_reopens_with_same_results(indexes):
    ours, _ = indexes
    reopened = index.open_dir(ours.dirname)
    with reopened.searcher() as searcher:
        results = searcher.search(query.Term("content", "query"), limit=None)
        assert result_ids(results) == ["c", "f"]


def test_update_and_delete_document(tmp_path):
    schema = fields.Schema(
        id=fields.ID(stored=True, unique=True), body=fields.TEXT(stored=True)
    )
    ix = index.create_in(tmp_path, schema)
    with ix.writer() as writer:
        writer.add_document(id="one", body="old value")
        writer.add_document(id="two", body="delete value")
    with ix.writer() as writer:
        writer.update_document(id="one", body="new value")
        assert writer.delete_by_term("body", "delete") == 1
    with ix.searcher() as searcher:
        assert searcher.doc_count_all() == 1
        assert searcher.document(id="one")["body"] == "new value"


def test_writer_cancels_on_exception(tmp_path):
    schema = fields.Schema(id=fields.ID(stored=True))
    ix = index.create_in(tmp_path, schema)
    with pytest.raises(RuntimeError):
        with ix.writer() as writer:
            writer.add_document(id="lost")
            raise RuntimeError("cancel")
    assert ix.doc_count() == 0


def test_phrase_positions_match_across_removed_stopword(tmp_path):
    schema = fields.Schema(id=fields.ID(stored=True), body=fields.TEXT())
    ix = index.create_in(tmp_path, schema)
    with ix.writer() as writer:
        writer.add_document(id="one", body="fast and search")
    with ix.searcher() as searcher:
        results = searcher.search(query.Phrase("body", ["fast", "search"]))
        assert result_ids(results) == ["one"]


def test_schema_metadata():
    schema = fields.Schema(
        title=fields.TEXT(stored=True), slug=fields.ID(unique=True), raw=fields.STORED()
    )
    assert schema.names() == ["title", "slug", "raw"]
    assert schema.stored_names() == ["title", "raw"]
    assert schema["slug"].unique


def test_numeric_exact_values_exists_and_delete_document(tmp_path):
    schema = fields.Schema(
        id=fields.ID(stored=True),
        number=fields.NUMERIC(stored=True),
        payload=fields.STORED(),
    )
    assert not index.exists_in(tmp_path)
    ix = index.create_in(tmp_path, schema)
    assert index.exists_in(tmp_path)
    with ix.writer() as writer:
        writer.add_document(id="one", number=42, payload={"ok": True})
        writer.add_document(id="two", number=7, payload=None)
    with ix.searcher() as searcher:
        assert result_ids(searcher.search(query.Term("number", 42))) == ["one"]
    with ix.writer() as writer:
        writer.delete_document(1)
    assert ix.doc_count() == 1


def test_andnot_query_matches(indexes):
    ours, theirs = paired_search(
        indexes,
        query.AndNot(
            query.Term("content", "rendering"),
            query.Term("content", "vector"),
        ),
        whoosh_query.AndNot(
            whoosh_query.Term("content", "rendering"),
            whoosh_query.Term("content", "vector"),
        ),
    )
    assert result_ids(ours) == result_ids(theirs)


def test_multifield_parser_matches(indexes):
    ours_index, theirs_index = indexes
    ours_parser = qparser.MultifieldParser(["title", "content"], ours_index.schema)
    theirs_parser = whoosh_qparser.MultifieldParser(
        ["title", "content"], theirs_index.schema
    )
    with ours_index.searcher() as ours, theirs_index.searcher() as theirs:
        assert result_ids(ours.search(ours_parser.parse("shade"), limit=None)) == result_ids(
            theirs.search(theirs_parser.parse("shade"), limit=None)
        )


def test_multiweighting_selects_field_model(indexes):
    ours_index, theirs_index = indexes
    ours_weighting = scoring.MultiWeighting(
        default=scoring.BM25F(), title=scoring.Frequency()
    )
    theirs_weighting = whoosh_scoring.MultiWeighting(
        default=whoosh_scoring.BM25F(), title=whoosh_scoring.Frequency()
    )
    ours, theirs = paired_search(
        indexes,
        query.Term("title", "rendering"),
        whoosh_query.Term("title", "rendering"),
        ours_weighting=ours_weighting,
        their_weighting=theirs_weighting,
    )
    assert result_ids(ours) == result_ids(theirs)
    assert [hit.score for hit in ours] == [hit.score for hit in theirs]


def test_commit_replaces_index_atomically(tmp_path, monkeypatch):
    schema = fields.Schema(id=fields.ID(stored=True))
    ix = index.create_in(tmp_path, schema)
    original = (tmp_path / "MAIN.mwhoosh.json").read_text()

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(index.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failure"):
        with ix.writer() as writer:
            writer.add_document(id="not-committed")
    assert (tmp_path / "MAIN.mwhoosh.json").read_text() == original
    assert ix.doc_count() == 0
    assert not list(tmp_path.glob("*.tmp"))
