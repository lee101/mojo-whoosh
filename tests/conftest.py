import pytest

from mojo_whoosh import fields as mojo_fields
from mojo_whoosh import index as mojo_index

whoosh_fields = pytest.importorskip("whoosh.fields")
whoosh_index = pytest.importorskip("whoosh.index")


DOCUMENTS = [
    {
        "id": "a",
        "title": "Rendering shade",
        "content": "rendering images with trees shade",
        "category": "guide",
    },
    {
        "id": "b",
        "title": "Vector rendering",
        "content": "fast vector rendering rendering pipeline",
        "category": "code",
    },
    {
        "id": "c",
        "title": "Search indexes",
        "content": "inverted index query evaluation",
        "category": "guide",
    },
    {
        "id": "d",
        "title": "Tree search",
        "content": "trees make shade summer",
        "category": "notes",
    },
    {
        "id": "e",
        "title": "Image query",
        "content": "image search rendering",
        "category": "code",
    },
    {
        "id": "f",
        "title": "Fast search",
        "content": "fast search query pipeline",
        "category": "notes",
    },
]


@pytest.fixture()
def indexes(tmp_path):
    mojo_path = tmp_path / "mojo"
    whoosh_path = tmp_path / "whoosh"
    mojo_path.mkdir()
    whoosh_path.mkdir()
    mojo_schema = mojo_fields.Schema(
        id=mojo_fields.ID(stored=True, unique=True),
        title=mojo_fields.TEXT(stored=True),
        content=mojo_fields.TEXT(stored=True),
        category=mojo_fields.ID(stored=True, sortable=True),
    )
    whoosh_schema = whoosh_fields.Schema(
        id=whoosh_fields.ID(stored=True, unique=True),
        title=whoosh_fields.TEXT(stored=True),
        content=whoosh_fields.TEXT(stored=True),
        category=whoosh_fields.ID(stored=True, sortable=True),
    )
    ours = mojo_index.create_in(mojo_path, mojo_schema)
    theirs = whoosh_index.create_in(whoosh_path, whoosh_schema)
    with ours.writer() as writer:
        for document in DOCUMENTS:
            writer.add_document(**document)
    with theirs.writer() as writer:
        for document in DOCUMENTS:
            writer.add_document(**document)
    return ours, theirs


def result_ids(results):
    return [hit["id"] for hit in results]
