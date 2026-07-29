"""A Mojo-accelerated subset of Whoosh's inverted index and query API."""

from . import fields, index, kernels, qparser, query, scoring
from .fields import ID, NUMERIC, STORED, TEXT, Schema
from .index import create_in, exists_in, open_dir

__all__ = [
    "ID",
    "NUMERIC",
    "STORED",
    "TEXT",
    "Schema",
    "create_in",
    "exists_in",
    "fields",
    "index",
    "kernels",
    "open_dir",
    "qparser",
    "query",
    "scoring",
]

__version__ = "0.1.0"
