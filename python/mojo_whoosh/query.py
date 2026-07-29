"""Query objects matching the commonly used :mod:`whoosh.query` subset."""

from __future__ import annotations


class Query:
    boost = 1.0

    def normalize(self):
        return self


class _NullQuery(Query):
    def __repr__(self):
        return "NullQuery"


NullQuery = _NullQuery()


class Term(Query):
    def __init__(self, fieldname, text, boost=1.0):
        self.fieldname = fieldname
        self.text = text
        self.boost = float(boost)

    def __repr__(self):
        return f"Term({self.fieldname!r}, {self.text!r})"


class CompoundQuery(Query):
    JOINT = " "

    def __init__(self, subqueries, boost=1.0):
        self.subqueries = list(subqueries)
        self.boost = float(boost)

    def __iter__(self):
        return iter(self.subqueries)

    def __repr__(self):
        return f"{type(self).__name__}({self.subqueries!r})"


class And(CompoundQuery):
    pass


class Or(CompoundQuery):
    def __init__(self, subqueries, boost=1.0, minmatch=0, scale=None):
        super().__init__(subqueries, boost)
        self.minmatch = minmatch
        self.scale = scale


class DisjunctionMax(Or):
    def __init__(self, subqueries, boost=1.0, tiebreak=0.0):
        super().__init__(subqueries, boost)
        self.tiebreak = tiebreak


class Not(Query):
    def __init__(self, query, boost=1.0):
        self.query = query
        self.boost = float(boost)


class AndNot(Query):
    def __init__(self, positive, negative):
        self.positive = positive
        self.negative = negative
        self.boost = 1.0


class Phrase(Query):
    def __init__(
        self, fieldname, words, slop=1, boost=1.0, char_ranges=None
    ):
        self.fieldname = fieldname
        self.words = list(words)
        self.slop = slop
        self.boost = float(boost)
        self.char_ranges = char_ranges


class Prefix(Query):
    def __init__(self, fieldname, text, boost=1.0, constantscore=True):
        self.fieldname = fieldname
        self.text = text
        self.boost = float(boost)
        self.constantscore = constantscore


class Wildcard(Query):
    def __init__(self, fieldname, text, boost=1.0, constantscore=True):
        self.fieldname = fieldname
        self.text = text
        self.boost = float(boost)
        self.constantscore = constantscore


class Every(Query):
    def __init__(self, fieldname=None, boost=1.0):
        self.fieldname = fieldname
        self.boost = float(boost)


class TermRange(Query):
    def __init__(
        self,
        fieldname,
        start,
        end,
        startexcl=False,
        endexcl=False,
        boost=1.0,
        constantscore=True,
    ):
        self.fieldname = fieldname
        self.start = start
        self.end = end
        self.startexcl = startexcl
        self.endexcl = endexcl
        self.boost = float(boost)
        self.constantscore = constantscore
