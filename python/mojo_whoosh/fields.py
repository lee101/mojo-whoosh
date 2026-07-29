"""The covered subset of :mod:`whoosh.fields`."""

from __future__ import annotations

from collections import OrderedDict


class FieldType:
    kind = "field"

    def __init__(
        self,
        *,
        stored: bool = False,
        unique: bool = False,
        field_boost: float = 1.0,
        sortable: bool = False,
    ):
        self.stored = stored
        self.unique = unique
        self.field_boost = float(field_boost)
        self.sortable = sortable

    def descriptor(self) -> dict:
        return {
            "kind": self.kind,
            "stored": self.stored,
            "unique": self.unique,
            "field_boost": self.field_boost,
            "sortable": self.sortable,
        }


class TEXT(FieldType):
    kind = "text"

    def __init__(
        self,
        analyzer=None,
        phrase: bool = True,
        vector=None,
        stored: bool = False,
        field_boost: float = 1.0,
        spelling: bool = False,
        chars: bool = False,
        lang=None,
        sortable: bool = False,
        multitoken_query: str = "default",
    ):
        super().__init__(
            stored=stored, field_boost=field_boost, sortable=sortable
        )
        self.analyzer = analyzer
        self.phrase = phrase
        self.vector = vector
        self.spelling = spelling
        self.chars = chars
        self.lang = lang
        self.multitoken_query = multitoken_query

    def descriptor(self) -> dict:
        result = super().descriptor()
        result.update({"phrase": self.phrase})
        return result


class ID(FieldType):
    kind = "id"

    def __init__(
        self,
        stored: bool = False,
        unique: bool = False,
        field_boost: float = 1.0,
        sortable: bool = False,
        analyzer=None,
    ):
        super().__init__(
            stored=stored,
            unique=unique,
            field_boost=field_boost,
            sortable=sortable,
        )
        self.analyzer = analyzer


class STORED(FieldType):
    kind = "stored"

    def __init__(self):
        super().__init__(stored=True)


class NUMERIC(FieldType):
    kind = "numeric"

    def __init__(
        self,
        numtype=int,
        bits: int = 64,
        stored: bool = False,
        unique: bool = False,
        field_boost: float = 1.0,
        decimal_places: int = 0,
        shift_step: int = 4,
        sortable: bool = False,
        signed: bool = True,
    ):
        super().__init__(
            stored=stored,
            unique=unique,
            field_boost=field_boost,
            sortable=sortable,
        )
        self.numtype = numtype
        self.bits = bits
        self.decimal_places = decimal_places
        self.shift_step = shift_step
        self.signed = signed


_FIELD_TYPES = {"text": TEXT, "id": ID, "stored": STORED, "numeric": NUMERIC}


class Schema:
    def __init__(self, *fields, **kwargs):
        self._fields = OrderedDict()
        for name, field in fields:
            self.add(name, field)
        for name, field in kwargs.items():
            self.add(name, field)

    def add(self, name: str, fieldtype: FieldType, glob: bool = False):
        if not isinstance(fieldtype, FieldType):
            raise TypeError(f"{name!r} is not a FieldType")
        self._fields[name] = fieldtype
        return self

    def remove(self, fieldname: str):
        del self._fields[fieldname]

    def names(self, check_names=None):
        names = list(self._fields)
        if check_names is None:
            return names
        return [name for name in names if name in check_names]

    def stored_names(self):
        return [name for name, field in self._fields.items() if field.stored]

    def sortable_names(self):
        return [name for name, field in self._fields.items() if field.sortable]

    def has_scorable_fields(self):
        return any(field.kind == "text" for field in self._fields.values())

    def scorable_fields(self):
        return [name for name, field in self._fields.items() if field.kind == "text"]

    def __contains__(self, name):
        return name in self._fields

    def __getitem__(self, name):
        return self._fields[name]

    def __iter__(self):
        return iter(self._fields)

    def __len__(self):
        return len(self._fields)

    def descriptor(self) -> dict:
        return {name: field.descriptor() for name, field in self._fields.items()}

    @classmethod
    def from_descriptor(cls, description: dict):
        schema = cls()
        for name, values in description.items():
            values = dict(values)
            kind = values.pop("kind")
            if kind == "stored":
                field = STORED()
            elif kind == "text":
                allowed = {
                    key: values[key]
                    for key in ("stored", "field_boost", "sortable", "phrase")
                    if key in values
                }
                field = TEXT(**allowed)
            elif kind == "id":
                field = ID(**values)
            else:
                field = NUMERIC(
                    **{
                        key: values[key]
                        for key in (
                            "stored",
                            "unique",
                            "field_boost",
                            "sortable",
                        )
                        if key in values
                    }
                )
            schema.add(name, field)
        return schema
