"""Whoosh-compatible weighting model configuration."""

from __future__ import annotations


class WeightingModel:
    use_final = False


class BM25F(WeightingModel):
    def __init__(self, B: float = 0.75, K1: float = 1.2, **kwargs):
        self.B = float(B)
        self.K1 = float(K1)
        self._field_B = {
            name[:-2]: float(value)
            for name, value in kwargs.items()
            if name.endswith("_B")
        }

    def field_B(self, fieldname: str) -> float:
        return self._field_B.get(fieldname, self.B)


class TF_IDF(WeightingModel):
    pass


class Frequency(WeightingModel):
    pass


class MultiWeighting(WeightingModel):
    def __init__(self, default=None, **weightings):
        self.default = default or BM25F()
        self.weightings = weightings

    def weighting(self, fieldname):
        return self.weightings.get(fieldname, self.default)
