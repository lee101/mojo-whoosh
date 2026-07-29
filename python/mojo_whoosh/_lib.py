"""ctypes loader for the Mojo query kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_WHOOSH_LIB") or os.path.join(
    ROOT, "dist", "libmojo-whoosh.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mw_intersect": ([I, I, I, I, I], I),
    "mw_union": ([I, I, I, I, I], I),
    "mw_difference": ([I, I, I, I, I], I),
    "mw_bm25_accumulate": ([I, I, I, I, I, F, F, F, F, I], None),
    "mw_bm25_scores": ([I, I, I, I, I, F, F, F, F, I], None),
    "mw_tfidf_accumulate": ([I, I, I, I, F, I], None),
    "mw_frequency_accumulate": ([I, I, I, I, F, I], None),
    "mw_topk": ([I, I, I, I, I, I], I),
}

_loaded: ctypes.CDLL | None = None


def build() -> str:
    if os.environ.get("MOJO_WHOOSH_LIB"):
        if not os.path.exists(LIB):
            raise RuntimeError(f"MOJO_WHOOSH_LIB does not exist: {LIB}")
        return LIB
    source = os.path.join(ROOT, "src", "kernels.mojo")
    if os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(source):
        return LIB
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise RuntimeError((proc.stderr or proc.stdout).strip())
    return LIB


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_loaded, name)
            function.argtypes = argtypes
            function.restype = restype
    return _loaded


def addr(array) -> int:
    address = int(array.ctypes.data)
    if array.size and not address:
        raise RuntimeError("NumPy returned a null address for a non-empty array")
    return address
