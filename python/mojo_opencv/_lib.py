from __future__ import annotations

import ctypes
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "dist", "libmojo-opencv-python.so")
I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mcv_filter2d": ([I] * 10 + [F, I, F, I, I], None),
    "mcv_filter2d_u8": ([I] * 10 + [F, I, I, I, I], None),
    "mcv_sepfilter2d": ([I] * 12 + [F, I, F], None),
    "mcv_sepfilter2d_u8": ([I] * 12 + [F, I, I], None),
    "mcv_median": ([I] * 7 + [I, I], None),
    "mcv_median_u8": ([I] * 7 + [I, I], None),
    "mcv_resize": ([I] * 8 + [I, I], None),
    "mcv_resize_u8": ([I] * 8 + [I, I], None),
    "mcv_resize_linear_u8": ([I] * 12 + [I, I], None),
    "mcv_warp": ([I] * 11 + [F, I, I], None),
    "mcv_morph": ([I] * 12 + [F, I, I], None),
    "mcv_corner_gradient": ([I, I, I, I, I, I, F, I, I], None),
    "mcv_corner_response": ([I, I, I, I, I, I, F, I, I, I], None),
    "mcv_canny": ([I] * 9 + [F, F, I, I, I], None),
}

# Threading only pays above roughly 2 flops/byte, so the fan-out gate is a
# scalar-operation count. Bandwidth-bound kernels (a 3x3 filter, a nearest
# resize) never clear it and stay serial; morphology, warps, Harris responses
# and the Canny gradient pass do.
MAX_WORKERS = 8
MIN_OPS = 1 << 25
_pool: ThreadPoolExecutor | None = None


def rows(call, count: int, ops_per_row: int) -> None:
    """Fan ``call(y0, y1)`` out over image rows. ctypes drops the GIL."""
    global _pool
    workers = min(MAX_WORKERS, os.cpu_count() or 1)
    step = count // workers
    if workers <= 1 or step == 0 or count * ops_per_row < MIN_OPS:
        call(0, count)
        return
    if _pool is None:
        _pool = ThreadPoolExecutor(max_workers=workers)
    bounds = [(part * step, (part + 1) * step) for part in range(workers - 1)]
    bounds.append((bounds[-1][1], count))
    for future in [_pool.submit(call, lo, hi) for lo, hi in bounds]:
        future.result()

_lib: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        if not os.path.exists(LIB):
            raise RuntimeError("compiled library missing; run `pixi run build`")
        _lib = ctypes.CDLL(LIB)
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib


def f64(value) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.float64)


def addr(value: np.ndarray) -> int:
    if not isinstance(value, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if value.size == 0:
        raise ValueError("FFI buffers must not be empty")
    if not value.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous")
    address = int(value.ctypes.data)
    if address == 0:
        raise RuntimeError("NumPy returned a null buffer address")
    return address
