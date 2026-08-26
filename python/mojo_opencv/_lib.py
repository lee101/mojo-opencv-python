from __future__ import annotations

import ctypes
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "dist", "libmojo-opencv-python.so")
I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mcv_filter2d": ([I] * 10 + [F, I, F], None),
    "mcv_filter2d_u8": ([I] * 10 + [F, I, I], None),
    "mcv_sepfilter2d": ([I] * 12 + [F, I, F], None),
    "mcv_sepfilter2d_u8": ([I] * 12 + [F, I, I], None),
    "mcv_median": ([I] * 7, None),
    "mcv_median_u8": ([I] * 7, None),
    "mcv_resize": ([I] * 8, None),
    "mcv_resize_u8": ([I] * 8, None),
    "mcv_resize_linear_u8": ([I] * 12, None),
    "mcv_warp": ([I] * 11 + [F], None),
    "mcv_morph": ([I] * 12 + [F], None),
    "mcv_corner": ([I, I, I, I, I, I, I, F, I, F], None),
    "mcv_canny": ([I] * 9 + [F, F, I], None),
}

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
