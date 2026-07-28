"""Benchmarks against OpenCV on identical inputs."""

from __future__ import annotations

import os
import platform
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojo_opencv as mcv  # noqa: E402


def best_time(function, repeats=4):
    function()
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def main():
    cv2.setNumThreads(1)
    rng = np.random.default_rng(7)
    gray = rng.integers(0, 256, (1024, 1024), dtype=np.uint8)
    color = rng.integers(0, 256, (1024, 1024, 3), dtype=np.uint8)
    fgray = gray.astype(np.float64)
    kernel = rng.normal(size=(7, 7))
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    affine = cv2.getRotationMatrix2D((512, 512), 17, 0.93)

    cases = [
        ("filter2D 1024x1024 f64, 7x7",
         lambda: mcv.filter2D(fgray, -1, kernel),
         lambda: cv2.filter2D(fgray, -1, kernel)),
        ("GaussianBlur 1024x1024 u8, 7x7",
         lambda: mcv.GaussianBlur(gray, (7, 7), 1.4),
         lambda: cv2.GaussianBlur(gray, (7, 7), 1.4)),
        ("medianBlur 1024x1024 u8, 5x5",
         lambda: mcv.medianBlur(gray, 5),
         lambda: cv2.medianBlur(gray, 5)),
        ("resize 1024x1024x3 -> 640x640",
         lambda: mcv.resize(color, (640, 640)),
         lambda: cv2.resize(color, (640, 640))),
        ("warpAffine 1024x1024 f64",
         lambda: mcv.warpAffine(fgray, affine, (1024, 1024)),
         lambda: cv2.warpAffine(fgray, affine, (1024, 1024))),
        ("erode 1024x1024 u8, 7x7",
         lambda: mcv.erode(gray, morph_kernel),
         lambda: cv2.erode(gray, morph_kernel)),
        ("cornerHarris 1024x1024 u8",
         lambda: mcv.cornerHarris(gray, 3, 3, 0.04),
         lambda: cv2.cornerHarris(gray, 3, 3, 0.04)),
        ("Canny 1024x1024 u8",
         lambda: mcv.Canny(gray, 80, 160),
         lambda: cv2.Canny(gray, 80, 160)),
    ]

    print(f"Machine: {platform.processor() or platform.machine()} | "
          f"{platform.platform()} | OpenCV {cv2.__version__} (1 thread)")
    print()
    print("| Kernel | Mojo (ms) | OpenCV (ms) | Speedup |")
    print("|---|---:|---:|---:|")
    for name, mojo_function, cv_function in cases:
        mojo_seconds = best_time(mojo_function)
        cv_seconds = best_time(cv_function)
        speedup = cv_seconds / mojo_seconds
        print(f"| {name} | {mojo_seconds * 1000:.2f} | "
              f"{cv_seconds * 1000:.2f} | {speedup:.3f}x |")


if __name__ == "__main__":
    main()
