# mojo-opencv-python

`mojo-opencv-python` is a standalone Mojo port of a useful, compute-heavy subset
of [`opencv-python`](https://pypi.org/project/opencv-python/). It provides an
OpenCV-shaped Python API backed by kernels compiled from Mojo into one shared
library. Import it as `mojo_opencv`; using the conventional `cv2` alias makes
covered calls easy to move between the two implementations.

This is a correctness-first port with SIMD fast paths, thresholded CPU
parallelism, and native `uint8` input paths for the benchmarked filters and
resize operation. OpenCV still has years of architecture-specific and
algorithmic optimization, so performance depends strongly on the kernel.

## Coverage

The covered API is:

- Filters: `filter2D`, `sepFilter2D`, `boxFilter`, `blur`, `GaussianBlur`,
  `getGaussianKernel`, `medianBlur`, `Sobel`, `Scharr`, and `Laplacian`.
- Warps: `resize`, `warpAffine`, `warpPerspective`, `getRotationMatrix2D`,
  `getAffineTransform`, `getPerspectiveTransform`, and
  `invertAffineTransform`.
- Morphology: `erode`, `dilate`, `morphologyEx`, and
  `getStructuringElement`, including rectangular, cross, and elliptical
  kernels.
- Features: `cornerHarris`, `cornerMinEigenVal`, `goodFeaturesToTrack`, and
  `Canny`.
- OpenCV constants used by those functions, including depths, borders,
  interpolation modes, and morphology operations.

Grayscale and channel-last images are supported. Accepted input dtypes are
`uint8`, `uint16`, `int16`, `float32`, and `float64`; unsupported dtypes are
rejected instead of being silently narrowed. The corresponding OpenCV output
depth constants are available, including the tested `uint8` to `int16` Sobel
path. Linear and nearest-neighbor interpolation, constant/replicate/reflect/
reflect-101 borders, custom filter anchors, morphology iterations, optional
destination arrays, and Canny's L1/L2 gradient modes are implemented.

Not covered: image codecs and video I/O, color conversion, drawing, contours,
connected components, remap/polar transforms, pyramids, bilateral filtering,
Hough transforms, keypoint/descriptor classes such as ORB or SIFT, geometric
calibration, DNN, CUDA, and OpenCV's GUI. Sobel and Laplacian currently cover
kernel sizes 1 and 3; Canny and corner detection use aperture size 3.
Per-channel constant border values and `MORPH_HITMISS` are not yet supported.

## Install

Clone the repository and run these commands from its root. The repository pins
the Mojo nightly used to build it. Pixi installs Mojo, Python, NumPy, pytest,
and OpenCV for parity testing:

```bash
pixi install
pixi run build
pixi run test
```

The test suite compares results to the `cv2` bindings supplied by the
conda-forge `opencv` package. It is a parity oracle for the covered API, not a
claim that this package replaces the rest of `opencv-python`.

## Usage

```python
import numpy as np
import mojo_opencv as cv2

image = np.zeros((64, 64), dtype=np.uint8)
image[16:48, 20:44] = 255

smoothed = cv2.GaussianBlur(image, (5, 5), 1.2)
edges = cv2.Canny(smoothed, 40, 100)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

print(smoothed.shape, closed.dtype, int(np.count_nonzero(closed)))
```

Run that example after `pixi run build` with:

```bash
pixi run python example.py
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 system running
Linux 6.8.0-136-generic, x86-64, glibc 2.39, and OpenCV 5.0.0. OpenCV was
restricted to one thread. Times are the best of four warmed runs; speedup is
OpenCV time divided by Mojo time, so values below 1 mean Mojo is slower.

| Kernel | Mojo (ms) | OpenCV (ms) | Speedup |
|---|---:|---:|---:|
| filter2D 1024x1024 f64, 7x7 | 21.87 | 23.03 | 1.053x |
| GaussianBlur 1024x1024 u8, 7x7 | 38.21 | 1.74 | 0.046x |
| medianBlur 1024x1024 u8, 5x5 | 73.62 | 3.13 | 0.043x |
| resize 1024x1024x3 -> 640x640 | 48.01 | 2.67 | 0.056x |
| warpAffine 1024x1024 f64 | 16.32 | 15.32 | 0.939x |
| erode 1024x1024 u8, 7x7 | 17.88 | 0.69 | 0.039x |
| cornerHarris 1024x1024 u8 | 31.87 | 11.45 | 0.359x |
| Canny 1024x1024 u8 | 85.52 | 25.78 | 0.301x |

Mojo is slightly faster on `filter2D` in this run. OpenCV retains a large lead
where it uses separable, histogram-based, or specialized morphology algorithms
while this port still uses direct kernels.

No GPU path is included or benchmarked.

## How it works

`src/capi.mojo` is the single compilation unit. `build/build.sh` runs
`mojo build --emit shared-lib` and writes
`dist/libmojo-opencv-python.so`. Exports use the C ABI; NumPy buffers cross the
boundary as integer addresses because exported Mojo functions cannot be
parametric over pointer origins in the pinned compiler.

The Python layer validates shapes, dimensions, dtypes, finite coefficients,
border modes, and nonempty buffers; ensures C-contiguous row-major storage;
and calls the shared library with `ctypes`. Images are
flattened logically as `(height, width, channels)` with interleaved,
channel-last pixels. Contiguous `uint8` filter, median, and resize inputs cross
the FFI boundary without a dtype conversion; other kernels calculate in
`float64`, after which the wrapper applies OpenCV-style rounding and saturation
for integer outputs. Each synchronous call keeps strong Python references to
all image, coefficient, result, and scratch arrays until the Mojo function
returns. The shared library stores no pointer and performs no allocation, so
no buffer lifetime crosses a call boundary.

The feature pipeline keeps the expensive Sobel, covariance, non-maximum
suppression, and edge-hysteresis loops in Mojo. Small matrix construction,
dtype conversion, and good-feature sorting remain in Python/NumPy.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

Always use the Pixi benchmark task: it takes a machine-wide file lock so
concurrent jobs do not intentionally overlap benchmark runs.

Licensed under the MIT License.
