"""A Mojo-accelerated, OpenCV-compatible image-processing subset."""

from __future__ import annotations

import math

import numpy as np

from ._lib import addr, f64, lib, rows

__version__ = "0.1.0"

CV_8U = 0
CV_8S = 1
CV_16U = 2
CV_16S = 3
CV_32S = 4
CV_32F = 5
CV_64F = 6

BORDER_CONSTANT = 0
BORDER_REPLICATE = 1
BORDER_REFLECT = 2
BORDER_WRAP = 3
BORDER_REFLECT_101 = 4
BORDER_TRANSPARENT = 5
BORDER_DEFAULT = BORDER_REFLECT_101
BORDER_ISOLATED = 16

INTER_NEAREST = 0
INTER_LINEAR = 1
WARP_INVERSE_MAP = 16

MORPH_ERODE = 0
MORPH_DILATE = 1
MORPH_OPEN = 2
MORPH_CLOSE = 3
MORPH_GRADIENT = 4
MORPH_TOPHAT = 5
MORPH_BLACKHAT = 6
MORPH_HITMISS = 7
MORPH_RECT = 0
MORPH_CROSS = 1
MORPH_ELLIPSE = 2

_SOURCE_DTYPES = {
    np.dtype(np.uint8),
    np.dtype(np.uint16),
    np.dtype(np.int16),
    np.dtype(np.float32),
    np.dtype(np.float64),
}
_BORDER_MODES = {
    BORDER_CONSTANT,
    BORDER_REPLICATE,
    BORDER_REFLECT,
    BORDER_WRAP,
    BORDER_REFLECT_101,
}


def _image(src):
    original = np.asarray(src)
    if original.ndim not in (2, 3):
        raise ValueError("src must be a 2D image or a channel-last 3D image")
    if original.dtype not in _SOURCE_DTYPES:
        raise TypeError(
            "src dtype must be uint8, uint16, int16, float32, or float64"
        )
    if any(size <= 0 for size in original.shape):
        raise ValueError("src dimensions and channel count must be positive")
    work = f64(original)
    h, w = work.shape[:2]
    channels = 1 if work.ndim == 2 else work.shape[2]
    return original, work, h, w, channels


def _size2(value, name, *, allow_zero=False):
    try:
        width, height = map(int, value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must contain two integer dimensions") from None
    minimum = 0 if allow_zero else 1
    if width < minimum or height < minimum:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} dimensions must be {qualifier}")
    return width, height


def _border(value):
    mode = int(value)
    isolated = mode & BORDER_ISOLATED
    base = mode & ~BORDER_ISOLATED
    if base not in _BORDER_MODES:
        raise NotImplementedError("unsupported border mode")
    return base, isolated


def _finite_array(value, name):
    result = f64(value)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _anchor(anchor, kernel_shape):
    ax, ay = anchor
    kh, kw = kernel_shape
    if ax < 0:
        ax = kw // 2
    if ay < 0:
        ay = kh // 2
    if not (0 <= ax < kw and 0 <= ay < kh):
        raise ValueError("anchor lies outside kernel")
    return int(ax), int(ay)


def _dtype_for_depth(depth, source_dtype):
    if depth == -1:
        return source_dtype
    table = {
        CV_8U: np.uint8,
        CV_8S: np.int8,
        CV_16U: np.uint16,
        CV_16S: np.int16,
        CV_32S: np.int32,
        CV_32F: np.float32,
        CV_64F: np.float64,
    }
    if depth not in table:
        raise NotImplementedError("unsupported output depth")
    return np.dtype(table[depth])


def _cast_output(values, dtype, dst=None):
    dtype = np.dtype(dtype)
    if dtype.kind in "ui":
        info = np.iinfo(dtype)
        values = np.clip(np.rint(values), info.min, info.max)
    result = values.astype(dtype, copy=False)
    if dst is not None:
        target = np.asarray(dst)
        if target.shape != result.shape or target.dtype != result.dtype:
            raise ValueError("dst has incompatible shape or dtype")
        target[...] = result
        return target
    return result


def filter2D(
    src, ddepth, kernel, dst=None, anchor=(-1, -1), delta=0,
    borderType=BORDER_DEFAULT
):
    original, converted, h, w, channels = _image(src)
    native_u8 = original.dtype == np.uint8 and original.ndim in (2, 3)
    if native_u8:
        source = np.ascontiguousarray(original)
    else:
        source = converted
    weights = _finite_array(kernel, "kernel")
    if weights.ndim != 2 or not all(n > 0 for n in weights.shape):
        raise ValueError("kernel must be a non-empty 2D array")
    ax, ay = _anchor(anchor, weights.shape)
    border, _ = _border(borderType)
    if border == BORDER_WRAP:
        raise NotImplementedError("filter2D does not support BORDER_WRAP")
    result = np.empty(source.shape, dtype=np.float64)
    common = (
        addr(source), addr(result), addr(weights), h, w, channels,
        weights.shape[0], weights.shape[1], ay, ax,
    )
    ops = w * channels * weights.shape[0] * weights.shape[1] * 2
    if native_u8:
        rows(
            lambda y0, y1: lib().mcv_filter2d_u8(
                *common, float(delta), border, 0, y0, y1
            ),
            h, ops,
        )
    else:
        rows(
            lambda y0, y1: lib().mcv_filter2d(
                *common, float(delta), border, 0.0, y0, y1
            ),
            h, ops,
        )
    return _cast_output(result, _dtype_for_depth(ddepth, original.dtype), dst)


def sepFilter2D(
    src, ddepth, kernelX, kernelY, dst=None, anchor=(-1, -1), delta=0,
    borderType=BORDER_DEFAULT
):
    original, converted, h, w, channels = _image(src)
    source = np.ascontiguousarray(original) if original.dtype == np.uint8 else converted
    kx = _finite_array(kernelX, "kernelX").reshape(-1)
    ky = _finite_array(kernelY, "kernelY").reshape(-1)
    if not kx.size or not ky.size:
        raise ValueError("separable kernels must not be empty")
    ax, ay = _anchor(anchor, (ky.size, kx.size))
    border, _ = _border(borderType)
    if border == BORDER_WRAP:
        raise NotImplementedError("sepFilter2D does not support BORDER_WRAP")
    scratch = np.empty(source.shape, dtype=np.float64)
    result = np.empty(source.shape, dtype=np.float64)
    common = (
        addr(source), addr(result), addr(scratch), addr(kx), addr(ky),
        h, w, channels, ky.size, kx.size, ay, ax, float(delta), border,
    )
    if original.dtype == np.uint8:
        lib().mcv_sepfilter2d_u8(*common, 0)
    else:
        lib().mcv_sepfilter2d(*common, 0.0)
    return _cast_output(result, _dtype_for_depth(ddepth, original.dtype), dst)


def boxFilter(
    src, ddepth, ksize, dst=None, anchor=(-1, -1), normalize=True,
    borderType=BORDER_DEFAULT
):
    kw, kh = _size2(ksize, "ksize")
    kernel = np.ones((kh, kw), dtype=np.float64)
    if normalize:
        kernel /= kh * kw
    return filter2D(src, ddepth, kernel, dst, anchor, 0, borderType)


def blur(src, ksize, dst=None, anchor=(-1, -1), borderType=BORDER_DEFAULT):
    return boxFilter(src, -1, ksize, dst, anchor, True, borderType)


def _gaussian_kernel(n, sigma):
    if n <= 0 or n % 2 == 0:
        raise ValueError("Gaussian kernel dimensions must be positive and odd")
    if sigma <= 0:
        sigma = 0.3 * ((n - 1) * 0.5 - 1) + 0.8
    x = np.arange(n, dtype=np.float64) - (n - 1) * 0.5
    values = np.exp(-(x * x) / (2 * sigma * sigma))
    return values / values.sum()


def getGaussianKernel(ksize, sigma, ktype=CV_64F):
    if ktype not in (CV_32F, CV_64F):
        raise NotImplementedError("getGaussianKernel supports CV_32F and CV_64F")
    dtype = np.float32 if ktype == CV_32F else np.float64
    return _gaussian_kernel(int(ksize), float(sigma)).astype(dtype)[:, None]


def GaussianBlur(
    src, ksize, sigmaX, dst=None, sigmaY=0, borderType=BORDER_DEFAULT,
    hint=0
):
    kw, kh = _size2(ksize, "ksize", allow_zero=True)
    if kw == 0 and float(sigmaX) <= 0:
        raise ValueError("sigmaX must be positive when kernel width is zero")
    if kh == 0 and float(sigmaY if sigmaY > 0 else sigmaX) <= 0:
        raise ValueError("sigma must be positive when kernel height is zero")
    if kw == 0 or kh == 0:
        sigma_y = sigmaX if sigmaY <= 0 else sigmaY
        if kw == 0:
            kw = int(round(float(sigmaX) * 6 + 1)) | 1
        if kh == 0:
            kh = int(round(float(sigma_y) * 6 + 1)) | 1
    sigma_y = sigmaX if sigmaY <= 0 else sigmaY
    return sepFilter2D(
        src, -1, _gaussian_kernel(kw, sigmaX),
        _gaussian_kernel(kh, sigma_y), dst, (-1, -1), 0, borderType,
    )


def medianBlur(src, ksize, dst=None):
    ksize = int(ksize)
    if ksize <= 1 or ksize % 2 == 0:
        raise ValueError("ksize must be odd and greater than one")
    original, converted, h, w, channels = _image(src)
    if original.dtype == np.uint8 and original.ndim in (2, 3):
        source = np.ascontiguousarray(original)
        result = np.empty_like(source)
        scratch = np.empty((h, 256), dtype=np.uint32)
        rows(
            lambda y0, y1: lib().mcv_median_u8(
                addr(source), addr(result), addr(scratch), h, w, channels,
                ksize, y0, y1,
            ),
            h, w * channels * ksize * ksize * 6,
        )
    else:
        source = converted
        result = np.empty_like(source)
        scratch = np.empty((h, ksize * ksize), dtype=np.float64)
        rows(
            lambda y0, y1: lib().mcv_median(
                addr(source), addr(result), addr(scratch), h, w, channels,
                ksize, y0, y1,
            ),
            h, w * channels * ksize * ksize * 6,
        )
    return _cast_output(result, original.dtype, dst)


def Sobel(
    src, ddepth, dx, dy, dst=None, ksize=3, scale=1, delta=0,
    borderType=BORDER_DEFAULT
):
    dx, dy, ksize = int(dx), int(dy), int(ksize)
    if (
        ksize not in (1, 3)
        or dx < 0
        or dy < 0
        or dx + dy not in (1, 2)
    ):
        raise NotImplementedError("Sobel currently supports derivative orders 1-2 and ksize 1 or 3")
    smooth = np.array([1.0]) if ksize == 1 else np.array([1.0, 2.0, 1.0])
    first = np.array([-1.0, 0.0, 1.0])
    second = np.array([1.0, -2.0, 1.0])
    kx = smooth if dx == 0 else (first if dx == 1 else second)
    ky = smooth if dy == 0 else (first if dy == 1 else second)
    return filter2D(
        src, ddepth, np.outer(ky, kx) * float(scale), dst,
        (-1, -1), delta, borderType,
    )


def Scharr(
    src, ddepth, dx, dy, dst=None, scale=1, delta=0,
    borderType=BORDER_DEFAULT
):
    if (dx, dy) == (1, 0):
        kernel = np.outer([3.0, 10.0, 3.0], [-1.0, 0.0, 1.0])
    elif (dx, dy) == (0, 1):
        kernel = np.outer([-1.0, 0.0, 1.0], [3.0, 10.0, 3.0])
    else:
        raise ValueError("Scharr requires (dx, dy) to be (1, 0) or (0, 1)")
    return filter2D(src, ddepth, kernel * scale, dst, (-1, -1), delta, borderType)


def Laplacian(
    src, ddepth, dst=None, ksize=1, scale=1, delta=0,
    borderType=BORDER_DEFAULT
):
    if ksize == 1:
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
    elif ksize == 3:
        kernel = np.array([[2, 0, 2], [0, -8, 0], [2, 0, 2]], dtype=float)
    else:
        raise NotImplementedError("Laplacian currently supports ksize 1 or 3")
    return filter2D(src, ddepth, kernel * scale, dst, (-1, -1), delta, borderType)


def resize(src, dsize, dst=None, fx=0, fy=0, interpolation=INTER_LINEAR):
    original, converted, sh, sw, channels = _image(src)
    native_u8 = original.dtype == np.uint8 and original.ndim in (2, 3)
    if native_u8:
        source = np.ascontiguousarray(original)
    else:
        source = converted
    dw, dh = _size2(dsize, "dsize", allow_zero=True)
    if (dw == 0) != (dh == 0):
        raise ValueError("dsize dimensions must both be zero or both be positive")
    if dw == 0:
        if fx <= 0 or fy <= 0:
            raise ValueError("either dsize or both fx and fy must be positive")
        dw, dh = int(round(sw * fx)), int(round(sh * fy))
        if dw <= 0 or dh <= 0:
            raise ValueError("fx and fy produce an empty destination")
    if interpolation not in (INTER_NEAREST, INTER_LINEAR):
        raise NotImplementedError("resize supports INTER_NEAREST and INTER_LINEAR")
    shape = (dh, dw) if source.ndim == 2 else (dh, dw, channels)
    result = np.empty(shape, dtype=np.float64)
    if native_u8 and interpolation == INTER_LINEAR:
        sx = (np.arange(dw, dtype=np.float64) + 0.5) * sw / dw - 0.5
        sy = (np.arange(dh, dtype=np.float64) + 0.5) * sh / dh - 0.5
        x0 = np.floor(sx).astype(np.int64)
        y0 = np.floor(sy).astype(np.int64)
        xa = np.ascontiguousarray(np.clip(x0, 0, sw - 1))
        xb = np.ascontiguousarray(np.clip(x0 + 1, 0, sw - 1))
        ya = np.ascontiguousarray(np.clip(y0, 0, sh - 1))
        yb = np.ascontiguousarray(np.clip(y0 + 1, 0, sh - 1))
        fx_values = np.ascontiguousarray(sx - x0)
        fy_values = np.ascontiguousarray(sy - y0)
        rows(
            lambda y0, y1: lib().mcv_resize_linear_u8(
                addr(source), addr(result), addr(xa), addr(xb), addr(fx_values),
                addr(ya), addr(yb), addr(fy_values), sw, dh, dw, channels, y0, y1,
            ),
            dh, dw * channels * 9,
        )
    elif native_u8:
        rows(
            lambda y0, y1: lib().mcv_resize_u8(
                addr(source), addr(result), sh, sw, dh, dw, channels,
                interpolation, y0, y1,
            ),
            dh, dw * channels * 9,
        )
    else:
        rows(
            lambda y0, y1: lib().mcv_resize(
                addr(source), addr(result), sh, sw, dh, dw, channels,
                interpolation, y0, y1,
            ),
            dh, dw * channels * 9,
        )
    return _cast_output(result, original.dtype, dst)


def _border_scalar(value):
    if np.isscalar(value):
        return float(value)
    values = tuple(value)
    if not values:
        return 0.0
    if any(float(v) != float(values[0]) for v in values):
        raise NotImplementedError("per-channel border values are not yet supported")
    return float(values[0])

def _warp(src, matrix, dsize, dst, flags, borderMode, borderValue, perspective):
    original, source, sh, sw, channels = _image(src)
    dw, dh = _size2(dsize, "dsize")
    interpolation = int(flags) & 7
    if interpolation not in (INTER_NEAREST, INTER_LINEAR):
        raise NotImplementedError("warps support INTER_NEAREST and INTER_LINEAR")

    matrix = _finite_array(matrix, "matrix")
    expected = (3, 3) if perspective else (2, 3)
    if matrix.shape != expected:
        raise ValueError(f"matrix must have shape {expected}")
    border, _ = _border(borderMode)
    if not flags & WARP_INVERSE_MAP:
        if perspective:
            matrix = np.ascontiguousarray(np.linalg.inv(matrix))
        else:
            full = np.vstack([matrix, [0, 0, 1]])
            matrix = np.ascontiguousarray(np.linalg.inv(full)[:2])
    if not perspective:
        matrix = np.ascontiguousarray(np.vstack([matrix, [0, 0, 1]]))
    shape = (dh, dw) if source.ndim == 2 else (dh, dw, channels)
    result = np.empty(shape, dtype=np.float64)
    rows(
        lambda y0, y1: lib().mcv_warp(
            addr(source), addr(result), addr(matrix), sh, sw, dh, dw, channels,
            int(perspective), interpolation, border,
            _border_scalar(borderValue), y0, y1,
        ),
        dh, dw * channels * 9,
    )
    return _cast_output(result, original.dtype, dst)


def warpAffine(
    src, M, dsize, dst=None, flags=INTER_LINEAR,
    borderMode=BORDER_CONSTANT, borderValue=0
):
    return _warp(src, M, dsize, dst, flags, borderMode, borderValue, False)


def warpPerspective(
    src, M, dsize, dst=None, flags=INTER_LINEAR,
    borderMode=BORDER_CONSTANT, borderValue=0
):
    return _warp(src, M, dsize, dst, flags, borderMode, borderValue, True)


def getRotationMatrix2D(center, angle, scale):
    alpha = math.cos(math.radians(angle)) * scale
    beta = math.sin(math.radians(angle)) * scale
    cx, cy = center
    return np.array([
        [alpha, beta, (1 - alpha) * cx - beta * cy],
        [-beta, alpha, beta * cx + (1 - alpha) * cy],
    ])


def getAffineTransform(src, dst):
    source = f64(src)
    target = f64(dst)
    if source.shape != (3, 2) or target.shape != (3, 2):
        raise ValueError("src and dst must each have shape (3, 2)")
    a = np.column_stack([source, np.ones(3)])
    return np.linalg.solve(a, target).T


def getPerspectiveTransform(src, dst, solveMethod=0):
    source, target = f64(src), f64(dst)
    if source.shape != (4, 2) or target.shape != (4, 2):
        raise ValueError("src and dst must each have shape (4, 2)")
    a, b = [], []
    for (x, y), (u, v) in zip(source, target):
        a.extend([[x, y, 1, 0, 0, 0, -u * x, -u * y],
                  [0, 0, 0, x, y, 1, -v * x, -v * y]])
        b.extend([u, v])
    h = np.linalg.solve(np.asarray(a), np.asarray(b))
    return np.append(h, 1).reshape(3, 3)


def invertAffineTransform(M):
    matrix = f64(M)
    return np.linalg.inv(np.vstack([matrix, [0, 0, 1]]))[:2]


def getStructuringElement(shape, ksize, anchor=(-1, -1)):
    width, height = _size2(ksize, "ksize")
    ax, ay = _anchor(anchor, (height, width))
    if shape == MORPH_RECT:
        return np.ones((height, width), dtype=np.uint8)
    if shape == MORPH_CROSS:
        result = np.zeros((height, width), dtype=np.uint8)
        result[ay, :] = 1
        result[:, ax] = 1
        return result
    if shape == MORPH_ELLIPSE:
        result = np.zeros((height, width), dtype=np.uint8)
        r = height // 2
        for y in range(height):
            dy = y - r
            if abs(dy) > r:
                continue
            dx = int(round((width // 2) * math.sqrt(max(0, r * r - dy * dy)) / max(r, 1)))
            result[y, max(0, ax - dx):min(width, ax + dx + 1)] = 1
        return result
    raise ValueError("unknown structuring element shape")


def _morph(src, kernel, operation, dst, anchor, iterations, borderType, borderValue):
    original, current, h, w, channels = _image(src)
    weights = _finite_array(kernel, "kernel")
    if weights.ndim != 2 or not np.any(weights):
        raise ValueError("kernel must be a non-empty 2D array with a nonzero element")
    ax, ay = _anchor(anchor, weights.shape)
    iterations = int(iterations)
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    border, _ = _border(borderType)
    if borderValue is None:
        border_value = np.inf if operation == MORPH_ERODE else -np.inf
    else:
        border_value = _border_scalar(borderValue)
    for _ in range(iterations):
        result = np.empty_like(current)
        rows(
            lambda y0, y1: lib().mcv_morph(
                addr(current), addr(result), addr(weights), h, w, channels,
                weights.shape[0], weights.shape[1], ay, ax, operation,
                border, border_value, y0, y1,
            ),
            h, w * channels * weights.shape[0] * weights.shape[1],
        )
        current = result
    return _cast_output(current, original.dtype, dst)


def erode(
    src, kernel, dst=None, anchor=(-1, -1), iterations=1,
    borderType=BORDER_CONSTANT, borderValue=None
):
    return _morph(
        src, kernel, MORPH_ERODE, dst, anchor, iterations, borderType, borderValue
    )


def dilate(
    src, kernel, dst=None, anchor=(-1, -1), iterations=1,
    borderType=BORDER_CONSTANT, borderValue=None
):
    return _morph(
        src, kernel, MORPH_DILATE, dst, anchor, iterations, borderType, borderValue
    )


def morphologyEx(
    src, op, kernel, dst=None, anchor=(-1, -1), iterations=1,
    borderType=BORDER_CONSTANT, borderValue=None
):
    if op == MORPH_ERODE:
        return erode(src, kernel, dst, anchor, iterations, borderType, borderValue)
    if op == MORPH_DILATE:
        return dilate(src, kernel, dst, anchor, iterations, borderType, borderValue)
    eroded = erode(src, kernel, None, anchor, iterations, borderType, borderValue)
    dilated = dilate(src, kernel, None, anchor, iterations, borderType, borderValue)
    if op == MORPH_OPEN:
        result = dilate(eroded, kernel, None, anchor, iterations, borderType, borderValue)
    elif op == MORPH_CLOSE:
        result = erode(dilated, kernel, None, anchor, iterations, borderType, borderValue)
    elif op == MORPH_GRADIENT:
        result = dilated.astype(np.float64) - eroded.astype(np.float64)
    elif op == MORPH_TOPHAT:
        opened = dilate(eroded, kernel, None, anchor, iterations, borderType, borderValue)
        result = np.asarray(src, dtype=np.float64) - opened
    elif op == MORPH_BLACKHAT:
        closed = erode(dilated, kernel, None, anchor, iterations, borderType, borderValue)
        result = closed.astype(np.float64) - np.asarray(src, dtype=np.float64)
    else:
        raise NotImplementedError("MORPH_HITMISS is not covered")
    return _cast_output(result, np.asarray(src).dtype, dst)


def _corner_response(source, result, gx, gy, h, w, block_size, k, min_eigen,
                     input_scale):
    # The Sobel pass and the structure-tensor pass each read rows the other
    # writes, so they are fanned out separately with a barrier between them.
    rows(
        lambda y0, y1: lib().mcv_corner_gradient(
            addr(source), addr(gx), addr(gy), h, w, block_size,
            input_scale, y0, y1,
        ),
        h, w * 12 * 24,
    )
    rows(
        lambda y0, y1: lib().mcv_corner_response(
            addr(result), addr(gx), addr(gy), h, w, block_size,
            k, min_eigen, y0, y1,
        ),
        h, w * block_size * block_size * 6,
    )


def cornerHarris(src, blockSize, ksize, k, dst=None, borderType=BORDER_DEFAULT):
    if int(ksize) != 3 or int(borderType) != BORDER_DEFAULT:
        raise NotImplementedError("cornerHarris currently supports ksize=3 and BORDER_DEFAULT")
    original, source, h, w, channels = _image(src)
    if channels != 1:
        raise ValueError("cornerHarris requires a single-channel image")
    result = np.empty((h, w), dtype=np.float64)
    gx = np.empty((h, w), dtype=np.float64)
    gy = np.empty((h, w), dtype=np.float64)
    input_scale = 1 / 255 if original.dtype == np.uint8 else 1.0
    _corner_response(
        source, result, gx, gy, h, w, int(blockSize), float(k), 0, input_scale
    )
    return _cast_output(result, np.float32, dst)


def cornerMinEigenVal(src, blockSize, dst=None, ksize=3, borderType=BORDER_DEFAULT):
    if int(ksize) != 3 or int(borderType) != BORDER_DEFAULT:
        raise NotImplementedError("cornerMinEigenVal supports ksize=3 and BORDER_DEFAULT")
    original, source, h, w, channels = _image(src)
    if channels != 1:
        raise ValueError("cornerMinEigenVal requires a single-channel image")
    result = np.empty((h, w), dtype=np.float64)
    gx = np.empty((h, w), dtype=np.float64)
    gy = np.empty((h, w), dtype=np.float64)
    input_scale = 1 / 255 if original.dtype == np.uint8 else 1.0
    _corner_response(
        source, result, gx, gy, h, w, int(blockSize), 0.0, 1, input_scale
    )
    return _cast_output(result, np.float32, dst)


def goodFeaturesToTrack(
    image, maxCorners, qualityLevel, minDistance, corners=None, mask=None,
    blockSize=3, useHarrisDetector=False, k=0.04
):
    response = (
        cornerHarris(image, blockSize, 3, k)
        if useHarrisDetector else cornerMinEigenVal(image, blockSize)
    )
    allowed = np.ones(response.shape, dtype=bool) if mask is None else np.asarray(mask) != 0
    padded = np.pad(response, 1, mode="constant", constant_values=-np.inf)
    maxima = allowed.copy()
    for dy in range(3):
        for dx in range(3):
            maxima &= response >= padded[dy:dy + response.shape[0], dx:dx + response.shape[1]]
    threshold = float(qualityLevel) * float(response[allowed].max())
    ys, xs = np.nonzero(maxima & (response > threshold))
    order = np.argsort(response[ys, xs], kind="stable")[::-1]
    selected = []
    distance2 = float(minDistance) ** 2
    for index in order:
        point = (int(xs[index]), int(ys[index]))
        if all((point[0] - x) ** 2 + (point[1] - y) ** 2 >= distance2 for x, y in selected):
            selected.append(point)
            if maxCorners > 0 and len(selected) >= maxCorners:
                break
    result = np.asarray(selected, dtype=np.float32).reshape(-1, 1, 2)
    if corners is not None:
        target = np.asarray(corners)
        target[...] = result
        return target
    return result if result.size else None


def Canny(
    image, threshold1, threshold2, edges=None, apertureSize=3, L2gradient=False
):
    original, source, h, w, channels = _image(image)
    if original.dtype != np.uint8 or channels != 1:
        raise ValueError("Canny requires an 8-bit single-channel image")
    if apertureSize != 3:
        raise NotImplementedError("Canny currently supports apertureSize=3")
    result = np.empty((h, w), dtype=np.float64)
    scratch = [np.empty((h, w), dtype=np.float64) for _ in range(5)]
    low, high = sorted((float(threshold1), float(threshold2)))
    # The Sobel pass feeds non-maximum suppression and hysteresis, which are
    # serial anyway; fanning the gradient rows out measured slower than the
    # serial pass, so Canny stays single-threaded.
    lib().mcv_canny(
        addr(source), addr(result), *(addr(item) for item in scratch),
        h, w, low, high, int(L2gradient), 0, h,
    )
    return _cast_output(result, np.uint8, edges)


__all__ = [name for name in globals() if not name.startswith("_")]
