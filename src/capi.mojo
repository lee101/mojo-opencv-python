"""Image-processing kernels exposed through a compact C ABI."""

from std.math import floor, sqrt
from std.sys.info import simd_width_of

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime U8Ptr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def p_u8(addr: Int) -> U8Ptr:
    return U8Ptr(unsafe_from_address=addr)


def border_index(i: Int, n: Int, mode: Int) -> Int:
    if i >= 0 and i < n:
        return i
    if mode == 0:
        return -1
    if n == 1:
        return 0
    if mode == 1:
        return 0 if i < 0 else n - 1
    if mode == 3:
        var j = i % n
        return j + n if j < 0 else j
    var j = i
    if mode == 2:
        while j < 0 or j >= n:
            j = -j - 1 if j < 0 else 2 * n - j - 1
    else:
        while j < 0 or j >= n:
            j = -j if j < 0 else 2 * n - j - 2
    return j


def sample(
    src: Ptr, y: Int, x: Int, ch: Int, h: Int, w: Int, c: Int,
    border: Int, border_value: Float64
) -> Float64:
    var yy = border_index(y, h, border)
    var xx = border_index(x, w, border)
    if yy < 0 or xx < 0:
        return border_value
    return src[unsafe_offset=(yy * w + xx) * c + ch]


def sample_u8(
    src: U8Ptr, y: Int, x: Int, ch: Int, h: Int, w: Int, c: Int,
    border: Int, border_value: UInt8
) -> UInt8:
    var yy = border_index(y, h, border)
    var xx = border_index(x, w, border)
    if yy < 0 or xx < 0:
        return border_value
    return src[unsafe_offset=(yy * w + xx) * c + ch]


def correlate(
    src: Ptr, dst: Ptr, kernel: Ptr, h: Int, w: Int, c: Int,
    kh: Int, kw: Int, ay: Int, ax: Int, delta: Float64,
    border: Int, border_value: Float64
):
    @__parameter
    def row(y: Int):
        var interior_y = y >= ay and y < h - (kh - ay - 1)
        var left = ax if interior_y else w
        var right = w - (kw - ax - 1) if interior_y else w
        for x in range(left):
            for ch in range(c):
                var acc = delta
                for ky in range(kh):
                    for kx in range(kw):
                        acc += kernel[unsafe_offset=ky * kw + kx] * sample(
                            src, y + ky - ay, x + kx - ax, ch,
                            h, w, c, border, border_value
                        )
                dst[unsafe_offset=(y * w + x) * c + ch] = acc
        var begin = left * c
        var end = right * c
        var i = begin
        while i + W <= end:
            var acc_vec = SIMD[DType.float64, W](delta)
            for ky in range(kh):
                var source_row = (y + ky - ay) * w * c
                for kx in range(kw):
                    acc_vec += kernel[unsafe_offset=ky * kw + kx] * src.unsafe_load[width=W](
                        source_row + (left + kx - ax) * c + i - begin
                    )
            dst.unsafe_store(y * w * c + i, acc_vec)
            i += W
        while i < end:
            var x = i // c
            var ch = i - x * c
            var acc_tail = delta
            for ky in range(kh):
                for kx in range(kw):
                    acc_tail += kernel[unsafe_offset=ky * kw + kx] * src[unsafe_offset=
                        ((y + ky - ay) * w + x + kx - ax) * c + ch
                    ]
            dst[unsafe_offset=y * w * c + i] = acc_tail
            i += 1
        for x in range(right, w):
            for ch in range(c):
                var acc = delta
                for ky in range(kh):
                    for kx in range(kw):
                        acc += kernel[unsafe_offset=ky * kw + kx] * sample(
                            src, y + ky - ay, x + kx - ax, ch,
                            h, w, c, border, border_value
                        )
                dst[unsafe_offset=(y * w + x) * c + ch] = acc

    for y in range(h):
        row(y)


def correlate_u8(
    src: U8Ptr, dst: Ptr, kernel: Ptr, h: Int, w: Int, c: Int,
    kh: Int, kw: Int, ay: Int, ax: Int, delta: Float64,
    border: Int, border_value: UInt8
):
    @__parameter
    def row(y: Int):
        var interior_y = y >= ay and y < h - (kh - ay - 1)
        var left = ax if interior_y else w
        var right = w - (kw - ax - 1) if interior_y else w
        for x in range(left):
            for ch in range(c):
                var acc = delta
                for ky in range(kh):
                    for kx in range(kw):
                        acc += kernel[unsafe_offset=ky * kw + kx] * Float64(sample_u8(
                            src, y + ky - ay, x + kx - ax, ch,
                            h, w, c, border, border_value
                        ))
                dst[unsafe_offset=(y * w + x) * c + ch] = acc
        var begin = left * c
        var end = right * c
        var i = begin
        while i + W <= end:
            var acc_vec = SIMD[DType.float64, W](delta)
            for ky in range(kh):
                var source_row = (y + ky - ay) * w * c
                for kx in range(kw):
                    var values = src.unsafe_load[width=W](
                        source_row + (left + kx - ax) * c + i - begin
                    ).cast[DType.float64]()
                    acc_vec += kernel[unsafe_offset=ky * kw + kx] * values
            dst.unsafe_store(y * w * c + i, acc_vec)
            i += W
        while i < end:
            var x = i // c
            var ch = i - x * c
            var acc_tail = delta
            for ky in range(kh):
                for kx in range(kw):
                    acc_tail += kernel[unsafe_offset=ky * kw + kx] * Float64(src[unsafe_offset=
                        ((y + ky - ay) * w + x + kx - ax) * c + ch
                    ])
            dst[unsafe_offset=y * w * c + i] = acc_tail
            i += 1
        for x in range(right, w):
            for ch in range(c):
                var acc = delta
                for ky in range(kh):
                    for kx in range(kw):
                        acc += kernel[unsafe_offset=ky * kw + kx] * Float64(sample_u8(
                            src, y + ky - ay, x + kx - ax, ch,
                            h, w, c, border, border_value
                        ))
                dst[unsafe_offset=(y * w + x) * c + ch] = acc

    for y in range(h):
        row(y)


@export("mcv_filter2d")
def mcv_filter2d(
    src: Int, dst: Int, kernel: Int, h: Int, w: Int, c: Int,
    kh: Int, kw: Int, ay: Int, ax: Int, delta: Float64,
    border: Int, border_value: Float64
) abi("C"):
    correlate(
        p(src), p(dst), p(kernel), h, w, c, kh, kw, ay, ax,
        delta, border, border_value
    )


@export("mcv_filter2d_u8")
def mcv_filter2d_u8(
    src: Int, dst: Int, kernel: Int, h: Int, w: Int, c: Int,
    kh: Int, kw: Int, ay: Int, ax: Int, delta: Float64,
    border: Int, border_value: Int
) abi("C"):
    correlate_u8(
        p_u8(src), p(dst), p(kernel), h, w, c, kh, kw, ay, ax,
        delta, border, UInt8(border_value)
    )


@export("mcv_median")
def mcv_median(
    src: Int, dst: Int, scratch: Int, h: Int, w: Int, c: Int, ksize: Int
) abi("C"):
    var s = p(src)
    var d = p(dst)
    var work = p(scratch)
    var r = ksize // 2
    var count = ksize * ksize
    @__parameter
    def row(y: Int):
        var row_work = work.unsafe_offset(y * count)
        for x in range(w):
            for ch in range(c):
                var n = 0
                for ky in range(ksize):
                    for kx in range(ksize):
                        var value = sample(
                            s, y + ky - r, x + kx - r, ch, h, w, c, 1, 0.0
                        )
                        var j = n
                        while j > 0 and row_work[unsafe_offset=j - 1] > value:
                            row_work[unsafe_offset=j] = row_work[unsafe_offset=j - 1]
                            j -= 1
                        row_work[unsafe_offset=j] = value
                        n += 1
                d[unsafe_offset=(y * w + x) * c + ch] = row_work[unsafe_offset=count // 2]

    for y in range(h):
        row(y)


@export("mcv_median_u8")
def mcv_median_u8(
    src: Int, dst: Int, scratch: Int, h: Int, w: Int, c: Int, ksize: Int
) abi("C"):
    var s = p_u8(src)
    var d = p_u8(dst)
    var work = p_u8(scratch)
    var r = ksize // 2
    var count = ksize * ksize
    @__parameter
    def row(y: Int):
        var row_work = work.unsafe_offset(y * count)
        for x in range(w):
            for ch in range(c):
                var n = 0
                for ky in range(ksize):
                    for kx in range(ksize):
                        var value = sample_u8(
                            s, y + ky - r, x + kx - r, ch, h, w, c,
                            1, UInt8(0)
                        )
                        var j = n
                        while j > 0 and row_work[unsafe_offset=j - 1] > value:
                            row_work[unsafe_offset=j] = row_work[unsafe_offset=j - 1]
                            j -= 1
                        row_work[unsafe_offset=j] = value
                        n += 1
                d[unsafe_offset=(y * w + x) * c + ch] = row_work[unsafe_offset=count // 2]

    for y in range(h):
        row(y)


def interp(
    src: Ptr, sy: Float64, sx: Float64, ch: Int, h: Int, w: Int, c: Int,
    interpolation: Int, border: Int, border_value: Float64
) -> Float64:
    if interpolation == 0:
        return sample(
            src, Int(floor(sy)), Int(floor(sx)), ch, h, w, c, border, border_value
        )
    if interpolation == 2:
        return sample(
            src, Int(floor(sy + 0.5)), Int(floor(sx + 0.5)), ch,
            h, w, c, border, border_value
        )
    var y0 = Int(floor(sy))
    var x0 = Int(floor(sx))
    var fy = sy - Float64(y0)
    var fx = sx - Float64(x0)
    var a = sample(src, y0, x0, ch, h, w, c, border, border_value)
    var b = sample(src, y0, x0 + 1, ch, h, w, c, border, border_value)
    var q = sample(src, y0 + 1, x0, ch, h, w, c, border, border_value)
    var d = sample(src, y0 + 1, x0 + 1, ch, h, w, c, border, border_value)
    return (a * (1.0 - fx) + b * fx) * (1.0 - fy) + (
        q * (1.0 - fx) + d * fx
    ) * fy


def interp_u8(
    src: U8Ptr, sy: Float64, sx: Float64, ch: Int, h: Int, w: Int, c: Int,
    interpolation: Int
) -> Float64:
    if interpolation == 0:
        return Float64(sample_u8(
            src, Int(floor(sy)), Int(floor(sx)), ch, h, w, c, 1, UInt8(0)
        ))
    var y0 = Int(floor(sy))
    var x0 = Int(floor(sx))
    var fy = sy - Float64(y0)
    var fx = sx - Float64(x0)
    var a = Float64(sample_u8(src, y0, x0, ch, h, w, c, 1, UInt8(0)))
    var b = Float64(sample_u8(src, y0, x0 + 1, ch, h, w, c, 1, UInt8(0)))
    var q = Float64(sample_u8(src, y0 + 1, x0, ch, h, w, c, 1, UInt8(0)))
    var d = Float64(sample_u8(src, y0 + 1, x0 + 1, ch, h, w, c, 1, UInt8(0)))
    return (a * (1.0 - fx) + b * fx) * (1.0 - fy) + (
        q * (1.0 - fx) + d * fx
    ) * fy


@export("mcv_resize")
def mcv_resize(
    src: Int, dst: Int, sh: Int, sw: Int, dh: Int, dw: Int,
    c: Int, interpolation: Int
) abi("C"):
    var s = p(src)
    var d = p(dst)
    var scale_y = Float64(sh) / Float64(dh)
    var scale_x = Float64(sw) / Float64(dw)
    @__parameter
    def row(y: Int):
        for x in range(dw):
            var sy = Float64(y) * scale_y
            var sx = Float64(x) * scale_x
            if interpolation != 0:
                sy = (Float64(y) + 0.5) * scale_y - 0.5
                sx = (Float64(x) + 0.5) * scale_x - 0.5
            for ch in range(c):
                d[unsafe_offset=(y * dw + x) * c + ch] = interp(
                    s, sy, sx, ch, sh, sw, c, interpolation, 1, 0.0
                )

    for y in range(dh):
        row(y)


@export("mcv_resize_u8")
def mcv_resize_u8(
    src: Int, dst: Int, sh: Int, sw: Int, dh: Int, dw: Int,
    c: Int, interpolation: Int
) abi("C"):
    var s = p_u8(src)
    var d = p(dst)
    var scale_y = Float64(sh) / Float64(dh)
    var scale_x = Float64(sw) / Float64(dw)
    @__parameter
    def row(y: Int):
        for x in range(dw):
            var sy = Float64(y) * scale_y
            var sx = Float64(x) * scale_x
            if interpolation != 0:
                sy = (Float64(y) + 0.5) * scale_y - 0.5
                sx = (Float64(x) + 0.5) * scale_x - 0.5
            for ch in range(c):
                d[unsafe_offset=(y * dw + x) * c + ch] = interp_u8(
                    s, sy, sx, ch, sh, sw, c, interpolation
                )

    for y in range(dh):
        row(y)


@export("mcv_warp")
def mcv_warp(
    src: Int, dst: Int, matrix: Int, sh: Int, sw: Int, dh: Int, dw: Int,
    c: Int, perspective: Int, interpolation: Int, border: Int,
    border_value: Float64
) abi("C"):
    var s = p(src)
    var d = p(dst)
    var m = p(matrix)
    @__parameter
    def row(y: Int):
        for x in range(dw):
            var den = 1.0
            if perspective != 0:
                den = m[unsafe_offset=6] * Float64(x) + m[unsafe_offset=7] * Float64(y) + m[unsafe_offset=8]
            var sx = (m[unsafe_offset=0] * Float64(x) + m[unsafe_offset=1] * Float64(y) + m[unsafe_offset=2]) / den
            var sy = (m[unsafe_offset=3] * Float64(x) + m[unsafe_offset=4] * Float64(y) + m[unsafe_offset=5]) / den
            var interp_mode = interpolation
            if interpolation == 0:
                interp_mode = 2
            else:
                sx = floor(sx * 32.0 + 0.5) / 32.0
                sy = floor(sy * 32.0 + 0.5) / 32.0
            for ch in range(c):
                d[unsafe_offset=(y * dw + x) * c + ch] = interp(
                    s, sy, sx, ch, sh, sw, c, interp_mode, border, border_value
                )

    for y in range(dh):
        row(y)


@export("mcv_morph")
def mcv_morph(
    src: Int, dst: Int, kernel: Int, h: Int, w: Int, c: Int,
    kh: Int, kw: Int, ay: Int, ax: Int, operation: Int,
    border: Int, border_value: Float64
) abi("C"):
    var s = p(src)
    var d = p(dst)
    var k = p(kernel)
    @__parameter
    def row(y: Int):
        var interior_y = y >= ay and y < h - (kh - ay - 1)
        var left = ax if interior_y else w
        var right = w - (kw - ax - 1) if interior_y else w
        for x in range(left):
            for ch in range(c):
                var best = 1.7976931348623157e308 if operation == 0 else -1.7976931348623157e308
                for ky in range(kh):
                    for kx in range(kw):
                        if k[unsafe_offset=ky * kw + kx] == 0.0:
                            continue
                        var value = sample(
                            s, y + ky - ay, x + kx - ax, ch,
                            h, w, c, border, border_value
                        )
                        if operation == 0:
                            if value < best:
                                best = value
                        else:
                            if value > best:
                                best = value
                d[unsafe_offset=(y * w + x) * c + ch] = best
        var begin = left * c
        var end = right * c
        var i = begin
        while i + W <= end:
            var best_vec = SIMD[DType.float64, W](
                1.7976931348623157e308 if operation == 0 else -1.7976931348623157e308
            )
            for ky in range(kh):
                var source_row = (y + ky - ay) * w * c
                for kx in range(kw):
                    if k[unsafe_offset=ky * kw + kx] != 0.0:
                        var value = s.unsafe_load[width=W](
                            source_row + (left + kx - ax) * c + i - begin
                        )
                        best_vec = min(best_vec, value) if operation == 0 else max(best_vec, value)
            d.unsafe_store(y * w * c + i, best_vec)
            i += W
        while i < end:
            var x = i // c
            var ch = i - x * c
            var best_tail = 1.7976931348623157e308 if operation == 0 else -1.7976931348623157e308
            for ky in range(kh):
                for kx in range(kw):
                    if k[unsafe_offset=ky * kw + kx] != 0.0:
                        var value = s[unsafe_offset=
                            ((y + ky - ay) * w + x + kx - ax) * c + ch
                        ]
                        best_tail = (
                            min(best_tail, value) if operation == 0
                            else max(best_tail, value)
                        )
            d[unsafe_offset=y * w * c + i] = best_tail
            i += 1
        for x in range(right, w):
            for ch in range(c):
                var best = 1.7976931348623157e308 if operation == 0 else -1.7976931348623157e308
                for ky in range(kh):
                    for kx in range(kw):
                        if k[unsafe_offset=ky * kw + kx] == 0.0:
                            continue
                        var value = sample(
                            s, y + ky - ay, x + kx - ax, ch,
                            h, w, c, border, border_value
                        )
                        best = min(best, value) if operation == 0 else max(best, value)
                d[unsafe_offset=(y * w + x) * c + ch] = best

    for y in range(h):
        row(y)


def gray_sample(src: Ptr, y: Int, x: Int, h: Int, w: Int) -> Float64:
    return sample(src, y, x, 0, h, w, 1, 4, 0.0)


def sobel_x(src: Ptr, y: Int, x: Int, h: Int, w: Int) -> Float64:
    return (
        -gray_sample(src, y - 1, x - 1, h, w)
        + gray_sample(src, y - 1, x + 1, h, w)
        - 2.0 * gray_sample(src, y, x - 1, h, w)
        + 2.0 * gray_sample(src, y, x + 1, h, w)
        - gray_sample(src, y + 1, x - 1, h, w)
        + gray_sample(src, y + 1, x + 1, h, w)
    )


def sobel_y(src: Ptr, y: Int, x: Int, h: Int, w: Int) -> Float64:
    return (
        -gray_sample(src, y - 1, x - 1, h, w)
        - 2.0 * gray_sample(src, y - 1, x, h, w)
        - gray_sample(src, y - 1, x + 1, h, w)
        + gray_sample(src, y + 1, x - 1, h, w)
        + 2.0 * gray_sample(src, y + 1, x, h, w)
        + gray_sample(src, y + 1, x + 1, h, w)
    )


@export("mcv_corner")
def mcv_corner(
    src: Int, dst: Int, h: Int, w: Int, block_size: Int,
    k: Float64, min_eigen: Int, input_scale: Float64
) abi("C"):
    var s = p(src)
    var d = p(dst)
    var r = block_size // 2
    var scale = input_scale / (4.0 * Float64(block_size))
    @__parameter
    def row(y: Int):
        for x in range(w):
            var a = 0.0
            var b = 0.0
            var q = 0.0
            for yy in range(y - r, y + block_size - r):
                for xx in range(x - r, x + block_size - r):
                    var yi = border_index(yy, h, 4)
                    var xi = border_index(xx, w, 4)
                    var gx = sobel_x(s, yi, xi, h, w) * scale
                    var gy = sobel_y(s, yi, xi, h, w) * scale
                    a += gx * gx
                    b += gx * gy
                    q += gy * gy
            if min_eigen != 0:
                d[unsafe_offset=y * w + x] = 0.5 * (a + q - sqrt((a - q) * (a - q) + 4.0 * b * b))
            else:
                d[unsafe_offset=y * w + x] = a * q - b * b - k * (a + q) * (a + q)

    for y in range(h):
        row(y)


@export("mcv_canny")
def mcv_canny(
    src: Int, dst: Int, mag: Int, gx_addr: Int, gy_addr: Int, state: Int,
    queue: Int, h: Int, w: Int, low: Float64, high: Float64, l2: Int
) abi("C"):
    var s = p(src)
    var d = p(dst)
    var m = p(mag)
    var gx = p(gx_addr)
    var gy = p(gy_addr)
    var st = p(state)
    var q = p(queue)
    @__parameter
    def gradient_row(y: Int):
        for x in range(w):
            var i = y * w + x
            gx[unsafe_offset=i] = sobel_x(s, y, x, h, w)
            gy[unsafe_offset=i] = sobel_y(s, y, x, h, w)
            if l2 != 0:
                m[unsafe_offset=i] = sqrt(gx[unsafe_offset=i] * gx[unsafe_offset=i] + gy[unsafe_offset=i] * gy[unsafe_offset=i])
            else:
                m[unsafe_offset=i] = abs(gx[unsafe_offset=i]) + abs(gy[unsafe_offset=i])
            st[unsafe_offset=i] = 0.0
            d[unsafe_offset=i] = 0.0

    for y in range(h):
        gradient_row(y)
    var tail = 0
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            var i = y * w + x
            var ax = abs(gx[unsafe_offset=i])
            var ay = abs(gy[unsafe_offset=i])
            var n1 = 0.0
            var n2 = 0.0
            var keep = False
            if ay * 2.414213562373095 < ax:
                n1 = m[unsafe_offset=i - 1]
                n2 = m[unsafe_offset=i + 1]
                keep = m[unsafe_offset=i] > n1 and m[unsafe_offset=i] >= n2
            elif ax * 2.414213562373095 < ay:
                n1 = m[unsafe_offset=i - w]
                n2 = m[unsafe_offset=i + w]
                keep = m[unsafe_offset=i] > n1 and m[unsafe_offset=i] >= n2
            elif gx[unsafe_offset=i] * gy[unsafe_offset=i] > 0.0:
                n1 = m[unsafe_offset=i - w - 1]
                n2 = m[unsafe_offset=i + w + 1]
                keep = m[unsafe_offset=i] > n1 and m[unsafe_offset=i] > n2
            else:
                n1 = m[unsafe_offset=i - w + 1]
                n2 = m[unsafe_offset=i + w - 1]
                keep = m[unsafe_offset=i] > n1 and m[unsafe_offset=i] > n2
            if keep and m[unsafe_offset=i] >= low:
                st[unsafe_offset=i] = 1.0
                if m[unsafe_offset=i] >= high:
                    st[unsafe_offset=i] = 2.0
                    q[unsafe_offset=tail] = Float64(i)
                    tail += 1
    var head = 0
    while head < tail:
        var i = Int(q[unsafe_offset=head])
        head += 1
        d[unsafe_offset=i] = 255.0
        var y = i // w
        var x = i - y * w
        for yy in range(y - 1, y + 2):
            for xx in range(x - 1, x + 2):
                var j = yy * w + xx
                if yy >= 0 and yy < h and xx >= 0 and xx < w and st[unsafe_offset=j] == 1.0:
                    st[unsafe_offset=j] = 2.0
                    q[unsafe_offset=tail] = Float64(j)
                    tail += 1
