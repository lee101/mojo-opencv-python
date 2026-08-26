import cv2
import numpy as np
import pytest

import mojo_opencv as mcv

rng = np.random.default_rng(42)


def assert_close(actual, expected, atol=1e-12):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=atol)


@pytest.mark.parametrize(
    "dtype", [np.uint8, np.uint16, np.int16, np.float32, np.float64]
)
def test_filter2d_matches_opencv(dtype):
    image = rng.integers(0, 256, (31, 37, 3), dtype=np.uint8).astype(dtype)
    kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)
    actual = mcv.filter2D(image, -1, kernel)
    expected = cv2.filter2D(image, -1, kernel)
    assert_close(actual, expected, atol=2e-4 if dtype == np.float32 else 0)


def test_filter2d_anchor_delta_and_constant_border():
    image = rng.normal(size=(13, 17))
    kernel = rng.normal(size=(4, 3))
    actual = mcv.filter2D(
        image, mcv.CV_64F, kernel, anchor=(0, 2), delta=1.25,
        borderType=mcv.BORDER_CONSTANT,
    )
    expected = cv2.filter2D(
        image, cv2.CV_64F, kernel, anchor=(0, 2), delta=1.25,
        borderType=cv2.BORDER_CONSTANT,
    )
    assert_close(actual, expected, atol=2e-14)


@pytest.mark.parametrize("shape", [(255, 257), (256, 257)])
def test_filter2d_simd_tail_across_parallel_threshold(shape):
    image = np.random.default_rng(sum(shape)).normal(size=shape)
    kernel = np.array([[1.0, -2.0, 1.0], [0.5, 0.0, -0.5], [2.0, 1.0, -1.0]])
    actual = mcv.filter2D(image, mcv.CV_64F, kernel)
    expected = cv2.filter2D(image, cv2.CV_64F, kernel)
    assert_close(actual, expected, atol=4e-15)


@pytest.mark.parametrize(
    "border", [mcv.BORDER_REPLICATE, mcv.BORDER_REFLECT, mcv.BORDER_REFLECT_101]
)
def test_box_filter_border_modes(border):
    image = rng.normal(size=(23, 29))
    actual = mcv.boxFilter(image, -1, (5, 3), borderType=border)
    expected = cv2.boxFilter(image, -1, (5, 3), borderType=border)
    assert_close(actual, expected, atol=5e-16)


def test_blur_color_uint8_is_exact():
    image = rng.integers(0, 256, (25, 27, 4), dtype=np.uint8)
    assert np.array_equal(mcv.blur(image, (5, 5)), cv2.blur(image, (5, 5)))


def test_sepfilter_matches_opencv():
    image = rng.normal(size=(19, 21))
    kx = np.array([-1.0, 0, 1])
    ky = np.array([1.0, 2, 1])
    actual = mcv.sepFilter2D(image, mcv.CV_64F, kx, ky)
    expected = cv2.sepFilter2D(image, cv2.CV_64F, kx, ky)
    assert_close(actual, expected, atol=2e-15)


def test_gaussian_kernel_matches_opencv():
    assert_close(
        mcv.getGaussianKernel(7, 1.4),
        cv2.getGaussianKernel(7, 1.4, cv2.CV_64F),
        atol=3e-17,
    )


@pytest.mark.parametrize("dtype", [np.uint8, np.float64])
def test_gaussian_blur_matches_opencv(dtype):
    image = rng.integers(0, 256, (31, 35), dtype=np.uint8).astype(dtype)
    actual = mcv.GaussianBlur(image, (7, 5), 1.3, sigmaY=0.9)
    expected = cv2.GaussianBlur(image, (7, 5), 1.3, sigmaY=0.9)
    if dtype == np.uint8:
        assert np.max(np.abs(actual.astype(int) - expected.astype(int))) <= 1
    else:
        assert_close(actual, expected, atol=5e-13)


def test_gaussian_separable_parallel_tail_matches_opencv():
    image = np.random.default_rng(89).integers(
        0, 256, (256, 257), dtype=np.uint8
    )
    actual = mcv.GaussianBlur(image, (7, 7), 1.4)
    expected = cv2.GaussianBlur(image, (7, 7), 1.4)
    assert np.max(np.abs(actual.astype(int) - expected.astype(int))) <= 1


@pytest.mark.parametrize("ksize", [3, 5])
def test_median_blur_is_exact(ksize):
    image = rng.integers(0, 256, (27, 29, 3), dtype=np.uint8)
    assert np.array_equal(mcv.medianBlur(image, ksize), cv2.medianBlur(image, ksize))


def test_median_parallel_threshold_is_exact():
    image = np.random.default_rng(81).integers(
        0, 256, (256, 257), dtype=np.uint8
    )
    assert np.array_equal(mcv.medianBlur(image, 3), cv2.medianBlur(image, 3))


@pytest.mark.parametrize("dx,dy", [(1, 0), (0, 1), (2, 0), (1, 1)])
def test_sobel_matches_opencv(dx, dy):
    image = rng.normal(size=(23, 25))
    actual = mcv.Sobel(image, mcv.CV_64F, dx, dy, scale=0.75, delta=0.2)
    expected = cv2.Sobel(image, cv2.CV_64F, dx, dy, scale=0.75, delta=0.2)
    assert_close(actual, expected, atol=4e-15)


def test_sobel_uint8_to_signed_16_is_exact():
    image = rng.integers(0, 256, (31, 35), dtype=np.uint8)
    actual = mcv.Sobel(image, mcv.CV_16S, 1, 0)
    expected = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("dx,dy", [(1, 0), (0, 1)])
def test_scharr_matches_opencv(dx, dy):
    image = rng.normal(size=(21, 25)).astype(np.float32)
    actual = mcv.Scharr(image, mcv.CV_32F, dx, dy, scale=0.5)
    expected = cv2.Scharr(image, cv2.CV_32F, dx, dy, scale=0.5)
    assert_close(actual, expected, atol=8e-6)


@pytest.mark.parametrize("ksize", [1, 3])
def test_laplacian_matches_opencv(ksize):
    image = rng.normal(size=(17, 19))
    actual = mcv.Laplacian(image, mcv.CV_64F, ksize=ksize)
    expected = cv2.Laplacian(image, cv2.CV_64F, ksize=ksize)
    assert_close(actual, expected, atol=2e-15)


@pytest.mark.parametrize("interpolation", [mcv.INTER_NEAREST, mcv.INTER_LINEAR])
def test_resize_float_matches_opencv(interpolation):
    image = rng.normal(size=(31, 37, 3))
    actual = mcv.resize(image, (23, 19), interpolation=interpolation)
    expected = cv2.resize(image, (23, 19), interpolation=interpolation)
    assert_close(actual, expected, atol=5e-16)


def test_resize_fx_fy_and_dst():
    image = rng.integers(0, 256, (12, 16), dtype=np.uint8)
    target = np.empty((6, 8), dtype=np.uint8)
    returned = mcv.resize(
        image, (0, 0), dst=target, fx=0.5, fy=0.5,
        interpolation=mcv.INTER_NEAREST,
    )
    assert returned is target
    assert np.array_equal(target, cv2.resize(image, (0, 0), fx=0.5, fy=0.5,
                                             interpolation=cv2.INTER_NEAREST))


def test_resize_u8_precomputed_parallel_tail_matches_opencv():
    image = np.random.default_rng(90).integers(
        0, 256, (256, 257, 3), dtype=np.uint8
    )
    actual = mcv.resize(image, (193, 191))
    expected = cv2.resize(image, (193, 191))
    assert np.max(np.abs(actual.astype(int) - expected.astype(int))) <= 1


def test_rotation_matrix_matches_opencv():
    actual = mcv.getRotationMatrix2D((13.5, 8.0), -27.0, 0.7)
    expected = cv2.getRotationMatrix2D((13.5, 8.0), -27.0, 0.7)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-15)


def test_warp_affine_linear_matches_opencv():
    image = np.random.default_rng(123).normal(size=(37, 41))
    matrix = cv2.getRotationMatrix2D((20, 18), 13, 0.85)
    actual = mcv.warpAffine(
        image, matrix, (43, 39), flags=mcv.INTER_LINEAR,
        borderMode=mcv.BORDER_REFLECT,
    )
    expected = cv2.warpAffine(
        image, matrix, (43, 39), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    assert actual.dtype == expected.dtype
    # Both use 1/32-pixel interpolation tables; a few inverse-map coordinates
    # can fall on opposite quantization boundaries.
    difference = np.abs(actual - expected)
    assert np.max(difference) < 0.1
    assert np.mean(difference > 1e-12) < 0.025
    assert np.mean(difference) < 0.001


def test_warp_affine_nearest_integer_translation():
    image = rng.integers(0, 256, (21, 23), dtype=np.uint8)
    matrix = np.array([[1, 0, 3], [0, 1, -2]], dtype=float)
    actual = mcv.warpAffine(image, matrix, (23, 21), flags=mcv.INTER_NEAREST)
    expected = cv2.warpAffine(image, matrix, (23, 21), flags=cv2.INTER_NEAREST)
    assert np.array_equal(actual, expected)


def test_warp_perspective_matches_opencv():
    image = rng.normal(size=(33, 35))
    source = np.float32([[0, 0], [34, 0], [34, 32], [0, 32]])
    target = np.float32([[2, 1], [31, 3], [34, 30], [1, 31]])
    matrix = cv2.getPerspectiveTransform(source, target)
    actual = mcv.warpPerspective(image, matrix, (35, 33))
    expected = cv2.warpPerspective(image, matrix, (35, 33))
    assert_close(actual, expected, atol=7e-16)


def test_transform_constructors_match_opencv():
    src3 = np.float32([[0, 0], [4, 0], [0, 5]])
    dst3 = np.float32([[1, 2], [6, 1], [2, 8]])
    np.testing.assert_allclose(
        mcv.getAffineTransform(src3, dst3), cv2.getAffineTransform(src3, dst3),
        atol=2e-15,
    )
    src4 = np.float32([[0, 0], [4, 0], [4, 5], [0, 5]])
    dst4 = np.float32([[1, 2], [6, 1], [5, 7], [2, 8]])
    np.testing.assert_allclose(
        mcv.getPerspectiveTransform(src4, dst4),
        cv2.getPerspectiveTransform(src4, dst4),
        atol=3e-15,
    )
    matrix = np.array([[0.8, -0.2, 4.0], [0.3, 1.1, -2.0]])
    np.testing.assert_allclose(
        mcv.invertAffineTransform(matrix),
        cv2.invertAffineTransform(matrix),
        rtol=0,
        atol=1e-15,
    )


@pytest.mark.parametrize("shape", [mcv.MORPH_RECT, mcv.MORPH_CROSS, mcv.MORPH_ELLIPSE])
@pytest.mark.parametrize("ksize", [(3, 3), (5, 7)])
def test_structuring_elements_match_opencv(shape, ksize):
    assert np.array_equal(
        mcv.getStructuringElement(shape, ksize),
        cv2.getStructuringElement(shape, ksize),
    )


@pytest.mark.parametrize("operation", ["erode", "dilate"])
def test_basic_morphology_matches_opencv(operation):
    image = rng.integers(0, 256, (39, 41, 3), dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3))
    actual = getattr(mcv, operation)(image, kernel, iterations=2)
    expected = getattr(cv2, operation)(image, kernel, iterations=2)
    assert np.array_equal(actual, expected)


def test_morphology_simd_tail_parallel_path_is_exact():
    image = np.random.default_rng(82).integers(
        0, 256, (256, 257), dtype=np.uint8
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    assert np.array_equal(mcv.erode(image, kernel), cv2.erode(image, kernel))


@pytest.mark.parametrize(
    "operation",
    [mcv.MORPH_OPEN, mcv.MORPH_CLOSE, mcv.MORPH_GRADIENT,
     mcv.MORPH_TOPHAT, mcv.MORPH_BLACKHAT],
)
def test_morphology_ex_matches_opencv(operation):
    image = rng.integers(0, 256, (31, 37), dtype=np.uint8)
    kernel = np.ones((3, 5), dtype=np.uint8)
    actual = mcv.morphologyEx(image, operation, kernel)
    expected = cv2.morphologyEx(image, operation, kernel)
    assert np.array_equal(actual, expected)


def test_harris_response_matches_opencv():
    image = rng.integers(0, 256, (47, 51), dtype=np.uint8)
    actual = mcv.cornerHarris(image, 3, 3, 0.04)
    expected = cv2.cornerHarris(image, 3, 3, 0.04)
    assert_close(actual, expected, atol=6e-8)


def test_min_eigen_response_matches_opencv():
    image = rng.normal(size=(41, 43)).astype(np.float32)
    actual = mcv.cornerMinEigenVal(image, 5)
    expected = cv2.cornerMinEigenVal(image, 5)
    assert_close(actual, expected, atol=4e-7)


def test_min_eigen_even_block_size_matches_opencv():
    image = rng.normal(size=(23, 27)).astype(np.float32)
    actual = mcv.cornerMinEigenVal(image, 2)
    expected = cv2.cornerMinEigenVal(image, 2)
    assert_close(actual, expected, atol=4e-7)


def test_good_features_matches_opencv_on_geometric_image():
    image = np.zeros((80, 90), dtype=np.uint8)
    image[10:35, 12:40] = 255
    image[47:70, 55:82] = 180
    actual = mcv.goodFeaturesToTrack(image, 20, 0.01, 4)
    expected = cv2.goodFeaturesToTrack(image, 20, 0.01, 4)
    assert np.array_equal(actual, expected)


def test_good_features_mask_and_harris():
    image = np.zeros((80, 90), dtype=np.uint8)
    image[10:35, 12:40] = 255
    image[47:70, 55:82] = 180
    mask = np.zeros_like(image)
    mask[:, :50] = 1
    actual = mcv.goodFeaturesToTrack(
        image, 20, 0.01, 3, mask=mask, useHarrisDetector=True
    )
    expected = cv2.goodFeaturesToTrack(
        image, 20, 0.01, 3, mask=mask, useHarrisDetector=True
    )
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("l2", [False, True])
def test_canny_matches_opencv_on_step_edges(l2):
    image = np.zeros((71, 79), dtype=np.uint8)
    image[12:55, 15:61] = 255
    actual = mcv.Canny(image, 50, 120, L2gradient=l2)
    expected = cv2.Canny(image, 50, 120, L2gradient=l2)
    assert np.array_equal(actual, expected)


def test_canny_parallel_gradient_path_is_exact():
    image = np.zeros((256, 257), dtype=np.uint8)
    image[31:211, 47:219] = 255
    assert np.array_equal(mcv.Canny(image, 50, 120), cv2.Canny(image, 50, 120))


def test_validation_rejects_unsupported_inputs():
    with pytest.raises(ValueError):
        mcv.medianBlur(np.zeros((3, 3), dtype=np.uint8), 4)
    with pytest.raises(NotImplementedError):
        mcv.resize(np.zeros((3, 3)), (2, 2), interpolation=3)
    with pytest.raises(ValueError):
        mcv.Canny(np.zeros((3, 3), dtype=np.float32), 1, 2)
    with pytest.raises(NotImplementedError):
        mcv.Sobel(np.zeros((3, 3)), mcv.CV_64F, -1, 2)
    with pytest.raises(NotImplementedError):
        mcv.Sobel(np.zeros((3, 3)), mcv.CV_64F, 2, 2)


@pytest.mark.parametrize(
    "image",
    [
        np.empty((0, 3), dtype=np.uint8),
        np.empty((3, 0), dtype=np.float64),
        np.empty((3, 3, 0), dtype=np.uint8),
    ],
)
def test_ffi_rejects_empty_buffers(image):
    with pytest.raises(ValueError):
        mcv.filter2D(image, -1, np.ones((3, 3)))


@pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float16])
def test_ffi_rejects_dtypes_that_would_narrow(dtype):
    with pytest.raises(TypeError):
        mcv.resize(np.ones((3, 3), dtype=dtype), (2, 2))


def test_ffi_rejects_invalid_sizes_borders_and_nonfinite_data():
    image = np.ones((3, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        mcv.resize(image, (-1, 2))
    with pytest.raises(ValueError):
        mcv.resize(image, (0, 2))
    with pytest.raises(NotImplementedError):
        mcv.filter2D(image, -1, np.ones((3, 3)), borderType=mcv.BORDER_TRANSPARENT)
    with pytest.raises(ValueError):
        mcv.filter2D(image, -1, np.array([[np.nan]]))
    with pytest.raises(ValueError):
        mcv.warpAffine(image, [[1, 0, np.inf], [0, 1, 0]], (3, 3))
    with pytest.raises(ValueError):
        mcv.erode(image, np.ones((3, 3)), iterations=-1)


def test_noncontiguous_input_is_copied_safely():
    image = np.arange(120, dtype=np.uint8).reshape(10, 12)[:, ::2]
    assert not image.flags.c_contiguous
    assert np.array_equal(
        mcv.resize(image, (4, 7), interpolation=mcv.INTER_NEAREST),
        cv2.resize(image, (4, 7), interpolation=cv2.INTER_NEAREST),
    )
