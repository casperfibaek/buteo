import numpy as np
from scipy.stats import norm
from lib.stats_local import kernel_filter
# from lib.stats_local_no_kernel import truncate_array, threshold_array
from lib.stats_kernel import create_kernel


def standardise_filter(in_raster, cdf_norm=False):
    m = np.nanmean(in_raster)
    s = np.nanstd(in_raster)
    if cdf_norm == False:
        return (in_raster - m) / s
    return ((norm.cdf((in_raster - m) / s) - 0.5) / 0.5).astype(in_raster.dtype)


def normalise_filter(in_raster):
    mi = np.nanmin(in_raster)
    ma = np.nanmax(in_raster)
    return ((in_raster - mi) / (ma - mi)).astype(in_raster.dtype)


def invert_filter(in_raster):
    mi = np.nanmin(in_raster)
    ma = np.nanmax(in_raster)
    return ((ma - in_raster) + mi).astype(in_raster.dtype)


def sum_filter(
    in_raster,
    width=3,
    circular=True,
    holed=False,
    weighted_edges=True,
    weighted_distance=False,
    distance_calc="gaussian",
    sigma=2,
    _kernel=False,
):
    kernel = (
        _kernel
        if _kernel is not False
        else create_kernel(
            width,
            circular=circular,
            holed=holed,
            normalise=False,
            weighted_edges=weighted_edges,
            weighted_distance=weighted_distance,
            distance_calc=distance_calc,
            sigma=sigma,
        )
    )
    return kernel_filter(in_raster, kernel, "mean").astype(in_raster.dtype)


def mean_filter(
    in_raster,
    width=3,
    iterations=1,
    circular=True,
    holed=False,
    weighted_edges=True,
    weighted_distance=True,
    distance_calc="gaussian",
    sigma=2,
    dtype='float32',
    _kernel=False,
):
    kernel = (
        _kernel
        if _kernel is not False
        else create_kernel(
            width,
            circular=circular,
            normalise=True,
            holed=holed,
            weighted_edges=weighted_edges,
            weighted_distance=weighted_distance,
            distance_calc=distance_calc,
            sigma=sigma,
        )
    )
    if iterations == 1:
        return kernel_filter(in_raster, kernel, "mean", dtype)
    else:
        result = kernel_filter(in_raster, kernel, "mean", dtype)
        for _ in range(iterations - 1):
            result = kernel_filter(in_raster, kernel, "mean", dtype)
        return result


def median_filter(
    in_raster,
    width=3,
    iterations=1,
    circular=True,
    holed=False,
    weighted_edges=True,
    weighted_distance=True,
    distance_calc="gaussian",
    sigma=2,
    dtype='float32',
    _kernel=False,
):
    kernel = (
        _kernel
        if _kernel is not False
        else create_kernel(
            width,
            circular=circular,
            holed=holed,
            weighted_edges=weighted_edges,
            weighted_distance=weighted_distance,
            distance_calc=distance_calc,
            sigma=sigma,
        )
    )
    
    if iterations == 1:
        return kernel_filter(in_raster, kernel, "median", dtype)
    else:
        result = kernel_filter(in_raster, kernel, "median", dtype)
        for _ in range(iterations - 1):
            result = kernel_filter(result, kernel, "median", dtype)
        return result


def variance_filter(
    in_raster,
    width=3,
    circular=True,
    holed=False,
    weighted_edges=True,
    weighted_distance=True,
    distance_calc="gaussian",
    sigma=2,
    dtype='float32',
    _kernel=False,
):
    kernel = (
        _kernel
        if _kernel is not False
        else create_kernel(
            width,
            circular=circular,
            holed=holed,
            weighted_edges=weighted_edges,
            weighted_distance=weighted_distance,
            distance_calc=distance_calc,
            sigma=sigma,
        )
    )
    return kernel_filter(in_raster, kernel, "variance", dtype)


def standard_deviation_filter(
    in_raster,
    width=3,
    circular=True,
    holed=False,
    weighted_edges=True,
    weighted_distance=True,
    distance_calc="gaussian",
    sigma=2,
    dtype='float32',
    _kernel=False,
):
    kernel = (
        _kernel
        if _kernel is not False
        else create_kernel(
            width,
            circular=circular,
            holed=holed,
            weighted_edges=weighted_edges,
            weighted_distance=weighted_distance,
            distance_calc=distance_calc,
            sigma=sigma,
        )
    )
    return kernel_filter(in_raster, kernel, "standard_deviation", dtype)


def q1_filter(
    in_raster,
    width=3,
    circular=True,
    holed=False,
    weighted_edges=True,
    weighted_distance=True,
    distance_calc="gaussian",
    sigma=2,
    dtype='float32',
    _kernel=False,
):
    kernel = (
        _kernel
        if _kernel is not False
        else create_kernel(
            width,
            circular=circular,
            holed=holed,
            weighted_edges=weighted_edges,
            weighted_distance=weighted_distance,
            distance_calc=distance_calc,
            sigma=sigma,
        )
    )
    return kernel_filter(in_raster, kernel, "q1", dtype)


def q3_filter(
    in_raster,
    width=3,
    circular=True,
    holed=False,
    weighted_edges=True,
    weighted_distance=True,
    distance_calc="gaussian",
    sigma=2,
    dtype='float32',
    _kernel=False,
):
    kernel = (
        _kernel
        if _kernel is not False
        else create_kernel(
            width,
            circular=circular,
            holed=holed,
            weighted_edges=weighted_edges,
            weighted_distance=weighted_distance,
            distance_calc=distance_calc,
            sigma=sigma,
        )
    )
    return kernel_filter(in_raster, kernel, "q3", dtype)


def iqr_filter(
    in_raster,
    width=3,
    circular=True,
    holed=False,
    weighted_edges=True,
    weighted_distance=True,
    distance_calc="gaussian",
    sigma=2,
    dtype='float32',
    _kernel=False,
):
    kernel = (
        _kernel
        if _kernel is not False
        else create_kernel(
            width,
            circular=circular,
            holed=holed,
            weighted_edges=weighted_edges,
            weighted_distance=weighted_distance,
            distance_calc=distance_calc,
            sigma=sigma,
        )
    )
    return kernel_filter(in_raster, kernel, "iqr", dtype)


def mad_filter(
    in_raster,
    width=3,
    circular=True,
    holed=False,
    weighted_edges=True,
    weighted_distance=True,
    distance_calc="gaussian",
    sigma=2,
    dtype='float32',
    _kernel=False,
):
    kernel = (
        _kernel
        if _kernel is not False
        else create_kernel(
            width,
            circular=circular,
            holed=holed,
            weighted_edges=weighted_edges,
            weighted_distance=weighted_distance,
            distance_calc=distance_calc,
            sigma=sigma,
        )
    )
    return kernel_filter(in_raster, kernel, "mad", dtype)


def mad_std_filter(
    in_raster,
    width=3,
    circular=True,
    holed=False,
    weighted_edges=True,
    weighted_distance=True,
    distance_calc="gaussian",
    sigma=2,
    dtype='float32',
    _kernel=False,
):
    kernel = (
        _kernel
        if _kernel is not False
        else create_kernel(
            width,
            circular=circular,
            holed=holed,
            weighted_edges=weighted_edges,
            weighted_distance=weighted_distance,
            distance_calc=distance_calc,
            sigma=sigma,
        )
    )
    return kernel_filter(in_raster, kernel, "mad_std", dtype)


def skew_fp_filter(
    in_raster,
    width=3,
    circular=True,
    holed=False,
    weighted_edges=True,
    weighted_distance=True,
    distance_calc="gaussian",
    sigma=2,
    dtype='float32',
    _kernel=False,
):
    kernel = (
        _kernel
        if _kernel is not False
        else create_kernel(
            width,
            circular=circular,
            holed=holed,
            weighted_edges=weighted_edges,
            weighted_distance=weighted_distance,
            distance_calc=distance_calc,
            sigma=sigma,
        )
    )
    return kernel_filter(in_raster, kernel, "skew_fp", dtype)


def skew_p2_filter(
    in_raster,
    width=3,
    circular=True,
    holed=False,
    weighted_edges=True,
    weighted_distance=True,
    distance_calc="gaussian",
    sigma=2,
    dtype='float32',
    _kernel=False,
):
    kernel = (
        _kernel
        if _kernel is not False
        else create_kernel(
            width,
            circular=circular,
            holed=holed,
            weighted_edges=weighted_edges,
            weighted_distance=weighted_distance,
            distance_calc=distance_calc,
            sigma=sigma,
        )
    )
    return kernel_filter(in_raster, kernel, "skew_p2", dtype)


def skew_g_filter(
    in_raster,
    width=3,
    circular=True,
    holed=False,
    weighted_edges=True,
    weighted_distance=True,
    distance_calc="gaussian",
    sigma=2,
    dtype='float32',
    _kernel=False,
):
    kernel = (
        _kernel
        if _kernel is not False
        else create_kernel(
            width,
            circular=circular,
            holed=holed,
            weighted_edges=weighted_edges,
            weighted_distance=weighted_distance,
            distance_calc=distance_calc,
            sigma=sigma,
        )
    )
    return kernel_filter(in_raster, kernel, "skew_g", dtype)


def kurtosis_filter(
    in_raster,
    width=3,
    circular=True,
    holed=False,
    weighted_edges=True,
    weighted_distance=True,
    distance_calc="gaussian",
    sigma=2,
    dtype='float32',
    _kernel=False,
):
    kernel = (
        _kernel
        if _kernel is not False
        else create_kernel(
            width,
            circular=circular,
            holed=holed,
            weighted_edges=weighted_edges,
            weighted_distance=weighted_distance,
            distance_calc=distance_calc,
            sigma=sigma,
        )
    )
    return kernel_filter(in_raster, kernel, "kurtosis", dtype)


def z_filter(
    in_raster,
    width=3,
    circular=True,
    holed=False,
    weighted_edges=True,
    weighted_distance=True,
    distance_calc="gaussian",
    sigma=2,
    dtype='float32',
    _kernel=False,
):
    kernel = (
        _kernel
        if _kernel is not False
        else create_kernel(
            width,
            circular=circular,
            holed=holed,
            weighted_edges=weighted_edges,
            weighted_distance=weighted_distance,
            distance_calc=distance_calc,
            sigma=sigma,
        )
    )
    return ((
        in_raster - mean_filter(in_raster, _kernel=kernel)
    ) / standard_deviation_filter(in_raster, _kernel=kernel)).astype(dtype)


def median_deviation_filter(
    in_raster,
    width=3,
    absolute_value=False,
    circular=True,
    weighted_edges=True,
    holed=True,
    weighted_distance=True,
    distance_calc="gaussian",
    sigma=2,
    dtype='float32',
    _kernel=False,
):
    kernel = (
        _kernel
        if _kernel is not False
        else create_kernel(
            width,
            circular=circular,
            holed=holed,
            weighted_edges=weighted_edges,
            weighted_distance=weighted_distance,
            distance_calc=distance_calc,
            sigma=sigma,
        )
    )
    result = in_raster - median_filter(in_raster, _kernel=kernel)
    if absolute_value is True:
        return np.abs(result).astype(dtype)
    return result.astype(dtype)


def mean_deviation_filter(
    in_raster,
    width=3,
    absolute_value=False,
    circular=True,
    weighted_edges=True,
    holed=True,
    weighted_distance=True,
    distance_calc="gaussian",
    sigma=2,
    dtype='float32',
    _kernel=False,
):
    kernel = (
        _kernel
        if _kernel is not False
        else create_kernel(
            width,
            circular=circular,
            holed=holed,
            weighted_edges=weighted_edges,
            weighted_distance=weighted_distance,
            distance_calc=distance_calc,
            sigma=sigma,
        )
    )
    result = in_raster - mean_filter(in_raster, _kernel=kernel)
    if absolute_value is True:
        return np.abs(result).astype(dtype)
    return result.astype(dtype)


def snr_filter(
    in_raster,
    width=3,
    circular=True,
    weighted_edges=True,
    weighted_distance=True,
    distance_calc="gaussian",
    sigma=2,
    dtype="float32",
    _kernel=False,
):
    kernel = (
        _kernel
        if _kernel is not False
        else create_kernel(
            width,
            circular=circular,
            holed=True,
            weighted_edges=weighted_edges,
            weighted_distance=weighted_distance,
            distance_calc=distance_calc,
            sigma=sigma,
        )
    )
    return np.power(mean_filter(in_raster, _kernel=kernel), 2) / variance_filter(
        in_raster, _kernel=kernel
    ).astype(dtype)


def normalise_to_range_filter(in_raster, low=0, high=255):
    return ((in_raster - low) / (high - low)).astype(in_raster.dtype)


def dilation_filter(
    in_raster,
    width=3,
    iterations=1,
    circular=True,
    weighted_distance=True,
    sigma=3.5,
    _kernel=False,
):
    kernel = (
        _kernel
        if _kernel is not False
        else create_kernel(
            width,
            circular=circular,
            weighted_edges=True,
            weighted_distance=weighted_distance,
            sigma=sigma,
            normalise=False,
        )
    )
    if iterations == 1:
        return kernel_filter(in_raster, kernel, "dilate").astype(in_raster.dtype)
    else:
        result = kernel_filter(in_raster, kernel, "dilate").astype(in_raster.dtype)
        for _ in range(iterations - 1):
            result = kernel_filter(result, kernel, "dilate").astype(in_raster.dtype)
        return result


def erosion_filter(
    in_raster,
    width=3,
    iterations=1,
    circular=True,
    weighted_distance=True,
    sigma=3.5,
    _kernel=False,
):
    kernel = (
        _kernel
        if _kernel is not False
        else create_kernel(
            width,
            circular=circular,
            weighted_edges=True,
            weighted_distance=weighted_distance,
            sigma=sigma,
            normalise=False,
        )
    )
    if iterations == 1:
        return kernel_filter(in_raster, kernel, "erode").astype(in_raster.dtype)
    else:
        result = kernel_filter(in_raster, kernel, "erode").astype(in_raster.dtype)
        for _ in range(iterations - 1):
            result = kernel_filter(result, kernel, "erode").astype(in_raster.dtype)
        return result


def open_filter(
    in_raster,
    width=3,
    circular=True,
    weighted_distance=True,
    sigma=3.5,
    dim3=False,
    _kernel=False,
):
    kernel = (
        _kernel
        if _kernel is not False
        else create_kernel(
            width,
            circular=circular,
            weighted_edges=True,
            weighted_distance=weighted_distance,
            sigma=sigma,
            normalise=False,
        )
    )
    return kernel_filter(
        kernel_filter(in_raster, kernel, "erode"), kernel, "dilate",
    ).astype(in_raster.dtype)


def close_filter(
    in_raster, width=3, circular=True, weighted_distance=True, sigma=3.5, _kernel=False,
):
    kernel = (
        _kernel
        if _kernel is not False
        else create_kernel(
            width,
            circular=circular,
            weighted_edges=True,
            weighted_distance=weighted_distance,
            sigma=sigma,
            normalise=False,
        )
    )
    return kernel_filter(
        kernel_filter(in_raster, kernel, "dilate"), kernel, "erode",
    ).astype(in_raster.dtype)


# def threshold_filter(in_raster, min_value=False, max_value=False, invert=False):
#     return threshold_array(
#         in_raster, min_value=min_value, max_value=max_value, invert=invert
#     ).astype(in_raster.dtype)


# def truncate_filter(in_raster, min_value=False, max_value=False, dim3=False):
#     return truncate_array(in_raster, min_value=min_value, max_value=max_value).astype(
#         in_raster.dtype
#     ).astype(in_raster.dtype)

