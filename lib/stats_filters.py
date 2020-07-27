import numpy as np
import scipy.signal
from skimage.filters.rank import majority
from lib.stats_local import kernel_filter, fast_sum, mode_array, feather_s2_array
from lib.stats_local_no_kernel import (
    truncate_array,
    threshold_array,
    cdef_from_z,
    select_highest,
)
from lib.stats_kernel import create_kernel

def sigma_to_db(arr):
    return 10 * np.log10(np.abs(arr))

def to_8bit(arr, min_target, max_target):
    return np.interp(arr, (min_target, max_target), (0, 255)).astype("uint8")


def scale_to_range_filter(in_raster, min_target, max_target):
    return np.interp(
        in_raster, (in_raster.min(), in_raster.max()), (min_target, max_target)
    )


def standardise_filter(in_raster, cdf_norm=False):
    m = np.nanmean(in_raster)
    s = np.nanstd(in_raster)
    if cdf_norm == False:
        return (in_raster - m) / s
    return cdef_from_z((in_raster - m) / s)


def robust_scaler_filter(in_raster):
    q1 = np.quantile(in_raster, 0.25)
    q3 = np.quantile(in_raster, 0.75)
    iqr = q3 - q1

    return (in_raster - q1) / iqr


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
    dtype=False,
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

    if dtype is False:
        return kernel_filter(in_raster, kernel, "mean").astype(in_raster.dtype)

    return kernel_filter(in_raster, kernel, "mean").astype(dtype)


def fast_sum_filter(
    in_raster,
    width=3,
    circular=True,
    holed=False,
    weighted_edges=True,
    weighted_distance=False,
    distance_calc="gaussian",
    sigma=2,
    dtype=False,
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

    # TODO: HANDLE NODATA
    # return fast_sum(in_raster, kernel).astype(in_raster.dtype)
    return scipy.signal.fftconvolve(in_raster, kernel, mode="same").astype(
        in_raster.dtype
    )


def cdef_filter(in_raster):
    return cdef_from_z(in_raster)


def cdef_difference_filter(
    in_raster_master,
    in_raster_slave,
    width=3,
    circular=True,
    holed=False,
    weighted_edges=True,
    weighted_distance=False,
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
            holed=holed,
            normalise=True,
            weighted_edges=weighted_edges,
            weighted_distance=weighted_distance,
            distance_calc=distance_calc,
            sigma=sigma,
        )
    )

    master_mean = kernel_filter(in_raster_master, kernel, "mean")
    slave_mean = kernel_filter(in_raster_slave, kernel, "mean")

    master_median = kernel_filter(in_raster_master, kernel, "median")
    slave_median = kernel_filter(in_raster_slave, kernel, "median")

    abs_median_diff = np.abs(master_median - slave_median)
    master_mean_difference = np.abs(master_mean - np.abs(master_median - slave_median))
    slave_mean_difference = np.abs(slave_mean - np.abs(master_median - slave_median))

    master_mad_std = kernel_filter(in_raster_master, kernel, "mad_std")
    slave_mad_std = kernel_filter(in_raster_slave, kernel, "mad_std")

    with np.errstate(divide="ignore", invalid="ignore"):
        zscores_master = (master_mean - master_mean_difference) / master_mad_std
        zscores_master[master_mad_std == 0] = 0

        zscores_slave = (slave_mean - slave_mean_difference) / slave_mad_std
        zscores_slave[slave_mad_std == 0] = 0

    return np.min([zscores_master, zscores_slave], axis=0)

    # master_cdef = cdef_from_z(zscores_master)
    # slave_cdef = cdef_from_z(zscores_slave)

    # return (master_cdef + slave_cdef) / 2

    # if dtype is False:
    #     return cdef_from_z(zscores).astype(in_raster_master.dtype)
    # else:
    #     return cdef_from_z(zscores).astype(dtype)


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
    dtype="float32",
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
    dtype="float32",
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
    dtype="float32",
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
    dtype="float32",
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
    dtype="float32",
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
    dtype="float32",
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
    dtype="float32",
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
    dtype="float32",
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
    dtype="float32",
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
    dtype="float32",
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
    dtype="float32",
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
    dtype="float32",
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
    dtype="float32",
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
    dtype="float32",
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
    return (
        (in_raster - mean_filter(in_raster, _kernel=kernel))
        / standard_deviation_filter(in_raster, _kernel=kernel)
    ).astype(dtype)


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
    dtype="float32",
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
    dtype="float32",
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


def threshold_filter(in_raster, min_value=False, max_value=False, invert=False):
    return threshold_array(
        in_raster, min_value=min_value, max_value=max_value, invert=invert,
    ).astype(in_raster.dtype)


def truncate_filter(in_raster, min_value=False, max_value=False, dim3=False):
    return truncate_array(in_raster, min_value=min_value, max_value=max_value,).astype(
        in_raster.dtype
    )


def highest_filter(in_rasters, weights):
    return select_highest(in_rasters, weights).astype("uint8")


def mode_filter(in_raster, width=5, iterations=1, circular=True):
    kernel = create_kernel(
        width,
        circular=circular,
        holed=False,
        normalise=False,
        weighted_edges=False,
        weighted_distance=False,
    )

    if iterations == 1:
        return mode_array(in_raster, kernel)
        # return majority(in_raster, kernel)
    else:
        result = mode_array(in_raster, kernel)
        # result = majority(in_raster, kernel)
        for x in range(iterations - 1):
            # result = majority(result, kernel)
            result = mode_array(result, kernel)
        return result


def feather_s2_filter(tracking_image, value_to_count, width=21, circular=True):
    kernel = create_kernel(
        width,
        circular=circular,
        holed=False,
        normalise=False,
        weighted_edges=False,
        weighted_distance=False,
    )
    return feather_s2_array(tracking_image, value_to_count, kernel)
