import numpy as np
from scipy.stats import norm
from lib.stats_local import local_filter, _truncate_filter, _threshold_filter
from lib.stats_kernel import create_kernel


def standardise_filter(in_raster, cdf_norm=False):
    m = np.nanmean(in_raster)
    s = np.nanstd(in_raster)
    if cdf_norm == False:
        return (in_raster - m) / s
    return (norm.cdf((in_raster - m) / s) - 0.5) / 0.5


def normalise_filter(in_raster):
    mi = np.nanmin(in_raster)
    ma = np.nanmax(in_raster)
    return (in_raster - mi) / (ma - mi)


def invert_filter(in_raster):
    mi = np.nanmin(in_raster)
    ma = np.nanmax(in_raster)
    return (ma - in_raster) + mi

 
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
    return local_filter(in_raster, kernel, "mean")


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
        return local_filter(in_raster, kernel, "mean")
    else:
        result = local_filter(in_raster, kernel, "mean")
        for _ in range(iterations - 1):
            result = local_filter(in_raster, kernel, "mean")
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
        return local_filter(in_raster, kernel, "median")
    else:
        result = local_filter(in_raster, kernel, "median")
        for _ in range(iterations - 1):
            result = local_filter(result, kernel, "median")
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
    return local_filter(in_raster, kernel, "variance")


def standard_deviation_filter(
    in_raster,
    width=3,
    circular=True,
    holed=False,
    weighted_edges=True,
    weighted_distance=True,
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
            weighted_edges=weighted_edges,
            weighted_distance=weighted_distance,
            distance_calc=distance_calc,
            sigma=sigma,
        )
    )
    return local_filter(in_raster, kernel, "standard_deviation")


def q1_filter(
    in_raster,
    width=3,
    circular=True,
    holed=False,
    weighted_edges=True,
    weighted_distance=True,
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
            weighted_edges=weighted_edges,
            weighted_distance=weighted_distance,
            distance_calc=distance_calc,
            sigma=sigma,
        )
    )
    return local_filter(in_raster, kernel, "q1")


def q3_filter(
    in_raster,
    width=3,
    circular=True,
    holed=False,
    weighted_edges=True,
    weighted_distance=True,
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
            weighted_edges=weighted_edges,
            weighted_distance=weighted_distance,
            distance_calc=distance_calc,
            sigma=sigma,
        )
    )
    return local_filter(in_raster, kernel, "q3")


def iqr_filter(
    in_raster,
    width=3,
    circular=True,
    holed=False,
    weighted_edges=True,
    weighted_distance=True,
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
            weighted_edges=weighted_edges,
            weighted_distance=weighted_distance,
            distance_calc=distance_calc,
            sigma=sigma,
        )
    )
    return local_filter(in_raster, kernel, "iqr")


def mad_filter(
    in_raster,
    width=3,
    circular=True,
    holed=False,
    weighted_edges=True,
    weighted_distance=True,
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
            weighted_edges=weighted_edges,
            weighted_distance=weighted_distance,
            distance_calc=distance_calc,
            sigma=sigma,
        )
    )
    return local_filter(in_raster, kernel, "mad")


def mad_std_filter(
    in_raster,
    width=3,
    circular=True,
    holed=False,
    weighted_edges=True,
    weighted_distance=True,
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
            weighted_edges=weighted_edges,
            weighted_distance=weighted_distance,
            distance_calc=distance_calc,
            sigma=sigma,
        )
    )
    return local_filter(in_raster, kernel, "mad_std")


def skew_fp_filter(
    in_raster,
    width=3,
    circular=True,
    holed=False,
    weighted_edges=True,
    weighted_distance=True,
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
            weighted_edges=weighted_edges,
            weighted_distance=weighted_distance,
            distance_calc=distance_calc,
            sigma=sigma,
        )
    )
    return local_filter(in_raster, kernel, "skew_fp")


def skew_p2_filter(
    in_raster,
    width=3,
    circular=True,
    holed=False,
    weighted_edges=True,
    weighted_distance=True,
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
            weighted_edges=weighted_edges,
            weighted_distance=weighted_distance,
            distance_calc=distance_calc,
            sigma=sigma,
        )
    )
    return local_filter(in_raster, kernel, "skew_p2")


def skew_g_filter(
    in_raster,
    width=3,
    circular=True,
    holed=False,
    weighted_edges=True,
    weighted_distance=True,
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
            weighted_edges=weighted_edges,
            weighted_distance=weighted_distance,
            distance_calc=distance_calc,
            sigma=sigma,
        )
    )
    return local_filter(in_raster, kernel, "skew_g")


def kurtosis_filter(
    in_raster,
    width=3,
    circular=True,
    holed=False,
    weighted_edges=True,
    weighted_distance=True,
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
            weighted_edges=weighted_edges,
            weighted_distance=weighted_distance,
            distance_calc=distance_calc,
            sigma=sigma,
        )
    )
    return local_filter(in_raster, kernel, "kurtosis")


def z_filter(
    in_raster,
    width=3,
    circular=True,
    holed=False,
    weighted_edges=True,
    weighted_distance=True,
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
            weighted_edges=weighted_edges,
            weighted_distance=weighted_distance,
            distance_calc=distance_calc,
            sigma=sigma,
        )
    )
    return (
        in_raster - mean_filter(in_raster, _kernel=kernel)
    ) / standard_deviation_filter(in_raster, _kernel=kernel)


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
        return np.abs(result)
    return result


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
        return np.abs(result)
    return result


def snr_filter(
    in_raster,
    width=3,
    circular=True,
    weighted_edges=True,
    weighted_distance=True,
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
            holed=True,
            weighted_edges=weighted_edges,
            weighted_distance=weighted_distance,
            distance_calc=distance_calc,
            sigma=sigma,
        )
    )
    return np.power(mean_filter(in_raster, _kernel=kernel), 2) / variance_filter(
        in_raster, _kernel=kernel
    )


def normalise_to_range_filter(in_raster, low=0, high=255):
    return (in_raster - low) / (high - low)


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
        return local_filter(in_raster, kernel, "dilate").astype(in_raster.dtype)
    else:
        result = local_filter(in_raster, kernel, "dilate").astype(in_raster.dtype)
        for _ in range(iterations - 1):
            result = local_filter(result, kernel, "dilate").astype(in_raster.dtype)
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
        return local_filter(in_raster, kernel, "erode").astype(in_raster.dtype)
    else:
        result = local_filter(in_raster, kernel, "erode").astype(in_raster.dtype)
        for _ in range(iterations - 1):
            result = local_filter(result, kernel, "erode").astype(in_raster.dtype)
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
    return local_filter(
        local_filter(in_raster, kernel, "erode"), kernel, "dilate"
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
    return local_filter(
        local_filter(in_raster, kernel, "dilate"), kernel, "erode"
    ).astype(in_raster.dtype)


def threshold_filter(in_raster, min_value=False, max_value=False, invert=False):
    return _threshold_filter(
        in_raster, min_value=min_value, max_value=max_value, invert=invert
    )


def truncate_filter(in_raster, min_value=False, max_value=False, dim3=False):
    return _truncate_filter(in_raster, min_value=min_value, max_value=max_value).astype(
        in_raster.dtype
    )

