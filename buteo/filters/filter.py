import numpy as np
import sys; sys.path.append('../../')
from buteo.filters.convolutions import kernel_filter
from buteo.filters.kernel_generator import create_kernel


# filters = [
#     "mean",
#     "median",
#     "variance",
#     "standard_deviation",
#     "quintile_xxx",
#     "median_absolute_deviation",
#     "skew",
#     "kurtosis",
#     "mode",
# ]

# morphology = [
#     "dilation",
#     "erosion",
#     "open",
#     "close",
#     "mode",
# ]

def conv_filter(in_raster, kernel=None, filter="sum", iterations=1):
    _kernel = kernel if kernel != None else create_kernel()
    return kernel_filter(in_raster, _kernel, filter, iterations)


def sigma_to_db(arr):
    return 10 * np.log10(np.abs(arr))


def db_to_sigma(arr):
    return 10 ** (np.divide(arr, 10))


def to_8bit(arr, min_target, max_target):
    return np.interp(arr, (min_target, max_target), (0, 255)).astype("uint8")


def scale_to_range_filter(in_raster, min_target, max_target):
    return np.interp(
        in_raster, (in_raster.nanmin(), in_raster.nanmax()), (min_target, max_target)
    )


def standardise_filter(in_raster):
    m = np.nanmean(in_raster)
    s = np.nanstd(in_raster)

    return (in_raster - m) / s


def robust_scaler_filter(in_raster):
    q1 = np.quantile(in_raster, 0.25)
    q3 = np.quantile(in_raster, 0.75)
    iqr = q3 - q1

    return (in_raster - q1) / iqr


def normalise_filter(in_raster):
    mi = np.nanmin(in_raster)
    ma = np.nanmax(in_raster)
    return ((in_raster - mi) / (ma - mi)).astype(in_raster.dtype)


def normalise_to_range_filter(in_raster, low=0, high=255):
    return ((in_raster - low) / (high - low)).astype(in_raster.dtype)


def threshold_filter(in_raster, min_value=False, max_value=False, inverted=False):
    if min_value == False and max_value == False:
        return in_raster

    min_v = min_value if min_value != False else -np.Infinity
    max_v = max_value if max_value != False else np.Infinity

    if not inverted:
        return in_raster[np.logical_and(in_raster >= min_v, in_raster <= max_v)]
    
    return in_raster[np.logical_or(in_raster < min_value, in_raster > max_value)]


def truncate_filter(in_raster, min_value=False, max_value=False, inverted=False, replace_value=0):
    if min_value == False and max_value == False:
        return in_raster

    min_v = min_value if min_value != False else -np.Infinity
    max_v = max_value if max_value != False else np.Infinity

    thresholded = np.copy(in_raster)
    if not inverted:
        thresholded[thresholded <= min_v] = min_v
        thresholded[thresholded >= max_v] = max_v
    else:
        thresholded[np.logical_and(thresholded >= min_v, thresholded <= max_v)] = replace_value
    
    return thresholded


def invert_filter(in_raster):
    mi = np.nanmin(in_raster)
    ma = np.nanmax(in_raster)
    return ((ma - in_raster) + mi).astype(in_raster.dtype)
