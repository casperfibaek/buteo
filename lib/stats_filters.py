import cv2
import numpy as np
import numexpr as ne
from scipy.stats import norm
from lib.stats_local import filter_2d, filter_3d
from lib.stats_kernel import create_kernel

def standardise_filter(in_raster, scaled_0_to_1=False):
    m = np.nanmean(in_raster)
    s = np.nanstd(in_raster)
    if scaled_0_to_1 == False:
        return (in_raster - m) / s
    return (norm.cdf((in_raster - m) / s) - 0.5) / 0.5

def normalise_filter(in_raster):
    mi = np.nanmin(in_raster)
    ma = np.nanmax(in_raster)
    return (in_raster - mi) / (ma - mi)

def invert_filter(in_raster):
    mi = np.nanmin(in_raster)
    ma = np.nanmax(in_raster)
    return np.ma.add(np.ma.subtract(ma, in_raster), mi)

def sum_filter(in_raster, width=3, circular=True, holed=False, weighted_edges=True, weighted_distance=False, distance_calc='gaussian', sigma=2, dim3=False, _kernel=False):
    kernel = _kernel if _kernel is not False else create_kernel(width, circular=circular, holed=holed, normalise=False, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    if dim3 == True:
        return filter_3d(in_raster, kernel, 'mean')
    return filter_2d(in_raster, kernel, 'mean')

def mean_filter(in_raster, width=3, iterations=1, circular=True, holed=False, weighted_edges=True, weighted_distance=True, distance_calc='gaussian', sigma=2, dim3=False, _kernel=False):
    kernel = _kernel if _kernel is not False else create_kernel(width, circular=circular, holed=holed, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    if iterations == 1:
        if dim3 == True:
            return filter_3d(in_raster, kernel, 'mean')
        return filter_2d(in_raster, kernel, 'mean')
    else:
        if dim3 == True:
            result = filter_3d(in_raster, kernel, 'mean')

            for _ in range(iterations - 1):
                result =  filter_3d(result, kernel, 'mean')
        else:
            result = filter_2d(in_raster, kernel, 'mean')

            for _ in range(iterations - 1):
                result = filter_2d(result, kernel, 'mean')
        return result

def median_filter(in_raster, width=3, iterations=1, circular=True, holed=False, weighted_edges=True, weighted_distance=True, distance_calc='gaussian', sigma=2, dim3=False, _kernel=False):
    kernel = _kernel if _kernel is not False else create_kernel(width, circular=circular, holed=holed, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    if iterations == 1:
        if dim3 == True:
            return filter_3d(in_raster, kernel, 'median')
        return filter_2d(in_raster, kernel, 'median')
    else:
        if dim3 == True:
            result = filter_3d(in_raster, kernel, 'median')

            for _ in range(iterations - 1):
                result =  filter_3d(result, kernel, 'median')
        else:
            result = filter_2d(in_raster, kernel, 'median')

            for _ in range(iterations - 1):
                result = filter_2d(result, kernel, 'median')
        return result

def variance_filter(in_raster, width=3, circular=True, holed=False, weighted_edges=True, weighted_distance=True, distance_calc='gaussian', sigma=2, dim3=False, _kernel=False):
    kernel = _kernel if _kernel is not False else create_kernel(width, circular=circular, holed=holed, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    if dim3 == True:
        return filter_3d(in_raster, kernel, 'variance')
    return filter_2d(in_raster, kernel, 'variance')

def standard_deviation_filter(in_raster, width=3, circular=True, holed=False, weighted_edges=True, weighted_distance=True, distance_calc='gaussian', sigma=2, dim3=False, _kernel=False):
    kernel = _kernel if _kernel is not False else create_kernel(width, circular=circular, holed=holed, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    if dim3 == True:
        return filter_3d(in_raster, kernel, 'standard_deviation')
    return filter_2d(in_raster, kernel, 'standard_deviation')

def q1_filter(in_raster, width=3, circular=True, holed=False, weighted_edges=True, weighted_distance=True, distance_calc='gaussian', sigma=2, dim3=False, _kernel=False):
    kernel = _kernel if _kernel is not False else create_kernel(width, circular=circular, holed=holed, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    if dim3 == True:
        return filter_3d(in_raster, kernel, 'q1')
    return filter_2d(in_raster, kernel, 'q1')

def q3_filter(in_raster, width=3, circular=True, holed=False, weighted_edges=True, weighted_distance=True, distance_calc='gaussian', sigma=2, dim3=False, _kernel=False):
    kernel = _kernel if _kernel is not False else create_kernel(width, circular=circular, holed=holed, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    if dim3 == True:
        return filter_3d(in_raster, kernel, 'q3')
    return filter_2d(in_raster, kernel, 'q3')

def iqr_filter(in_raster, width=3, circular=True, holed=False, weighted_edges=True, weighted_distance=True, distance_calc='gaussian', sigma=2, dim3=False, _kernel=False):
    kernel = _kernel if _kernel is not False else create_kernel(width, circular=circular, holed=holed, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    if dim3 == True:
        return filter_3d(in_raster, kernel, 'iqr')
    return filter_2d(in_raster, kernel, 'iqr')

def mad_filter(in_raster, width=3, circular=True, holed=False, weighted_edges=True, weighted_distance=True, distance_calc='gaussian', sigma=2, dim3=False, _kernel=False):
    kernel = _kernel if _kernel is not False else create_kernel(width, circular=circular, holed=holed, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    if dim3 == True:
        return filter_3d(in_raster, kernel, 'mad')
    return filter_2d(in_raster, kernel, 'mad')

def mad_std_filter(in_raster, width=3, circular=True, holed=False, weighted_edges=True, weighted_distance=True, distance_calc='gaussian', sigma=2, dim3=False, _kernel=False):
    kernel = _kernel if _kernel is not False else create_kernel(width, circular=circular, holed=holed, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    if dim3 == True:
        return filter_3d(in_raster, kernel, 'mad_std')
    return filter_2d(in_raster, kernel, 'mad_std')

def skew_fp_filter(in_raster, width=3, circular=True, holed=False, weighted_edges=True, weighted_distance=True, distance_calc='gaussian', sigma=2, dim3=False, _kernel=False):
    kernel = _kernel if _kernel is not False else create_kernel(width, circular=circular, holed=holed, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    if dim3 == True:
        return filter_3d(in_raster, kernel, 'skew_fp')
    return filter_2d(in_raster, kernel, 'skew_fp')

def skew_p2_filter(in_raster, width=3, circular=True, holed=False, weighted_edges=True, weighted_distance=True, distance_calc='gaussian', sigma=2, dim3=False, _kernel=False):
    kernel = _kernel if _kernel is not False else create_kernel(width, circular=circular, holed=holed, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    if dim3 == True:
        return filter_3d(in_raster, kernel, 'skew_p2')
    return filter_2d(in_raster, kernel, 'skew_p2')

def skew_g_filter(in_raster, width=3, circular=True, holed=False, weighted_edges=True, weighted_distance=True, distance_calc='gaussian', sigma=2, dim3=False, _kernel=False):
    kernel = _kernel if _kernel is not False else create_kernel(width, circular=circular, holed=holed, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    if dim3 == True:
        return filter_3d(in_raster, kernel, 'skew_g')
    return filter_2d(in_raster, kernel, 'skew_g')

def kurtosis_filter(in_raster, width=3, circular=True, holed=False, weighted_edges=True, weighted_distance=True, distance_calc='gaussian', sigma=2, dim3=False, _kernel=False):
    kernel = _kernel if _kernel is not False else create_kernel(width, circular=circular, holed=holed, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    if dim3 == True:
        return filter_3d(in_raster, kernel, 'kurtosis')
    return filter_2d(in_raster, kernel, 'kurtosis')

def z_filter(in_raster, width=3, circular=True, holed=False, weighted_edges=True, weighted_distance=True, distance_calc='gaussian', sigma=2, dim3=False, _kernel=False):
    kernel = _kernel if _kernel is not False else create_kernel(width, circular=circular, holed=holed, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    return ne.evaluate('(in_raster - m) / s', {
        "m": mean_filter(in_raster, _kernel=kernel, dim3=dim3),
        "s": standard_deviation_filter(in_raster, _kernel=kernel, dim3=dim3)
    })

def median_deviation_filter(in_raster, width=3, absolute_value=False, circular=True, weighted_edges=True, holed=True, weighted_distance=True, distance_calc='gaussian', sigma=2, dim3=False, _kernel=False):
    kernel = _kernel if _kernel is not False else create_kernel(width, circular=circular, holed=holed, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    result = in_raster - median_filter(in_raster, _kernel=kernel, dim3=dim3)
    if absolute_value is True:
        return np.abs(result)
    return result

def mean_deviation_filter(in_raster, width=3, absolute_value=False, circular=True, weighted_edges=True, holed=True, weighted_distance=True, distance_calc='gaussian', sigma=2, dim3=False, _kernel=False):
    kernel = _kernel if _kernel is not False else create_kernel(width, circular=circular, holed=holed, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    result = in_raster - mean_filter(in_raster, _kernel=kernel, dim3=dim3)
    if absolute_value is True:
        return np.abs(result)
    return result

def snr_filter(in_raster, width=3, circular=True, weighted_edges=True, weighted_distance=True, distance_calc='gaussian', sigma=2, dim3=False, _kernel=False):
    kernel = _kernel if _kernel is not False else create_kernel(width, circular=circular, holed=True, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    return np.power(mean_filter(in_raster, _kernel=kernel, dim3=dim3), 2) / variance_filter(in_raster, _kernel=kernel, dim3=dim3)

def normalise_to_range_filter(in_raster, low=0, high=255):
    return (in_raster - low) / (high - low)

def dilation_filter(in_raster, width=3, iterations=1, circular=True, weighted_distance=True, sigma=3.5, dim3=False, _kernel=False):
    kernel = _kernel if _kernel is not False else create_kernel(width, circular=circular, weighted_edges=True, weighted_distance=weighted_distance, sigma=sigma, normalise=False)
    if iterations == 1:
        if dim3 == True:
            return filter_3d(in_raster, kernel, 'dilate').astype(in_raster.dtype)
        return filter_2d(in_raster, kernel, 'dilate').astype(in_raster.dtype)
    if dim3 == True:
        result = filter_3d(in_raster, kernel, 'dilate').astype(in_raster.dtype)
        for _ in range(iterations - 1):
            result = filter_3d(result, kernel, 'dilate').astype(in_raster.dtype)
    else:
        result = filter_2d(in_raster, kernel, 'dilate').astype(in_raster.dtype)
        for _ in range(iterations - 1):
            result = filter_2d(result, kernel, 'dilate').astype(in_raster.dtype)
    return result
    

def erosion_filter(in_raster, width=3, iterations=1, circular=True, weighted_distance=True, sigma=3.5, dim3=False, _kernel=False):
    kernel = _kernel if _kernel is not False else create_kernel(width, circular=circular, weighted_edges=True, weighted_distance=weighted_distance, sigma=sigma, normalise=False)
    if iterations == 1:
        if dim3 == True:
            return filter_3d(in_raster, kernel, 'erode').astype(in_raster.dtype)
        return filter_2d(in_raster, kernel, 'erode').astype(in_raster.dtype)
    if dim3 == True:
        result = filter_3d(in_raster, kernel, 'erode').astype(in_raster.dtype)
        for _ in range(iterations - 1):
            result = filter_3d(result, kernel, 'erode').astype(in_raster.dtype)
    else:
        result = filter_2d(in_raster, kernel, 'erode').astype(in_raster.dtype)
        for _ in range(iterations - 1):
            result = filter_2d(result, kernel, 'erode').astype(in_raster.dtype)
    return result

def open_filter(in_raster, width=3, circular=True, weighted_distance=True, sigma=3.5, dim3=False, _kernel=False):
    kernel = _kernel if _kernel is not False else create_kernel(width, circular=circular, weighted_edges=True, weighted_distance=weighted_distance, sigma=sigma, normalise=False)
    if dim3 == True:
        return filter_3d(filter_3d(in_raster, kernel, 'erode'), kernel, 'dilate').astype(in_raster.dtype)
    return filter_2d(filter_2d(in_raster, kernel, 'erode'), kernel, 'dilate').astype(in_raster.dtype)

def close_filter(in_raster, width=3, circular=True, weighted_distance=True, sigma=3.5, dim3=False, _kernel=False):
    kernel = _kernel if _kernel is not False else create_kernel(width, circular=circular, weighted_edges=True, weighted_distance=weighted_distance, sigma=sigma, normalise=False)
    if dim3 == True:
        return filter_3d(filter_3d(in_raster, kernel, 'dilate'), kernel, 'erode').astype(in_raster.dtype)
    return filter_2d(filter_2d(in_raster, kernel, 'dilate'), kernel, 'erode').astype(in_raster.dtype)


def lee_filter(in_raster, width=3, circular=True, dim3=False):
    mean = mean_filter(in_raster, width, circular=circular)
    squared = np.power(in_raster, 2)
    sqr_mean = mean_filter(squared, width, circular=circular)
    img_variance = ne.evaluate('(sqr_mean - (mean ** 2))')
    overall_variance = np.var(in_raster)
    weights = ne.evaluate('(img_variance / (img_variance + overall_variance))')

    return ne.evaluate('mean + (weights * (in_raster - mean))')

def threshold_filter(in_raster, threshold_value, pass_value, threshold='binary', dtype='uint8'):
    if threshold == 'binary':
        threshold = cv2.THRESH_BINARY
    elif threshold == 'binary_inverse':
        threshold = cv2.THRESH_BINARY_INV
    elif threshold == 'to_zero':
        threshold = cv2.THRESH_TOZERO
    elif threshold == 'to_zero_inverse':
        threshold = cv2.THRESH_TOZERO_INV
    elif threshold == 'truncate':
        threshold = cv2.THRESH_TRUNC

    return cv2.threshold(in_raster, threshold_value, pass_value, threshold)[1].astype(dtype)

def bi_truncation_filter(in_raster, t_min, t_max):
    low = cv2.threshold(in_raster, t_min, 1, cv2.THRESH_BINARY_INV)[1]
    high = cv2.threshold(in_raster, t_max, 1, cv2.THRESH_BINARY)[1]
    middle = ne.evaluate('((high - 1) ** 2) - low')

    return ne.evaluate('(low * t_min) + (middle * in_raster) + (high * t_max)')
