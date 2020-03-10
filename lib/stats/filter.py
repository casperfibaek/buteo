import cv2
import numpy as np
import numexpr as ne
from base_filters import filter_2d, filter_3d
from kernel import create_kernel

def standardise_filter(in_raster):
    return ne.evaluate('(in_raster - mean) / std', { "mean": np.mean(in_raster), "std": np.std(in_raster) })

def normalise_filter(in_raster):
    return ne.evaluate('(in_raster - mi) / (ma - mi)', { "mi": np.min(in_raster), "ma": np.max(in_raster) })

def sum_filter(in_raster, width=3, circular=True, holed=False, weighted_edges=True, weighted_distance=False, distance_calc='gaussian', sigma=2, dim3=False):
    kernel = create_kernel(width, circular=circular, holed=holed, normalise=False, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    if dim3 == True:
        filter_3d(in_raster, kernel, 'mean')
    return filter_2d(in_raster, kernel, 'mean')

def mean_filter(in_raster, width=3, circular=True, holed=False, weighted_edges=True, weighted_distance=True, distance_calc='gaussian', sigma=2, dim3=False):
    kernel = create_kernel(width, circular=circular, holed=holed, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    if dim3 == True:
        filter_3d(in_raster, kernel, 'mean')
    return filter_2d(in_raster, kernel, 'mean')

def median_filter(in_raster, width=3, circular=True, holed=False, weighted_edges=True, weighted_distance=True, distance_calc='gaussian', sigma=2, dim3=False):
    kernel = create_kernel(width, circular=circular, holed=holed, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    if dim3 == True:
        filter_3d(in_raster, kernel, 'median')
    return filter_2d(in_raster, kernel, 'median')

def variance_filter(in_raster, width=3, circular=True, holed=False, weighted_edges=True, weighted_distance=True, distance_calc='gaussian', sigma=2, dim3=False):
    kernel = create_kernel(width, circular=circular, holed=holed, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    if dim3 == True:
        filter_3d(in_raster, kernel, 'variance')
    return filter_2d(in_raster, kernel, 'variance')

def standard_deviation_filter(in_raster, width=3, circular=True, holed=False, weighted_edges=True, weighted_distance=True, distance_calc='gaussian', sigma=2, dim3=False):
    kernel = create_kernel(width, circular=circular, holed=holed, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    if dim3 == True:
        filter_3d(in_raster, kernel, 'standard_deviation')
    return filter_2d(in_raster, kernel, 'standard_deviation')

def q1_filter(in_raster, width=3, circular=True, holed=False, weighted_edges=True, weighted_distance=True, distance_calc='gaussian', sigma=2, dim3=False):
    kernel = create_kernel(width, circular=circular, holed=holed, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    if dim3 == True:
        filter_3d(in_raster, kernel, 'q1')
    return filter_2d(in_raster, kernel, 'q1')

def q3_filter(in_raster, width=3, circular=True, holed=False, weighted_edges=True, weighted_distance=True, distance_calc='gaussian', sigma=2, dim3=False):
    kernel = create_kernel(width, circular=circular, holed=holed, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    if dim3 == True:
        filter_3d(in_raster, kernel, 'q3')
    return filter_2d(in_raster, kernel, 'q3')

def iqr_filter(in_raster, width=3, circular=True, holed=False, weighted_edges=True, weighted_distance=True, distance_calc='gaussian', sigma=2, dim3=False):
    kernel = create_kernel(width, circular=circular, holed=holed, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    if dim3 == True:
        filter_3d(in_raster, kernel, 'iqr')
    return filter_2d(in_raster, kernel, 'iqr')

def mad_filter(in_raster, width=3, circular=True, holed=False, weighted_edges=True, weighted_distance=True, distance_calc='gaussian', sigma=2, dim3=False):
    kernel = create_kernel(width, circular=circular, holed=holed, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    if dim3 == True:
        filter_3d(in_raster, kernel, 'mad')
    return filter_2d(in_raster, kernel, 'mad')

def mad_std_filter(in_raster, width=3, circular=True, holed=False, weighted_edges=True, weighted_distance=True, distance_calc='gaussian', sigma=2, dim3=False):
    kernel = create_kernel(width, circular=circular, holed=holed, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    if dim3 == True:
        filter_3d(in_raster, kernel, 'mad_std')
    return filter_2d(in_raster, kernel, 'mad_std')

def skew_fp_filter(in_raster, width=3, circular=True, holed=False, weighted_edges=True, weighted_distance=True, distance_calc='gaussian', sigma=2, dim3=False):
    kernel = create_kernel(width, circular=circular, holed=holed, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    if dim3 == True:
        filter_3d(in_raster, kernel, 'skew_fp')
    return filter_2d(in_raster, kernel, 'skew_fp')

def skew_p2_filter(in_raster, width=3, circular=True, holed=False, weighted_edges=True, weighted_distance=True, distance_calc='gaussian', sigma=2, dim3=False):
    kernel = create_kernel(width, circular=circular, holed=holed, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    if dim3 == True:
        filter_3d(in_raster, kernel, 'skew_p2')
    return filter_2d(in_raster, kernel, 'skew_p2')

def skew_g_filter(in_raster, width=3, circular=True, holed=False, weighted_edges=True, weighted_distance=True, distance_calc='gaussian', sigma=2, dim3=False):
    kernel = create_kernel(width, circular=circular, holed=holed, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    if dim3 == True:
        filter_3d(in_raster, kernel, 'skew_g')
    return filter_2d(in_raster, kernel, 'skew_g')

def kurtosis_filter(in_raster, width=3, circular=True, holed=False, weighted_edges=True, weighted_distance=True, distance_calc='gaussian', sigma=2, dim3=False):
    kernel = create_kernel(width, circular=circular, holed=holed, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    if dim3 == True:
        filter_3d(in_raster, kernel, 'kurtosis')
    return filter_2d(in_raster, kernel, 'kurtosis')

def z_filter(in_raster, width=3, circular=True, holed=False, weighted_edges=True, weighted_distance=True, distance_calc='gaussian', sigma=2, dim3=False):
    kernel = create_kernel(width, circular=circular, holed=holed, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    if dim3 == True:
        return ne.evaluate('(in_raster - m) / s', {
            "m": mean_filter(in_raster, kernel, dim3=True),
            "s": standard_deviation_filter(in_raster, kernel, dim3=True)
        })

    return ne.evaluate('(in_raster - m) / s', {
        "m": mean_filter(in_raster, kernel),
        "s": standard_deviation_filter(in_raster, kernel)
    })

def median_deviation_filter(in_raster, width=3, circular=True, weighted_edges=True, weighted_distance=True, distance_calc='gaussian', sigma=2, dim3=False):
    kernel = create_kernel(width, circular=circular, holed=True, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    if dim3 == True:
        return ne.evaluate('(in_raster - mh)', {
            "mh": median_filter(in_raster, kernel, dim3=True),
        })

    return ne.evaluate('(in_raster - mh)', {
        "mh": median_filter(in_raster, kernel),
    })

def mean_deviation_filter(in_raster, width=3, circular=True, weighted_edges=True, weighted_distance=True, distance_calc='gaussian', sigma=2, dim3=False):
    kernel = create_kernel(width, circular=circular, holed=True, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    if dim3 == True:
        return ne.evaluate('(in_raster - mh)', {
            "mh": mean_filter(in_raster, kernel, dim3=True),
        })

    return ne.evaluate('(in_raster - mh)', {
        "mh": mean_filter(in_raster, kernel),
    })

def snr_filter(in_raster, width=3, circular=True, weighted_edges=True, weighted_distance=True, distance_calc='gaussian', sigma=2, dim3=False):
    kernel = create_kernel(width, circular=circular, holed=True, weighted_edges=weighted_edges, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma)
    if dim3 == True:
        return ne.evaluate('(m ** 2) / v', {
            "m": mean_filter(in_raster, kernel, dim3=True),
            'v': variance_filter(in_raster, kernel, dim3=True)
        })

    return ne.evaluate('(m ** 2) / v', {
        "m": mean_filter(in_raster, kernel),
        'v': variance_filter(in_raster, kernel)
    })

def normalise_to_range_filter(in_raster, low=0, high=255):
    return ne.evaluate('(in_raster - low) / (high - low)')

def dilation_filter(in_raster, width, circular=True, iterations=1):
    kernel = create_kernel(width, circular=circular, weighted_edges=True, weighted_distance=False)
    return cv2.dilate(in_raster, kernel, iterations=iterations)

def erosion_filter(in_raster, width, circular=True, iterations=1, out_dtype=None):
    kernel = create_kernel(width, circular=circular, weighted_edges=True, weighted_distance=False)
    return cv2.erode(in_raster, kernel, iterations=iterations)

def lee_filter(in_raster, width, circular=True, dim3=False):
    ne.evaluate('(mean, mean_2, arr, arr_2)')
    mean = mean_filter(in_raster, width, circular=circular)
    squared = np.power(in_raster, 2)
    sqr_mean = mean_filter(squared, width, circular=circular)
    img_variance = ne.evaluate('(sqr_mean - (mean ** 2))')
    overall_variance = np.var(in_raster)
    weights = ne.evaluate('(img_variance / (img_variance + overall_variance))')

    return ne.evaluate('mean + (weights * (in_raster - mean))')

def threshold_filter(in_raster, threshold_value, pass_value, threshold='binary'):
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

    return cv2.threshold(in_raster, threshold_value, pass_value, threshold)[1]

def bi_truncation_filter(in_raster, t_min, t_max):
    low = cv2.threshold(in_raster, t_min, 1, cv2.THRESH_BINARY_INV)[1]
    high = cv2.threshold(in_raster, t_max, 1, cv2.THRESH_BINARY)[1]
    middle = ne.evaluate('((high - 1) ** 2) - low')

    return ne.evaluate('(low * t_min) + (middle * in_raster) + (high * t_max')


if __name__ == "__main__":
    import sys
    sys.path.append('../base')

    from raster_io import raster_to_array, array_to_raster

    folder = '/mnt/c/users/caspe/desktop/data/'
    in_path = folder + 'B04.jp2'
    in_raster = raster_to_array(in_path).astype(np.double)

    # array_to_raster(standardise_filter(in_raster), out_raster=folder + 'b4_standard.tif', reference_raster=in_path)
    # array_to_raster(normalise_filter(in_raster), out_raster=folder + 'b4_norm.tif', reference_raster=in_path)
    # array_to_raster(sum_filter(in_raster, 5), out_raster=folder + 'b4_sum_5.tif', reference_raster=in_path)
    # array_to_raster(mean_filter(in_raster, 5), out_raster=folder + 'b4_mean_5.tif', reference_raster=in_path)
    array_to_raster(median_filter(in_raster, 5), out_raster=folder + 'b4_median_5.tif', reference_raster=in_path)
    array_to_raster(variance_filter(in_raster, 5), out_raster=folder + 'b4_var_5.tif', reference_raster=in_path)
    array_to_raster(standard_deviation_filter(in_raster, 5), out_raster=folder + 'b4_std_5.tif', reference_raster=in_path)
    array_to_raster(q1_filter(in_raster, 5), out_raster=folder + 'b4_q1_5.tif', reference_raster=in_path)
    array_to_raster(q3_filter(in_raster, 5), out_raster=folder + 'b4_q3_5.tif', reference_raster=in_path)
    array_to_raster(iqr_filter(in_raster, 5), out_raster=folder + 'b4_iqr_5.tif', reference_raster=in_path)
    array_to_raster(mad_filter(in_raster, 5), out_raster=folder + 'b4_mad_5.tif', reference_raster=in_path)
    array_to_raster(mad_std_filter(in_raster, 5), out_raster=folder + 'b4_mad_std_5.tif', reference_raster=in_path)
    array_to_raster(skew_fp_filter(in_raster, 5), out_raster=folder + 'b4_skew_fp_5.tif', reference_raster=in_path)
    array_to_raster(skew_p2_filter(in_raster, 5), out_raster=folder + 'b4_skew_p2_5.tif', reference_raster=in_path)
    array_to_raster(skew_g_filter(in_raster, 5), out_raster=folder + 'b4_skew_g_5.tif', reference_raster=in_path)
    array_to_raster(kurtosis_filter(in_raster, 5), out_raster=folder + 'b4_kurt_5.tif', reference_raster=in_path)
    array_to_raster(z_filter(in_raster, 5), out_raster=folder + 'b4_zscr_5.tif', reference_raster=in_path)
    array_to_raster(median_deviation_filter(in_raster, 5), out_raster=folder + 'b4_meddev_5.tif', reference_raster=in_path)
    array_to_raster(mean_deviation_filter(in_raster, 5), out_raster=folder + 'b4_meandev_5.tif', reference_raster=in_path)
    array_to_raster(snr_filter(in_raster, 5), out_raster=folder + 'b4_snr_5.tif', reference_raster=in_path)
    array_to_raster(normalise_to_range_filter(in_raster, low=0, high=255), out_raster=folder + 'b4_norm_range_5.tif', reference_raster=in_path)
    array_to_raster(dilation_filter(in_raster, 5), out_raster=folder + 'b4_dilate_5.tif', reference_raster=in_path)
    array_to_raster(erosion_filter(in_raster, 5), out_raster=folder + 'b4_erode_5.tif', reference_raster=in_path)
    array_to_raster(lee_filter(in_raster, 5), out_raster=folder + 'b4_lee_5.tif', reference_raster=in_path)
    array_to_raster(threshold_filter(in_raster, 500, 1), out_raster=folder + 'b4_thresh_bin_5.tif', reference_raster=in_path)
    array_to_raster(bi_truncation_filter(in_raster, 200, 800), out_raster=folder + 'b4_trunk_5.tif', reference_raster=in_path)