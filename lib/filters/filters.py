import cv2
import numpy as np
import numpy.ma as ma
from math import sqrt, floor
from scipy.ndimage import median_filter as sci_median_filter
from skimage.util.shape import view_as_windows
from skimage.util import pad


def circular_kernel_mask(width, weighted_edges=False, inverted=False, average=False, dtype='float32'):
    assert(width % 2 is not 0)

    radius = floor(width / 2)
    mask = np.empty((width, width), dtype=dtype)

    for x in range(width):
        for y in range(width):
            xm = x - radius
            ym = y - radius
            dist = sqrt(pow(xm, 2) + pow(ym, 2))

            weight = 0

            if dist > radius:
                if weighted_edges is True:
                    if dist > (radius + 1):
                        weight = 0
                    else:
                        weight = 1 - (dist - int(dist))
                else:
                    weight = 0
            else:
                weight = 1

            if weighted_edges is False:
                if weight > 0.5:
                    weight = 1
                else:
                    weight = 0

            mask[x][y] = weight if inverted is False else 1 - weight

    if average is True:
        if weighted_edges is True:
            return np.multiply(mask, 1 / mask.size)
        else:
            return np.multiply(mask, 1 / mask.sum())

    return mask


def mean_filter(img_arr, width=3, circular=True, out_dtype=None):
    assert(width % 2 is not 0)
    assert(isinstance(img_arr, np.ndarray))
    assert(isinstance(circular, bool))
    out_dtype = img_arr.dtype if out_dtype is None else out_dtype

    if circular is True:
        kernel = circular_kernel_mask(width, weighted_edges=False, average=True)
        return cv2.filter2D(img_arr, -1, kernel).astype(out_dtype)
    else:
        return cv2.blur(img_arr, (width, width)).astype(out_dtype)


def sum_filter(img_arr, width=3, circular=True, weighted_edges=True, out_dtype=None):
    assert(width % 2 is not 0)
    assert(isinstance(img_arr, np.ndarray))
    assert(isinstance(circular, bool))
    assert(isinstance(weighted_edges, bool))
    out_dtype = img_arr.dtype if out_dtype is None else out_dtype

    if circular is True:
        kernel = circular_kernel_mask(width, weighted_edges=weighted_edges, average=False)
    else:
        kernel = np.ones((width, width), dtype='float32')

    return cv2.filter2D(img_arr, -1, kernel).astype(out_dtype)


def median_filter(img_arr, width=3, circular=True, out_dtype=None):
    assert(width % 2 is not 0)
    assert(isinstance(img_arr, np.ndarray))
    out_dtype = img_arr.dtype if out_dtype is None else out_dtype

    if circular is True:
        mask = circular_kernel_mask(width, dtype='uint8')
        return sci_median_filter(img_arr, footprint=mask).astype(out_dtype)
    else:
        return cv2.medianBlur(img_arr, width).astype(out_dtype)


def gaussian_filter(img_arr, width=3, sigmaX=0, sigmaY=0, out_dtype=None):
    assert(width % 2 is not 0)
    assert(isinstance(img_arr, np.ndarray))
    out_dtype = img_arr.dtype if out_dtype is None else out_dtype

    # kernel = cv2.getGaussianKernel(width, -1)

    return cv2.GaussianBlur(img_arr, (width, width), sigmaX, sigmaY=sigmaY).astype(out_dtype)


def bilateral_filter(img_arr, width=3, sigmaColor=81, sigmaSpace=81, out_dtype=None):
    assert(width % 2 is not 0)
    assert(isinstance(img_arr, np.ndarray))
    out_dtype = img_arr.dtype if out_dtype is None else out_dtype

    return cv2.bilateralFilter(img_arr, width, sigmaColor, sigmaSpace).astype(out_dtype)


def variance_filter(img_arr, width=3, circular=True, out_dtype=None):
    assert(width % 2 is not 0)
    assert(isinstance(img_arr, np.ndarray))
    assert(isinstance(circular, bool))
    out_dtype = img_arr.dtype if out_dtype is None else out_dtype

    mean_squared = mean_filter(np.power(img_arr, 2), width, circular=circular, out_dtype=out_dtype)
    squared_mean = np.power(mean_filter(img_arr, width, circular=circular, out_dtype=out_dtype), 2)

    return np.subtract(mean_squared, squared_mean).astype(out_dtype)


def stdev_filter(img_arr, width, circular=True, out_dtype=None):
    assert(width % 2 is not 0)
    assert(isinstance(img_arr, np.ndarray))
    assert(isinstance(circular, bool))
    out_dtype = img_arr.dtype if out_dtype is None else out_dtype

    return np.sqrt(variance_filter(img_arr, width, circular=circular, out_dtype=out_dtype))


def local_z_filter(img_arr, width, circular=True, out_dtype=None):
    assert(width % 2 is not 0)
    assert(isinstance(img_arr, np.ndarray))
    assert(isinstance(circular, bool))
    out_dtype = img_arr.dtype if out_dtype is None else out_dtype

    mean = mean_filter(img_arr, width, circular=circular)
    stdev = stdev_filter(img_arr, width, circular=circular)

    return np.divide(np.subtract(img_arr, mean), stdev).astype(out_dtype)


def standardize_filter(img_arr, out_dtype=None):
    assert(isinstance(img_arr, np.ndarray))
    out_dtype = img_arr.dtype if out_dtype is None else out_dtype

    return np.divide(np.subtract(img_arr, np.mean(img_arr)), np.std(img_arr)).astype(out_dtype)


def mad_filter(img_arr, out_dtype=None):
    assert(isinstance(img_arr, np.ndarray))
    out_dtype = img_arr.dtype if out_dtype is None else out_dtype

    median = ma.median(img_arr)
    deviations = np.subtract(median, img_arr)
    abs_deviation = np.abs(deviations)
    mad = ma.median(abs_deviation)
    mad_stdev = mad * 1.4826

    return np.divide(np.subtract(img_arr, median), mad_stdev).astype(out_dtype)


def local_madstd_filter(img_arr, width=3, circular=True, out_dtype=None):
    assert(isinstance(img_arr, np.ndarray))
    out_dtype = img_arr.dtype if out_dtype is None else out_dtype

    windows = view_as_windows(pad(img_arr, pad_width=int(floor(width / 2)), mode='edge'), (width, width))

    if circular is True:
        mask = circular_kernel_mask(width, dtype='uint8')
        mask_inv = circular_kernel_mask(width, dtype='uint8', inverted=True)
        med = sci_median_filter(img_arr, footprint=mask)

        tiled_mask = np.tile(mask_inv, (img_arr.shape[0], img_arr.shape[1], 1, 1))

        windows = ma.array(windows, mask=np.tile(mask_inv, (img_arr.shape[0], img_arr.shape[1], 1, 1)))

        return np.multiply(ma.median(np.abs(np.subtract(med[:, :, np.newaxis, np.newaxis], windows)), axis=(2, 3)), 1.4826).astype(out_dtype)
    else:
        med = median_filter(img_arr, width, circular=False, out_dtype=out_dtype)
        return np.multiply(np.median(np.abs(np.subtract(med[:, :, np.newaxis, np.newaxis], windows)), axis=(2, 3)), 1.4826).astype(out_dtype)


def normalise_to_range_filter(img_arr, new_range=(0, 255), out_dtype=None):
    assert(isinstance(img_arr, np.ndarray))
    out_dtype = img_arr.dtype if out_dtype is None else out_dtype

    return np.divide((img_arr - new_range[0]), (new_range[1] - new_range[0])).astype(out_dtype)


def normalise_filter(img_arr, out_dtype=None):
    assert(isinstance(img_arr, np.ndarray))
    out_dtype = img_arr.dtype if out_dtype is None else out_dtype

    return np.divide(img_arr, img_arr.max()).astype(out_dtype)


def zobel_filter(img_arr, width=3, scale=1, delta=0, out_dtype=None):
    assert(isinstance(img_arr, np.ndarray))
    out_dtype = img_arr.dtype if out_dtype is None else out_dtype
    x = cv2.Sobel(img_arr, -1, 1, 0, ksize=width, scale=scale, delta=delta)
    y = cv2.Sobel(img_arr, -1, 0, 1, ksize=width, scale=scale, delta=delta)

    return np.add(np.multiply(x, 0.5), np.multiply(y, 0.5)).astype(out_dtype)


def median_deviation_filter(img_arr, width, circular=True, out_dtype=None, inverse=False):
    assert(width % 2 is not 0)
    assert(isinstance(img_arr, np.ndarray))
    out_dtype = img_arr.dtype if out_dtype is None else out_dtype

    median = median_filter(img_arr, width, circular=circular, out_dtype=out_dtype)

    difference = np.subtract(median, img_arr) if inverse is False else np.subtract(img_arr, median)
    return difference.astype(out_dtype)


def mean_deviation_filter(img_arr, width, circular=True, out_dtype=None, inverse=False):
    assert(width % 2 is not 0)
    assert(isinstance(img_arr, np.ndarray))
    out_dtype = img_arr.dtype if out_dtype is None else out_dtype

    mean = mean_filter(img_arr, width, circular=circular, out_dtype=out_dtype)

    difference = np.subtract(mean, img_arr) if inverse is False else np.subtract(img_arr, mean)
    return difference.astype(out_dtype)


def simple_skew_filter(img_arr, width, circular=True, out_dtype=None, inverse=False):
    assert(width % 2 is not 0)
    assert(isinstance(img_arr, np.ndarray))
    out_dtype = img_arr.dtype if out_dtype is None else out_dtype

    median = median_filter(img_arr, width, circular=circular, out_dtype=out_dtype)
    mean = mean_filter(img_arr, width, circular=False, out_dtype=out_dtype)

    if inverse is False:
        return np.subtract(mean, median).astype(out_dtype)
    else:
        return np.subtract(median, mean).astype(out_dtype)


def dilation_filter(img_arr, width, circular=True, iterations=1, out_dtype=None):
        assert(width % 2 is not 0)
        assert(isinstance(img_arr, np.ndarray))
        out_dtype = img_arr.dtype if out_dtype is None else out_dtype

        if circular is True:
            kernel = circular_kernel_mask(width, dtype='uint8')
        else:
            kernel = np.ones((width, width), dtype='float32')

        return cv2.dilate(img_arr, kernel, iterations=iterations).astype(out_dtype)


def erosion_filter(img_arr, width, circular=True, iterations=1, out_dtype=None):
        assert(width % 2 is not 0)
        assert(isinstance(img_arr, np.ndarray))
        out_dtype = img_arr.dtype if out_dtype is None else out_dtype

        if circular is True:
            kernel = circular_kernel_mask(width, dtype='uint8')
        else:
            kernel = np.ones((width, width), dtype='float32')

        return cv2.erode(img_arr, kernel, iterations=iterations).astype(out_dtype)


def lee_filter(img_arr, width, circular=True, out_dtype=None):
        assert(width % 2 is not 0)
        assert(isinstance(img_arr, np.ndarray))
        out_dtype = img_arr.dtype if out_dtype is None else out_dtype

        mean = mean_filter(img_arr, width, circular=circular, out_dtype=out_dtype)
        squared = np.power(img_arr, 2)
        sqr_mean = mean_filter(squared, width, circular=circular, out_dtype=out_dtype)
        img_variance = np.subtract(sqr_mean, np.power(mean, 2))

        overall_variance = np.var(img_arr)
        weights = np.divide(img_variance, (np.add(img_variance, overall_variance)))
        output = np.add(mean, np.multiply(weights, np.subtract(img_arr, mean)))

        return output.astype(out_dtype)


def threshold_filter(img_arr, threshold_value, pass_value, threshold='binary', out_dtype=None):
    assert(isinstance(img_arr, np.ndarray))
    out_dtype = img_arr.dtype if out_dtype is None else out_dtype

    if threshold is 'binary':
        threshold = cv2.THRESH_BINARY
    elif threshold is 'binary_inverse':
        threshold = cv2.THRESH_BINARY_INV
    elif threshold is 'to_zero':
        threshold is cv2.THRESH_TOZERO
    elif threshold is 'to_zero_inverse':
        threshold is cv2.THRESH_TOZERO_INV
    elif threshold is 'truncate':
        threshold = cv2.THRESH_TRUNC

    return cv2.threshold(img_arr, threshold_value, pass_value, threshold)[1].astype(out_dtype)


def bi_truncation_filter(img_arr, threshold_value_min, threshold_value_max, out_dtype=None):
    assert(isinstance(img_arr, np.ndarray))
    out_dtype = img_arr.dtype if out_dtype is None else out_dtype

    binary_mask_low = cv2.threshold(img_arr, threshold_value_min, 1, cv2.THRESH_BINARY_INV)[1]
    binary_mask_high = cv2.threshold(img_arr, threshold_value_max, 1, cv2.THRESH_BINARY)[1]
    binary_mask_middle = np.subtract(np.power(np.subtract(binary_mask_high, 1), 2), binary_mask_low)

    low = np.multiply(binary_mask_low, threshold_value_min)
    middle = np.multiply(binary_mask_middle, img_arr)
    high = np.multiply(binary_mask_high, threshold_value_max)

    return np.add(np.add(low, middle), high).astype(out_dtype)


if __name__ == "__main__":
    import sys
    sys.path.append('../lib')

    from raster_io import raster_to_array, array_to_raster

    in_raster = 'C:\\Users\\CFI\\Desktop\\surf_test\\b4.tif'
    out_raster = 'C:\\Users\\CFI\\Desktop\\surf_test\\'

    raster_arr = raster_to_array(in_raster).astype('float32')
    # array_to_raster(median_filter(raster_arr, 5, circular=True), out_raster=out_raster + 'b4_median_circ.tif', reference_raster=in_raster, dst_nodata=None)
    # array_to_raster(sum_filter(raster_arr, 5), out_raster=out_raster + 'b4_sum.tif', reference_raster=in_raster, dst_nodata=None)
    # array_to_raster(mean_filter(raster_arr, 5), out_raster=out_raster + 'b4_mean.tif', reference_raster=in_raster, dst_nodata=None)
    # array_to_raster(gaussian_filter(raster_arr, 5), out_raster=out_raster + 'b4_gaussian.tif', reference_raster=in_raster, dst_nodata=None)
    # array_to_raster(bilateral_filter(raster_arr, 5), out_raster=out_raster + 'b4_billateral.tif', reference_raster=in_raster, dst_nodata=None)
    # array_to_raster(variance_filter(raster_arr, 5), out_raster=out_raster + 'b4_variance.tif', reference_raster=in_raster, dst_nodata=None)
    # array_to_raster(stdev_filter(raster_arr, 5), out_raster=out_raster + 'b4_stdev.tif', reference_raster=in_raster, dst_nodata=None)
    # array_to_raster(median_filter(local_z_filter(raster_arr, 101), 5), out_raster=out_raster + 'b4_local_z_101_median.tif', reference_raster=in_raster, dst_nodata=None)
    # array_to_raster(standardize_filter(raster_arr), out_raster=out_raster + 'b4_standardize.tif', reference_raster=in_raster, dst_nodata=None)
    # array_to_raster(normalise_to_range_filter(raster_arr, (0, 255)), out_raster=out_raster + 'b4_normalize_0_255.tif', reference_raster=in_raster, dst_nodata=None)
    # array_to_raster(normalise_filter(raster_arr), out_raster=out_raster + 'b4_normalize.tif', reference_raster=in_raster, dst_nodata=None)
    # array_to_raster(zobel_filter(raster_arr, 5), out_raster=out_raster + 'b4_zobel.tif', reference_raster=in_raster, dst_nodata=None)
    # array_to_raster(mad_filter(raster_arr), out_raster=out_raster + 'b4_mad.tif', reference_raster=in_raster, dst_nodata=None)
    array_to_raster(local_madstd_filter(raster_arr, 5, circular=False), out_raster=out_raster + 'b4_local_madstd_circ.tif', reference_raster=in_raster, dst_nodata=None)
    # array_to_raster(median_deviation_filter(raster_arr, 5), out_raster=out_raster + 'b4_med_dev.tif', reference_raster=in_raster, dst_nodata=None)
    # array_to_raster(bilateral_filter(mad_filter(mean_deviation_filter(raster_arr, 5)), 11), out_raster=out_raster + 'b4_magic.tif', reference_raster=in_raster, dst_nodata=None)
    # array_to_raster(simple_skew_filter(raster_arr, 5), out_raster=out_raster + 'b4_simple_skew.tif', reference_raster=in_raster, dst_nodata=None)
    # array_to_raster(dilation_filter(raster_arr, 5, iterations=3), out_raster=out_raster + 'b4_dilate.tif', reference_raster=in_raster, dst_nodata=None)
    # array_to_raster(erosion_filter(dilation_filter(raster_arr, 5, iterations=5), 5, iterations=5), out_raster=out_raster + 'b4_erode_dilate.tif', reference_raster=in_raster, dst_nodata=None)
    # array_to_raster(np.subtract(dilation_filter(raster_arr, 3), raster_arr), out_raster=out_raster + 'b4_dilate_edge.tif', reference_raster=in_raster, dst_nodata=None)
    # array_to_raster(sum_filter(binary_filter(bilateral_filter(lee_filter(raster_arr, 3), 3), 3, 1), 21), out_raster=out_raster + 'surf_lee_bilateral_binary_summed_trun3.tif', reference_raster=in_raster, dst_nodata=None)
    # array_to_raster(bi_threshold_filter(raster_arr, 0, 2, threshold='truncate'), out_raster=out_raster + 'dry_truncked_middle.tif', reference_raster=in_raster, dst_nodata=None)
