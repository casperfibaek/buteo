import cv2
import numpy as np
from math import sqrt, floor


def circular_kernel_mask(width, weighted_edges=True, inverted=False, average=False, dtype='float32'):
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


def mean_filter(img_arr, width=3, circular=True, weighted_edges=True, out_dtype=None):
    assert(width % 2 is not 0)
    assert(isinstance(img_arr, np.ndarray))
    assert(isinstance(circular, bool))
    assert(isinstance(weighted_edges, bool))

    if circular is True:
        kernel = circular_kernel_mask(width, weighted_edges=weighted_edges, average=True)
    else:
        kernel = np.ones((width, width), dtype='float32')

    out_dtype is img_arr.dtype if out_dtype is None else out_dtype

    return cv2.filter2D(img_arr, -1, kernel).astype(out_dtype)


def sum_filter(img_arr, width=3, circular=True, weighted_edges=True, out_dtype=None):
    assert(width % 2 is not 0)
    assert(isinstance(img_arr, np.ndarray))
    assert(isinstance(circular, bool))
    assert(isinstance(weighted_edges, bool))

    if circular is True:
        kernel = circular_kernel_mask(width, weighted_edges, average=False)
    else:
        kernel = np.ones((width, width), dtype='float32')

    out_dtype is img_arr.dtype if out_dtype is None else out_dtype

    return cv2.filter2D(img_arr, -1, kernel).astype(out_dtype)


def median_filter(img_arr, width=3, out_dtype=None):
    assert(width % 2 is not 0)
    assert(isinstance(img_arr, np.ndarray))

    out_dtype is img_arr.dtype if out_dtype is None else out_dtype

    return cv2.medianBlur(img_arr, width).astype(out_dtype)


def gaussian_filter(img_arr, width=3, sigmaX=0, sigmaY=0, out_dtype=None):
    assert(width % 2 is not 0)
    assert(isinstance(img_arr, np.ndarray))

    out_dtype is img_arr.dtype if out_dtype is None else out_dtype

    kernel = cv2.getGaussianKernel(width, -1)

    return cv2.GaussianBlur(img_arr, (width, width), sigmaX, sigmaY=sigmaY).astype(out_dtype)


def bilateral_filter(img_arr, width=3, sigmaColor=81, sigmaSpace=81, out_dtype=None):
    assert(width % 2 is not 0)
    assert(isinstance(img_arr, np.ndarray))

    out_dtype is img_arr.dtype if out_dtype is None else out_dtype

    return cv2.bilateralFilter(img_arr, width, sigmaColor, sigmaSpace).astype(out_dtype)


def variance_filter(img_arr, width=3, circular=True, out_dtype=None):
    assert(width % 2 is not 0)
    assert(isinstance(img_arr, np.ndarray))
    assert(isinstance(circular, bool))

    out_dtype is img_arr.dtype if out_dtype is None else out_dtype

    squared = np.power(img_arr.astype('float32'))
    mean_squared = mean_filter(squared, width, circular=circular, out_dtype='float32')
    squared_mean = np.power(mean_filter(img_arr.astype('float32'), width, circular=circular, out_dtype='float32'))

    return np.subtract(mean_squared, squared_mean).astype(out_dtype)


def stdev_filter(img_arr, width, circular=True, out_dtype=None):
    return np.sqrt(variance_filter(img_arr, width, circular, out_dtype))


if __name__ == "__main__":
    import sys
    sys.path.append('../lib')

    from raster_io import raster_to_array, array_to_raster

    in_raster = 'C:\\Users\\CFI\\Desktop\\surf_test\\dry_b04.tif'
    out_raster = 'C:\\Users\\CFI\\Desktop\\surf_test\\'

    raster_arr = raster_to_array(in_raster)
    # array_to_raster(median_filter(raster_arr, 5), out_raster=out_raster + 'b4_median.tif', reference_raster=in_raster, dst_nodata=None)
    # array_to_raster(sum_filter(raster_arr, 5), out_raster=out_raster + 'b4_sum.tif', reference_raster=in_raster, dst_nodata=None)
    # array_to_raster(mean_filter(raster_arr, 5), out_raster=out_raster + 'b4_mean.tif', reference_raster=in_raster, dst_nodata=None)
    # array_to_raster(gaussian_filter(raster_arr, 5), out_raster=out_raster + 'b4_gaussian.tif', reference_raster=in_raster, dst_nodata=None)
    # array_to_raster(bilateral_filter(raster_arr, 5), out_raster=out_raster + 'b4_billateral.tif', reference_raster=in_raster, dst_nodata=None)
    array_to_raster(mean_filter(raster_arr, 5, circular=False), out_raster=out_raster + 'b4_variance_1.tif', reference_raster=in_raster, dst_nodata=None)
    array_to_raster(mean_filter(raster_arr, 5), out_raster=out_raster + 'b4_variance_2.tif', reference_raster=in_raster, dst_nodata=None)
    # array_to_raster(stdev_filter(raster_arr, 5), out_raster=out_raster + 'b4_stdev_2.tif', reference_raster=in_raster, dst_nodata=None)
