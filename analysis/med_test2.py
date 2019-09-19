import numpy as np
import numpy.ma as ma
from skimage.util.shape import view_as_windows
from skimage.util import pad
from scipy.ndimage import median_filter
from timeit import timeit

image = np.arange(49).reshape(7, 7)

circle_kernel = np.array([
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0]], dtype='uint8')


median = median_filter(image, footprint=circle_kernel)

# It appears Numpy reads masks the other way around.
inverted_kernel = np.array([
    [1, 1, 0, 1, 1],
    [1, 0, 0, 0, 1],
    [0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1],
    [1, 1, 0, 1, 1]], dtype='uint8')

padded = pad(image, pad_width=2, mode='edge')
windows = view_as_windows(padded, (5, 5))
masked_windows = ma.array(windows, mask=np.tile(inverted_kernel, (image.shape[0], image.shape[1], 1, 1)))
deviations_from_median = np.subtract(median[:, :, np.newaxis, np.newaxis], masked_windows)
absolute_deviations = np.abs(deviations_from_median)
median_of_deviations = ma.median(absolute_deviations, axis=(2, 3))

import pdb; pdb.set_trace()


# def med():
    # 0.00466029999999984
    # return median_filter(image, footprint=circle_kernel)
    # 0.004236399999999918
    # return pad(image, pad_width=2, mode='edge')
    # 0.004502999999999702
    # return view_as_windows(padded, (5, 5))
    # 0.003247
    # return ma.array(windows, mask=np.tile(inverted_kernel, (image.shape[0], image.shape[1], 1, 1)))
    # 0.009165699999999832
    # return np.subtract(median[:, :, np.newaxis, np.newaxis], masked_windows)
    # 0.11729520000000004
    # return np.array(ma.median(np.abs(deviations_from_median), axis=(2, 3)))
    # 0.005040799999999956
    # return np.abs(deviations_from_median)
    # 0.07622720000000038
    # return ma.median(absolute_deviations, axis=(2, 3))
    # 0.0001538999999999291
    # return np.array(median_of_deviations)
