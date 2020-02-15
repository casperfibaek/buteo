from math import floor, sqrt
from skimage.util.shape import view_as_windows
from skimage.util import pad
from scipy.ndimage import median_filter as sci_median_filter
import numpy as np
import numpy.ma as ma


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

filt = 3
arr = np.array([
    [2, 1, 3, 2, 5],
    [2, 2, 3, 1, 5],
    [2, 3, 9, 5, 5],
    [2, 2, 3, 4, 5],
    [2, 1, 3, 1, 5],
])


windows = view_as_windows(pad(arr, pad_width=int(floor(filt / 2)), mode='edge'), (filt, filt))

mask = circular_kernel_mask(5, dtype='uint8')
mask_inv = circular_kernel_mask(5, dtype='uint8', inverted=True)

# med = sci_median_filter(arr, footprint=mask)

# masked_windows = ma.array(windows, mask=np.tile(mask_inv, (arr.shape[0], arr.shape[1], 1, 1)))
# dev = np.subtract(med[:, :, np.newaxis, np.newaxis], masked_windows)
# absdev = np.abs(dev)
# medabsdev = np.median(absdev, axis=(2, 3))

import pdb; pdb.set_trace()
