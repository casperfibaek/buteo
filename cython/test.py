import numpy as np
from c_filter import filter_2d, filter_median
from math import floor, sqrt
from time import time

import sys
sys.path.append('../lib')
from raster_io import raster_to_array, array_to_raster


def circular_kernel_mask(width, weighted_edges=False, normalise=False, inverted=False, dtype='float32'):
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

    if normalise is True:
        return np.divide(mask, mask.sum())

    return mask


kernel = circular_kernel_mask(5, normalise=True, weighted_edges=True).astype(np.float)

in_raster = 'C:\\Users\\CFI\\Desktop\\surf_test\\surf_wet.tif'
# in_raster = 'C:\\Users\\CFI\\Desktop\\surf_test\\b4_med7_absdif.tif'
out_raster = 'C:\\Users\\CFI\\Desktop\\surf_test\\'

raster_arr = raster_to_array(in_raster).astype(np.float)

before2 = time()
median = filter_median(raster_arr, kernel)
# absdif = np.abs(np.subtract(raster_arr, median)).astype('uint16')
print(time() - before2)

# import pdb; pdb.set_trace()
# array_to_raster(filter_2d(raster_arr, kernel), out_raster=out_raster + 'b4_blur.tif', reference_raster=in_raster, dst_nodata=None)
array_to_raster(median, out_raster=out_raster + 'surf_wet_maddif.tif', reference_raster=in_raster, dst_nodata=None)
