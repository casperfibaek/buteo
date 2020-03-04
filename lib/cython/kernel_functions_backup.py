import numpy as np
# import pyximport
# pyximport.install()
# from c_filter import filter_2d, filter_median
from math import floor, sqrt
from time import time
import matplotlib as mpl
from matplotlib import pyplot as plt

# import sys
# sys.path.append('../lib/base')
# sys.path.append('../lib/filters')
# from filters import stdev_filter, sum_filter
# from raster_io import raster_to_array, array_to_raster


def circular_kernel_mask(width, weighted_edges=False, normalise=False, inverted=False, dtype='float32'):
    assert(width % 2 is not 0)

    radius = floor(width / 2) # 4
    # mask = np.empty((width, width), dtype=dtype)
    mask = np.zeros((width, width), dtype=dtype)


    for x in range(width):
        for y in range(width):
            xm = x - radius
            ym = y - radius
            dist = sqrt(pow(xm, 2) + pow(ym, 2))

            weight = 0

            if dist > radius:
                if weighted_edges is True:
                    if dist >= (radius + 1):
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

            # import pdb; pdb.set_trace()
            mask[x][y] = weight if inverted is False else 1 - weight

            # import pdb; pdb.set_trace()


    if normalise is True:
        return np.divide(mask, mask.sum())

    return mask


kernel = circular_kernel_mask(5, normalise=False, weighted_edges=True).astype(np.float)
# kernel = circular_kernel_mask(5, normalise=False, weighted_edges=True).astype(np.float)

fig, ax = plt.subplots()
im = ax.imshow(kernel, cmap='Greys')
circle = plt.Circle((2, 2), 2.5, color='blue', fill=False, linestyle='-')
ax.add_artist(circle)

plt.colorbar(im)


plt.show()

# folder = 'C:\\Users\\caspe\\Desktop\\Data\\Ghana\\'
# in_raster = f'{folder}gar_roads_rasterized.tif'
# out_raster = f'{folder}gar_roads_rasterized-1km-radius-roadsegments.tif'

# raster_arr = raster_to_array(in_raster).astype(np.float)

# before2 = time()
# variance = stdev_filter(raster_arr, width=11)
# median = filter_median(raster_arr, kernel)
# array_to_raster(filter_2d(raster_arr, kernel), out_raster=out_raster, reference_raster=in_raster, dst_nodata=None)
# array_to_raster(sum_filter(raster_arr, width=201), out_raster=out_raster, reference_raster=in_raster, dst_nodata=None)
# print(time() - before2)

# import pdb; pdb.set_trace()
# array_to_raster(variance, out_raster=out_raster, reference_raster=in_raster, dst_nodata=None)
# array_to_raster(median, out_raster=out_raster, reference_raster=in_raster, dst_nodata=None)
