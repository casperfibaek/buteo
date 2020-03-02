import numpy as np
import pyximport
pyximport.install()
from c_filter import filter_2d, filter_median, filter_weighted_variance
from math import floor, sqrt
from time import time
import matplotlib as mpl
from matplotlib import pyplot as plt
from shapely.geometry import Point, Polygon

import sys
sys.path.append('../lib/base')
sys.path.append('../lib/filters')
from filters import stdev_filter, sum_filter
from raster_io import raster_to_array, array_to_raster


def create_kernel(width, circular=True, weighted_edges=False, normalise=False, inverted=False, distance_weighted=True, distance_calc='invsqrt', distance_scale=True, plot=False, dtype='float32'):
    assert(width % 2 is not 0)

    radius = floor(width / 2) # 4
    kernel = np.zeros((width, width), dtype=dtype)
    dist_offset = 0.2

    for x in range(width):
        for y in range(width):
            xm = x - radius
            ym = y - radius
            dist = sqrt(pow(xm, 2) + pow(ym, 2))

            weight = 1

            if distance_weighted is True:
                if xm is 0 and ym is 0:
                    weight = 1
                else:
                    if distance_scale is True:
                        scale = sqrt((radius ** 2) + (radius ** 2)) + sqrt(0.5)
                        if distance_calc is 'invsqrt':
                            weight = 1 - (sqrt(dist) / sqrt(scale))
                        if distance_calc is 'invpow':
                            weight = 1 - (pow(dist, 2) / pow(scale, 2))
                        if distance_calc is 'linear':
                            weight = 1 - (dist / scale)
                    else:
                        if distance_calc is 'invsqrt':
                            weight = 1 / (sqrt(dist) + dist_offset)
                        if distance_calc is 'invpow':
                            weight = 1 / (pow(dist, 2) + dist_offset)
                        if distance_calc is 'linear':
                            weight = 1 / dist

            if circular is True:
                if weighted_edges is False:
                    if (dist - radius) > sqrt(0.5):
                        weight = 0
                else:
                    circle = Point(0, 0).buffer(radius + 0.5)
                    polygon = Polygon([(xm - 0.5, ym - 0.5), (xm - 0.5, ym + 0.5), (xm + 0.5, ym + 0.5), (xm + 0.5, ym - 0.5)])
                    intersection = polygon.intersection(circle)

                    # Area of a pixel is 1, no need to normalise.
                    weight = weight * intersection.area

            kernel[x][y] = weight if inverted is False else 1 - weight



    if normalise is True:
        kernel = np.divide(kernel, kernel.sum())

    if plot is True:
        fig, ax = plt.subplots()

        if normalise is False:
            im = ax.imshow(kernel, cmap='Greys', interpolation=None, vmin=0, vmax=1)
        else:
            im = ax.imshow(kernel, cmap='Greys', interpolation=None)

        circle = plt.Circle((radius, radius), radius + 0.5, color='blue', fill=False, linestyle='-')
        ax.add_artist(circle)

        ax.invert_yaxis()

        plt.colorbar(im)
        plt.show()

    return kernel


kernel = create_kernel(11, circular=False, weighted_edges=True, normalise=True, distance_weighted=False, distance_scale=False, distance_calc='linear').astype(np.float)


folder = 'C:\\Users\\caspe\\Desktop\\Data\\Sentinel2\\'
in_raster = f'{folder}egypt_b8.jp2'
out_raster = f'{folder}egypt_b8_weighted_stdev_cv2.tif'

raster_arr = raster_to_array(in_raster).astype(np.float)

before2 = time()
# variance = stdev_filter(raster_arr, width=11)
# median = filter_median(raster_arr, kernel)
# array_to_raster(np.sqrt(filter_weighted_variance(raster_arr, kernel)), out_raster=out_raster, reference_raster=in_raster, dst_nodata=None)
array_to_raster(stdev_filter(raster_arr, width=11), out_raster=out_raster, reference_raster=in_raster, dst_nodata=None)
# array_to_raster(sum_filter(raster_arr, width=201), out_raster=out_raster, reference_raster=in_raster, dst_nodata=None)
print(time() - before2)

# import pdb; pdb.set_trace()
# array_to_raster(variance, out_raster=out_raster, reference_raster=in_raster, dst_nodata=None)
# array_to_raster(median, out_raster=out_raster, reference_raster=in_raster, dst_nodata=None)
