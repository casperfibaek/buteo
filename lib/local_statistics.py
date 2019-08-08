import numpy as np
import numpy.ma as ma
from math import sqrt
from skimage.util.shape import view_as_windows
from skimage.util import pad
from osgeo import gdal
from raster_io import raster_to_array, array_to_raster
from skimage import filters


def circular_kernel_mask(radius, length, adjusted=False, inverse=False):
    rd = radius + (sqrt(2) / 2) if adjusted is True else radius
    mask = np.empty((length, length), dtype=bool)

    for x in range(length):
        for y in range(length):
            xm = x - radius
            ym = y - radius
            dist = sqrt(pow(xm, 2) + pow(ym, 2))
            if (dist > rd):
                mask[x][y] = 0 if inverse is False else 1
            else:
                mask[x][y] = 1 if inverse is False else 0

    return mask


def local_statistics2(in_raster, stat='median', out_raster=None, radius=1, dst_nodata=False, circular=True, dtype=None):
    '''
        Calculates the local standard deviation in a given radius. Output is a dateframe.

    '''

    if isinstance(in_raster, gdal.Dataset):
        raster_arr = in_raster
    else:
        raster_arr = raster_to_array(in_raster)

    out_dtype = raster_arr.dtype if dtype is None else dtype

    if radius is 1:
        rad = 3
    else:
        if radius % 2 is 0:
            rad = 3 + radius
        else:
            rad = 3 + radius + 1

    if circular is True:
        mask = circular_kernel_mask(radius, rad, adjusted=True)

    stats = filters.rank.sum(raster_arr.astype(out_dtype), selem=mask)
    
    # stats = generic_filter.generic_filter(padded, np.sum, footprint=mask)
    # stats = filters.rank.sum(raster_arr.astype(out_dtype), selem=mask)

    # if stat is 'max':
    #     stats = np.max(window, axis=(2, 3))
    # elif stat is 'median':
    #     stats = np.median(window, axis=(2, 3), overwrite_input=True)
    # elif stat is 'sum':
    #     stats = np.sum(window, axis=(2, 3))
    # elif stat is 'mean':
    #     stats = np.mean(window, axis=(2, 3))

    if out_raster is None:
        return array_to_raster(stats, reference_raster=in_raster, dst_nodata=dst_nodata)
    else:
        return array_to_raster(stats, reference_raster=in_raster, out_raster=out_raster, dst_nodata=dst_nodata)
