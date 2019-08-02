import numpy as np
from skimage.util.shape import view_as_windows
from skimage.util import pad
from osgeo import gdal
from raster_io import raster_to_array, array_to_raster


def local_statistics2(in_raster, stat='median', out_raster=None, radius=1, dst_nodata=False):
    '''
        Calculates the local standard deviation in a given radius. Output is a dateframe.

    '''

    if isinstance(in_raster, gdal.Dataset):
        raster_arr = in_raster
    else:
        raster_arr = raster_to_array(in_raster)

    padded = np.pad(raster_arr, radius, 'edge')

    if radius is 1:
        window = view_as_windows(padded, 3, 1)
    else:
        if radius % 2 is 0:
            rad = 3 + radius
        else:
            rad = 3 + radius + 1

        window = view_as_windows(padded, rad, 1)

        # TODO: Add circular function

    if stat is 'max':
        stats = np.max(window, axis=(2, 3))
    elif stat is 'median':
        stats = np.median(window, axis=(2, 3))
    elif stat is 'sum':
        stats = np.sum(window, axis=(2, 3))
    elif stat is 'mean':
        stats = np.mean(window, axis=(2, 3))

    if out_raster is None:
        return array_to_raster(stats, reference_raster=in_raster, dst_nodata=dst_nodata)
    else:
        return array_to_raster(stats, reference_raster=in_raster, out_raster=out_raster, dst_nodata=dst_nodata)
