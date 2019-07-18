import sys
import os
import numpy as np
from glob import glob
from raster_io import raster_to_array, array_to_raster
from clip_raster import clip_raster


def layers_math(arr_of_rasters, out_raster, method='median'):
    '''
        Calculate descriptive measures across several rasters. Must be the same size.
    '''

    # Memory intens way of doing it. Could refactor to low RAM method.
    if isinstance(arr_of_rasters[0], np.ndarray):
        images_array = np.array(list(map(lambda x: x, arr_of_rasters)))
    else:
        images_array = np.array(list(map(lambda x: raster_to_array(x), arr_of_rasters)))

    if method is 'median':
        return array_to_raster(np.median(images_array, axis=0), out_raster=out_raster, reference_raster=images_array[0])
    elif method is 'average':
        return array_to_raster(np.average(images_array, axis=0), out_raster=out_raster, reference_raster=images_array[0])
    elif method is 'max':
        return array_to_raster(np.max(images_array, axis=0), out_raster=out_raster, reference_raster=images_array[0])
    elif method is 'min':
        return array_to_raster(np.min(images_array, axis=0), out_raster=out_raster, reference_raster=images_array[0])
