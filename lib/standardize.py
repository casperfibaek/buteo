import sys
import numpy as np
from lib.raster_io import raster_to_array, array_to_raster


def standardize(in_raster, out_raster, method='std'):
    '''
        Standardizes a layer using either the std or normalized to 0, 1 using min/max.
    '''
    ras = raster_to_array(in_raster)

    if method is 'std':
        new_ras = np.divide(np.subtract(ras, np.mean(ras)), np.std(ras))
    elif method is 'norm':
        new_ras = np.divide((ras - np.min(ras)), np.ptp(ras))

    return array_to_raster(new_ras, reference_raster=in_raster, out_raster=out_raster)
