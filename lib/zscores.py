from array_to_raster import array_to_raster
from raster_to_array import raster_to_array
from raster_stats import raster_stats
import numpy as np


def calc_zscores(layer, out_raster=None, vector=None, mad=False):
    '''
        Calculates the Z-scores either around the mean or the mad (using the median).
    '''
    if vector is None:
        if mad is False:
            org_stats = raster_stats(layer, statistics=['mean', 'std'])
        else:
            org_stats = raster_stats(layer, statistics=['med', 'madstd'])
    else:
        if mad is False:
            org_stats = raster_stats(layer, cutline=vector, statistics=['mean', 'std'])
        else:
            org_stats = raster_stats(layer, cutline=vector, statistics=['med', 'madstd'])

    org_arr = raster_to_array(layer)

    if mad is True:
        ret_arr = np.divide(np.subtract(org_arr, org_stats['med']), org_stats['madstd'])
    else:
        ret_arr = np.divide(np.subtract(org_arr, org_stats['mean']), org_stats['std'])

    if out_raster is None:
        return ret_arr
    else:
        array_to_raster(ret_arr, reference_raster=layer, out_raster=out_raster)
