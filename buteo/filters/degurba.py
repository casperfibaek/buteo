"""
Generates the different layers of the degurba filter on a population map.

TODO:
    - Finish workflow
    - Improve documentation
    - Circular import error
"""

import sys; sys.path.append("../../") # Path: buteo/filters/degurba.py

import numpy as np

from buteo.raster.io import raster_to_array, array_to_raster, raster_to_metadata
from buteo.raster.resample import resample_raster
from buteo.raster.vectorize import vectorize_raster
# from buteo.vector.zonal_statistics import zonal_statistics
from buteo.filters.convolutions import filter_array


def side_to_radius(side):
    return ((side ** 2) / np.pi) ** 0.5


def degurba_01(raster, out_path):
    """ OBS: raster must be in meters. """

    meta = raster_to_metadata(raster)
    arr = raster_to_array(raster)
    arr = arr.filled(0)

    pixel_width = meta["pixel_width"]
    side = 1000 / pixel_width # 1km per degurba specification
    radius = np.round(side_to_radius(side))

    kernel_width = int(np.round((radius * 2) + 1))

    filtered = filter_array(
        arr,
        (kernel_width, kernel_width, 1),
        normalised=False,
        distance_calc=False,
    )

    array_to_raster(filtered, raster, out_path)
    
    return out_path


def degurba_02(raster, out_path):

    arr = raster_to_array(raster)
    arr = arr.filled(0)

    arr[arr >= 1500] = 1
    arr[arr >= 5000] = 2

    tmp_high_res = array_to_raster(arr, raster)
    resampled = resample_raster(tmp_high_res, 1000, resample_alg="max")
    vectorized = vectorize_raster(resampled, out_path)
    # zones = zonal_statistics(vectorized, None, raster, stats=["sum"])
    return None
    # return zones


# folder = "C:/Users/caspe/Desktop/degurba_tests/"
# raster_path = folder + "population.tif"

# degurba_01(raster_path, folder + "degurba_01.tif")
# degurba_02(folder + "degurba_01.tif", folder + "degurba_02.shp")
