import numpy as np
from lib.raster_io import raster_to_array, array_to_raster


def sigma_to_db(arr):
    return 10 * np.log10(np.abs(arr))
