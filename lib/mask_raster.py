import numpy as np
import numpy.ma as ma
from raster_to_array import raster_to_array
from array_to_raster import array_to_raster
from resample import resample
from utils import numpy_fill_values


def mask_raster(in_raster, mask_raster=None, out_raster=None, nodata=[0, 1, 9, 11]):
    in_arr = raster_to_array(in_raster)
    ref_resampled = resample(mask_raster, referenceRaster=in_raster)
    ref_arr = raster_to_array(ref_resampled)
    nodata_fields = np.array(nodata)

    mask = np.ma.isin(ref_arr, nodata_fields)
    in_arr.mask = mask

    in_arr.set_fill_value(numpy_fill_values(in_arr.dtype.name))

    array_to_raster(in_arr, referenceRaster=in_raster, outRaster=out_raster)
