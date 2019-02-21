import numpy as np
import numpy.ma as ma
from raster_to_array import rasterToArray
from array_to_raster import arrayToRaster
from resample import resample
from utils import numpyFillValues


def mask_raster(in_raster, mask_raster=None, out_raster=None, nodata=[0, 1, 9, 11]):
    in_arr = rasterToArray(in_raster)
    ref_resampled = resample(mask_raster, referenceRaster=in_raster)
    ref_arr = rasterToArray(ref_resampled)
    nodata_fields = np.array(nodata)

    mask = np.ma.isin(ref_arr, nodata_fields)
    in_arr.mask = mask

    in_arr.set_fill_value(numpyFillValues(in_arr.dtype.name))

    arrayToRaster(in_arr, referenceRaster=in_raster, outRaster=out_raster)
