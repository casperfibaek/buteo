import numpy as np
import numpy.ma as ma
from rasterToArray import rasterToArray
from arrayToRaster import arrayToRaster
from resample import resample
from helpers import numpyFillValues


def mask_raster(inRaster, referenceRaster, outRaster=None, nodata=[0, 1, 2, 3, 8, 9, 10, 11]):
    in_arr = rasterToArray(inRaster)
    ref_resampled = resample(referenceRaster, referenceRaster=inRaster)
    ref_arr = rasterToArray(ref_resampled)
    nodata_fields = np.array(nodata)

    mask = np.ma.isin(ref_arr, nodata_fields)
    in_arr.mask = mask

    in_arr.set_fill_value(numpyFillValues(in_arr.dtype.name))

    arrayToRaster(in_arr, referenceRaster=referenceRaster, outRaster=outRaster)
