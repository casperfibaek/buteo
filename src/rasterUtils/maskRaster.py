import numpy as np
import numpy.ma as ma
from rasterToArray import rasterToArray
from arrayToRaster import arrayToRaster
from resample import resample
from helpers import numpyFillValues
import os


B5_pan = os.path.abspath('D:\pythonScripts\yellow\\raster\B5_pansharpened.tif')
SLC10 = os.path.abspath('D:\pythonScripts\yellow\\raster\S2B_MSIL2A_20180702T104019_N0208_R008_T32VNJ_20180702T150728.SAFE\GRANULE\L2A_T32VNJ_A006898_20180702T104021\IMG_DATA\R10m\T32VNJ_20180702T104019_SCL_10m.tif')
out_path = os.path.abspath('D:\pythonScripts\yellow\\raster\B5_masked.tif')


def mask_raster(inRaster, referenceRaster, outRaster=None, nodata=[0, 1, 2, 3, 8, 9, 10, 11]):
    in_arr = rasterToArray(inRaster)
    ref_arr = rasterToArray(referenceRaster)
    nodata_fields = np.array(nodata)

    mask = np.ma.isin(ref_arr, nodata_fields)
    in_arr.mask = mask

    in_arr.set_fill_value(numpyFillValues(in_arr.dtype.name))

    arrayToRaster(in_arr, referenceRaster=referenceRaster, outRaster=outRaster)


mask_raster(B5_pan, SLC10, outRaster=out_path)
