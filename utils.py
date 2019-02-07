import rasterio
import rasterio.warp as warp
from rasterio import Affine
import numpy as np
import os
from os.path import normpath, basename

def getFiletype( str ):
    return basename(normpath(str)).split('.')[-1]

def getFilename ( str ):
    return basename(normpath(str)).split('.')[0]

def getPath ( str ):
    return normpath(str)

def threads (hyperthreading = True):
    if hyperthreading == True:
        return os.cpu_count() * 2
    else:
        return os.cpu_count()

def resample(src, rm, method = 'nearest', nodata = None, band = 1):
    """
        Returns the resampled Numpy array and the output metadata
    src     -- Input raster
    rm      -- Multiple of destination resolution 2 = (512 * 2) = 1024px
    method  -- Resampling method from rasterio.warp.Resampling
    nodata  -- The nodata value of the destination raster
    band    -- Raster band index (one-based indexing) (default 1)
    """

    arr = src.read(band).astype(rasterio.uint16)
    if (rm == 1): return arr # If the resolution is the same, don't resample.

    # Set nodata to input
    nodataValue = src.nodata if nodata == None else nodata
    
    newarr = np.empty(shape=(1, round(arr.shape[0] * rm), round(arr.shape[1] * rm)), dtype = arr.dtype)

    dstTransform = Affine(src.transform.a / rm, src.transform.b, src.transform.c,
        src.transform.d, src.transform.e / rm, src.transform.f)

    warp.reproject(arr, newarr,
        src_transform = src.transform,
        dst_transform = dstTransform,
        src_nodata = nodata,
        dst_nodata = nodataValue,
        src_crs = src.crs,
        dst_crs = src.crs,
        num_threads = os.cpu_count() * 2,
        resampling = warp.Resampling[method])

    src.close()

    return newarr