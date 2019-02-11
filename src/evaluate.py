import gdal
import numpy as np
import numpy.ma as ma


def evaluate(geometry, inRaster, outRaster='./evaluated.tif', inMemory=True, nodata=None):
    inRasterDataframe = gdal.Open(inRaster)
    transform = inRasterDataframe.GetGeoTransform()

    cutlined = gdal.Warp(
        outRaster,
        inRasterDataframe,
        format='MEM',
        cutlineDSName=geometry,
        cropToCutline=True,
        multithread=True,
        targetAlignedPixels=True,
        xRes=transform[1],
        yRes=transform[5],
        warpOptions=['CUTLINE_ALL_TOUCHED=TRUE']
    )

    inRasterDataframe = None
    band = cutlined.GetRasterBand(1)

    bandNDV = band.GetNoDataValue()
    data = band.ReadAsArray()

    if bandNDV is not None:
        ndv = bandNDV
    elif nodata is not None:
        ndv = nodata
    else:
        ndv = None

    mdata = ma.masked_where(data == ndv, data)

    median = np.ma.median(mdata)
    deviations = np.ma.abs(median - mdata)
    madstd = np.ma.median(deviations) * 1.4826

    zscores = np.ma.divide((mdata - median), madstd)

    if inMemory is True:
        return array2raster(zscores, cutlined, inMemory=True, nodata=ndv)
    else:
        return array2raster(zscores, cutlined, outRaster, inMemory=False, nodata=ndv)

    return zscores
