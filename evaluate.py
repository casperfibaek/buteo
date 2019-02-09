from osgeo import gdal
import numpy as np
import numpy.ma as ma

def array2raster(array, copy, outRaster='./newraster.tif', inMemory=False, nodata=None):
    geoTransform = copy.GetGeoTransform()
    band = copy.GetRasterBand(1)
    ndv = band.GetNoDataValue()

    driver = gdal.GetDriverByName('GTiff')

    if gdal.GetDataTypeName(band.DataType) == 'Float32':
        predictor = 3
    else:
        predictor = 2

    options = ['COMPRESS=DEFLATE', f'PREDICTOR={predictor}', 'NUM_THREADS=ALL_CPUS']
    newRaster = driver.Create(outRaster, copy.RasterXSize, copy.RasterYSize, 1, band.DataType, options)
    newRaster.SetGeoTransform(geoTransform)
    newRaster.SetProjection(copy.GetProjection())

    outband = newRaster.GetRasterBand(1)
    if ndv is not None:
        outband.SetNoDataValue(ndv)
    elif nodata is not None:
        outband.SetNoDataValue(nodata)
    outband.WriteArray(array)
    
    if inMemory is False and len(outRaster) > 0:
        outband.FlushCache()
        return newRaster
    else:
        return newRaster


def evaluate(geometry, inRaster, outRaster='./evaluated.tif', inMemory=True, nodata=None):
    inRasterDataframe = gdal.Open(inRaster)
    transform = inRasterDataframe.GetGeoTransform()
    inRasterDataframe = None

    cutlined = gdal.Warp(
        outRaster,
        inRaster,
        format='MEM',
        cutlineDSName=geometry,
        cropToCutline=True,
        multithread=True,
        targetAlignedPixels=True,
        xRes=transform[1],
        yRes=transform[5],
    )

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
