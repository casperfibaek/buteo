from osgeo import gdal, gdalconst
import numpy.ma as ma
import os
from gdalHelpers import datatypeIsFloat, translateMaxValues


def arrayToRaster(array, reference, outRaster=None, outputFormat='MEM'):
    if outRaster is None and outputFormat is not 'MEM':
        raise ValueError('Either outraster or memory output must be selected.')

    referenceDataframe = gdal.Open(reference, gdalconst.GA_ReadOnly)
    referenceTransform = referenceDataframe.GetGeoTransform()
    referenceProjection = referenceDataframe.GetProjection()
    referenceBand = referenceDataframe.GetRasterBand(1)

    if outRaster is None:
        outputFormat = 'MEM'
        outRaster = 'memory'
    elif outRaster is not None and outputFormat is 'MEM':
        outputFormat = 'GTiff'
    elif outRaster is None and outputFormat is not 'MEM':
        outputFormat = 'MEM'
        outRaster = 'memory'

    if outputFormat is 'MEM':
        options = []
    else:
        if datatypeIsFloat(referenceBand.DataType) is True:
            predictor = 3
        else:
            predictor = 2
        options = ['COMPRESS=DEFLATE', f'PREDICTOR={predictor}', 'NUM_THREADS=ALL_CPUS']

    driver = gdal.GetDriverByName(outputFormat)
    destination = driver.Create(outRaster, referenceDataframe.RasterXSize, referenceDataframe.RasterYSize, 1, referenceBand.DataType, options)
    destination.SetGeoTransform(referenceTransform)
    destination.SetProjection(referenceProjection)
    destinationBand = destination.GetRasterBand(1)

    if ma.is_masked(array) is True:
        fillValue = array.get_fill_value()
        print(f'fill value before: {fillValue}')
        if fillValue > translateMaxValues(referenceBand.DataType):
            fillValue = translateMaxValues(referenceBand.DataType)
            print(f'fill value after:  {fillValue}')
        destinationBand.WriteArray(array.filled())
        destinationBand.SetNoDataValue(fillValue)
    else:
        destinationBand.WriteArray(array)

    if outputFormat is not 'MEM':
        destinationBand.FlushCache()
        return os.path.abspath(outRaster)
    else:
        return destination
