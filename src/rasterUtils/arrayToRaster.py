from osgeo import gdal, gdalconst
import numpy.ma as ma
import os
from gdalHelpers import datatypeIsFloat, translateMaxValues


# If nodata value is above max value, reset to max value.
def arrayToRaster(array, referenceRaster, outRaster=None, outputFormat='MEM'):
    if outRaster is None and outputFormat != 'MEM':
        print('WARNING: No output raster specified. Output changed to memory')
        outputFormat = 'MEM'

    referenceDataframe = gdal.Open(referenceRaster, gdalconst.GA_ReadOnly)
    referenceTransform = referenceDataframe.GetGeoTransform()
    referenceProjection = referenceDataframe.GetProjection()
    referenceBand = referenceDataframe.GetRasterBand(1)

    if outRaster is None:
        outputFormat = 'MEM'
        outRaster = 'memory'
    elif outRaster is None & outputFormat == 'MEM':
        outputFormat = 'GTiff'
    elif outRaster is None and outputFormat != 'MEM':
        outputFormat = 'MEM'
        outRaster = 'memory'

    if outputFormat == 'MEM':
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
        if fillValue > translateMaxValues(referenceBand.DataType):
            fillValue = translateMaxValues(referenceBand.DataType)
        destinationBand.WriteArray(array.filled())
        destinationBand.SetNoDataValue(fillValue)
    else:
        destinationBand.WriteArray(array)

    if outputFormat != 'MEM':
        destinationBand.FlushCache()
        return os.path.abspath(outRaster)
    else:
        return destination
