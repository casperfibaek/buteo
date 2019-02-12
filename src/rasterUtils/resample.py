from osgeo import gdal, gdalconst
import numpy.ma as ma
import os
from gdalHelpers import datatypeIsFloat, translateResampleMethod


def resample(inRaster, outRaster=None, byReference=None, byReferenceBandNumber=1,
             targetSize=None, outputFormat='MEM', method='nearest', quiet=False):
    if targetSize is None and byReference is None:
        raise ValueError('Either targetSize or a reference must be provided.')
    if outRaster is None:
        outputFormat = 'MEM'
        outRaster = 'memory'
    elif outputFormat is 'MEM':
        outRaster = 'memory'

    inputDataframe = gdal.Open(inRaster, gdalconst.GA_ReadOnly)
    inputTransform = inputDataframe.GetGeoTransform()
    inputProjection = inputDataframe.GetProjection()
    inputBandCount = inputDataframe.RasterCount
    inputBand = inputDataframe.GetRasterBand(1)
    inputDatatype = inputBand.DataType
    inputNodataValue = inputBand.GetNoDataValue()

    driver = gdal.GetDriverByName(outputFormat)

    if outputFormat is 'MEM':
        options = []
    else:
        if datatypeIsFloat(inputDatatype) is True:
            predictor = 3
        else:
            predictor = 2
        options = ['COMPRESS=DEFLATE', f'PREDICTOR={predictor}', 'NUM_THREADS=ALL_CPUS']

    if byReference is not None:
        referenceDataframe = gdal.Open(byReference, gdalconst.GA_ReadOnly)
        referenceTransform = referenceDataframe.GetGeoTransform()
        referenceProjection = referenceDataframe.GetProjection()
        referenceXSize = referenceDataframe.RasterXSize
        referenceYSize = referenceDataframe.RasterYSize
        referenceBand = referenceDataframe.GetRasterBand(byReferenceBandNumber)
        referenceDatatype = referenceBand.DataType
        referenceNoDataValue = referenceBand.GetNoDataValue()

        destination = driver.Create(outRaster, referenceXSize, referenceYSize, inputBandCount, referenceDatatype, options)
        destination.SetProjection(referenceProjection)
        destination.SetGeoTransform(referenceTransform)
        if referenceNoDataValue is not None:
            destination.SetNoDataValue(referenceNoDataValue)

        if quiet is True:
            gdal.PushErrorHandler('CPLQuietErrorHandler')

        gdal.Warp(
            destination, inRaster,
            format=outputFormat,
            multithread=True,
            targetAlignedPixels=True,
            xRes=referenceTransform[1],
            yRes=referenceTransform[5],
            srcSRS=inputProjection,
            dstSRS=referenceProjection,
            srcNodata=inputNodataValue,
            dstNodata=referenceNoDataValue,
            warpOptions=options,
            resampleAlg=translateResampleMethod(method),
        )

        if quiet is True:
            gdal.PopErrorHandler()

        if outputFormat is 'MEM':
            return destination
        else:
            return os.path.abspath(outRaster)
    else:
        inputPixelWidth = inputTransform[1]
        inputPixelHeight = inputTransform[5]
        inputXSize = inputDataframe.RasterXSize
        inputYSize = inputDataframe.RasterYSize
        xRatio = inputPixelWidth / targetSize[0]
        yRatio = inputPixelHeight / targetSize[1]
        xPixels = abs(round(xRatio * inputXSize))
        yPixels = abs(round(yRatio * inputYSize))

        destination = driver.Create(outRaster, xPixels, yPixels, inputBandCount, inputDatatype, options)
        destination.SetProjection(inputProjection)
        destination.SetGeoTransform([
            inputTransform[0], targetSize[0], inputTransform[2],
            inputTransform[3], inputTransform[4], -targetSize[1],
        ])
        if inputNodataValue is not None:
            destination.SetNoDataValue(inputNodataValue)

        if quiet is True:
            gdal.PushErrorHandler('CPLQuietErrorHandler')

        gdal.Warp(
            destination, inRaster,
            format=outputFormat,
            multithread=True,
            targetAlignedPixels=True,
            xRes=targetSize[0],
            yRes=targetSize[1],
            srcSRS=inputProjection,
            dstSRS=inputProjection,
            srcNodata=inputNodataValue,
            dstNodata=inputNodataValue,
            warpOptions=options,
            resampleAlg=translateResampleMethod(method),
        )

        if quiet is True:
            gdal.PopErrorHandler()

        if outputFormat is 'MEM':
            return destination
        else:
            return os.path.abspath(outRaster)
