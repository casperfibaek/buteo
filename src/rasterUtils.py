from osgeo import gdal, gdalconst
import os


def translateGDALResampleMethod(method):
    methods = {
        'nearest': 0,
        'bilinear': 1,
        'cubic': 2,
        'cubicSpline': 3,
        'lanczos': 4,
        'average': 5,
        'mode': 6,
        'max': 8,
        'min': 9,
        'median': 10,
        'q1': 11,
        'q3': 12,
    }

    if method in methods:
        return methods[method]
    else:
        return 0


def translateGDALDataTypes(datatype):
    datatypes = {
        'byte': 1,
        'uint16': 2,
        'int16': 3,
        'uint32': 4,
        'int32': 5,
        'float32': 6,
        'float64': 7,
        'cint16': 8,
        'cint32': 9,
        'cfloat32': 10,
        'cfloat64': 11,
    }

    if datatype in datatypes:
        return datatypes[datatype]
    else:
        return 6


def isFloatGDAL(datatype):
    floats = [6, 7, 10, 11]
    if datatype in floats:
        return True
    else:
        return False


def isIntegerGDAL(datatype):
    integers = [1, 2, 3, 4, 5, 8, 9]
    if datatype in integers:
        return True
    else:
        return False


def rasterToArray(inRaster, cutline=None, cutlineAllTouch=False, flatten=False, inRasterBand=1,
                  quiet=False, nodata=None):
    options = []
    if cutlineAllTouch is True:
        options.append('CUTLINE_ALL_TOUCHED=TRUE')

    inputDataframe = gdal.Open(inRaster)
    inputBand = inputDataframe.GetRasterBand(inRasterBand)
    inputNodataValue = inputBand.GetNoDataValue()
    if inputNodataValue is None:
        inputNodataValue = nodata

    if cutline is not None:
        inputTransform = inputDataframe.GetGeoTransform()
        inputProjection = inputDataframe.GetProjection()
        inputBandCount = inputDataframe.RasterCount

        destinationDriver = gdal.GetDriverByName('MEM')
        destination = destinationDriver.Create('memory', inputDataframe.RasterXSize, inputDataframe.RasterYSize, 1, inputBand.DataType)
        destination.SetGeoTransform(inputTransform)
        destination.SetProjection(inputProjection)

        if inputBandCount != 1:
            subsetDriver = gdal.GetDriverByName('MEM')
            subset = subsetDriver.Create('memory', inputDataframe.RasterXSize, inputDataframe.RasterYSize, 1, inputBand.DataType)
            subset.SetGeoTransform(inputTransform)
            subset.SetProjection(inputProjection)
            subsetBand = subset.GetRasterBand(1)
            if inputNodataValue is not None:
                subsetBand.SetNoDataValue(inputNodataValue)
            elif nodata is not None:
                subsetBand.SetNoDataValue(nodata)
            subsetBand.WriteArray(inputBand.ReadAsArray())
            origin = subset
        else:
            origin = inputDataframe

        if quiet is True:
            gdal.PushErrorHandler('CPLQuietErrorHandler')

        gdal.Warp(
            destination,
            origin,
            format='MEM',
            cutlineDSName=cutline,
            cropToCutline=True,
            multithread=True,
            targetAlignedPixels=True,
            xRes=inputTransform[1],
            yRes=inputTransform[5],
            warpOptions=options
        )

        if quiet is True:
            gdal.PopErrorHandler()

        data = destination.GetRasterBand(inRasterBand).ReadAsArray()
    else:
        data = inputBand.ReadAsArray()

    if inputNodataValue is not None:
        data = data[data != inputNodataValue]

    if flatten is True:
        return data.flatten()
    else:
        return data


def arrayToRaster(array, reference, outRaster=None, outputFormat='MEM', nodata=None):
    if outRaster is None and outputFormat is not 'MEM':
        raise ValueError('Either outraster or memory output must be selected.')
    referenceDataframe = gdal.Open(reference, gdalconst.GA_ReadOnly)
    referenceTransform = referenceDataframe.GetGeoTransform()
    referenceProjection = referenceDataframe.GetProjection()
    referenceBand = referenceDataframe.GetRasterBand(1)
    referenceNoDataValue = referenceBand.GetNoDataValue()

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
        if gdal.GetDataTypeName(referenceBand.DataType) == 'Float32':
            predictor = 3
        else:
            predictor = 2
        options = ['COMPRESS=DEFLATE', f'PREDICTOR={predictor}', 'NUM_THREADS=ALL_CPUS']

    driver = gdal.GetDriverByName(outputFormat)
    destination = driver.Create(outRaster, referenceDataframe.RasterXSize, referenceDataframe.RasterYSize, 1, referenceBand.DataType, options)
    destination.SetGeoTransform(referenceTransform)
    destination.SetProjection(referenceProjection)
    destinationBand = destination.GetRasterBand(1)
    destinationBand.WriteArray(array)

    if nodata is not None:
        destinationBand.SetNoDataValue(nodata)

    if outputFormat is not 'MEM':
        destinationBand.FlushCache()
        return os.path.abspath(outRaster)
    else:
        return destination


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
        if gdal.GetDataTypeName(inputBand.DataType) == 'Float32':
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
        referenceNoDataValue = referenceBand.GetNoDataValue()

        destination = driver.Create(outRaster, referenceXSize, referenceYSize, inputBandCount, inputDatatype, options)
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
            resampleAlg=translateGDALResampleMethod(method),
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
            resampleAlg=translateGDALResampleMethod(method),
        )

        if quiet is True:
            gdal.PopErrorHandler()

        if outputFormat is 'MEM':
            return destination
        else:
            return os.path.abspath(outRaster)
