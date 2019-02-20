from osgeo import gdal
import numpy.ma as ma
import os
from helpers import datatypeIsFloat, translateResampleMethod, copyDataframe
from progress import progress_callback, progress_callback_quiet


def resample(inRaster, outRaster=None, referenceRaster=None, referenceBandNumber=1,
             targetSize=None, outputFormat='MEM', method='nearest', quiet=False):
    ''' Resample an input raster to match a reference raster or
        a target size. The target size is in the same unit as
        the input raster. If the unit is wgs84 - beware that
        the target should be in degrees.

    Args:
        inRaster (URL or GDAL.DataFrame): The raster to clip.

    **kwargs:
        outRaster (URL): The name of the output raster. Only
        used when output format is not memory.

        referenceRaster (URL or GDAL.DataFrame): A reference
        raster from where to clip the extent of the inRaster.

        referenceBandNumber (Integer): The number of the band in
        the reference to use as a target.

        targetSize (List): The target pixel size of the destination.
        e.g. [10, 10] for 10m resolution if the map unit is meters.

        outputFormat (String): Output of calculation. MEM is
        default, if outRaster is specified but MEM is selected,
        GTiff is used as outputformat.

        method (String): The method to use for resampling. By default
        nearest pixel is used. Supports bilinear, cubic and so on.
        For more details look at the GDAL reference.

        quiet (Bool): Do not show the progressbars for warping.

    Returns:
        If the output format is memory outpus a GDAL dataframe
        containing the resampled raster. Otherwise a
        raster is created and the return value is the URL string
        pointing to the created raster.
    '''

    if targetSize is None and referenceRaster is None:
        raise ValueError('Either targetSize or a reference must be provided.')

    if outRaster is not None and outputFormat == 'MEM':
        outputFormat = 'GTiff'

    if outRaster is None and outputFormat != 'MEM':
        raise AttributeError("Either a reference raster or a cutline must be provided.")

    if outRaster is None:
        outRaster = 'ignored'

    if isinstance(inRaster, gdal.Dataset):
        inputDataframe = inRaster
    else:
        inputDataframe = gdal.Open(inRaster)

    # Throw error if GDAL cannot open the raster
    if inputDataframe is None:
        raise AttributeError(f"Unable to parse the input raster: {inRaster}")

    driver = gdal.GetDriverByName(outputFormat)

    inputTransform = inputDataframe.GetGeoTransform()
    inputPixelWidth = inputTransform[1]
    inputPixelHeight = inputTransform[5]
    inputProjection = inputDataframe.GetProjection()
    inputBandCount = inputDataframe.RasterCount
    inputBand = inputDataframe.GetRasterBand(1)
    inputDatatype = inputBand.DataType
    inputNodataValue = inputBand.GetNoDataValue()

    if outputFormat == 'MEM':
        options = []
    else:
        if datatypeIsFloat(inputDatatype) is True:
            predictor = 3
        else:
            predictor = 2
        options = ['COMPRESS=DEFLATE', f'PREDICTOR={predictor}', 'NUM_THREADS=ALL_CPUS']

    # Test if the same size is requested.
    if targetSize is not None:
        if abs(inputPixelWidth) == abs(targetSize[0]) and abs(inputPixelHeight) == abs(targetSize[1]):
            copy = copyDataframe(inputDataframe, outRaster, outputFormat)
            copy.FlushCache()

            if outputFormat == 'MEM':
                return copy
            else:
                return os.path.abspath(outRaster)

    if referenceRaster is not None:
        if isinstance(referenceRaster, gdal.Dataset):
            referenceDataframe = referenceRaster
        else:
            referenceDataframe = gdal.Open(referenceRaster)

        # Throw error if GDAL cannot open the raster
        if referenceDataframe is None:
            raise AttributeError(f"Unable to parse the reference raster: {referenceRaster}")

        referenceTransform = referenceDataframe.GetGeoTransform()
        referenceProjection = referenceDataframe.GetProjection()
        referenceXSize = referenceDataframe.RasterXSize
        referenceYSize = referenceDataframe.RasterYSize
        referencePixelWidth = referenceTransform[1]
        referencePixelHeight = referenceTransform[5]
        referenceBand = referenceDataframe.GetRasterBand(referenceBandNumber)
        referenceDatatype = referenceBand.DataType

        # Test if the reference size and the input size are the same
        if abs(inputPixelWidth) == abs(referencePixelWidth) and abs(inputPixelHeight) == abs(referencePixelHeight):
            copy = copyDataframe(inputDataframe, outRaster, outputFormat)
            copy.FlushCache()

            if outputFormat == 'MEM':
                return copy
            else:
                return os.path.abspath(outRaster)
        else:
            destination = driver.Create(outRaster, referenceXSize, referenceYSize, inputBandCount, referenceDatatype, options)
            destination.SetProjection(referenceProjection)
            destination.SetGeoTransform(referenceTransform)

        gdal.PushErrorHandler('CPLQuietErrorHandler')

        progressbar = progress_callback_quiet
        if quiet is False:
            print(f"Resampling raster:")
            progressbar = progress_callback

        try:
            warpSuccess = gdal.Warp(
                destination, inRaster,
                format=outputFormat,
                multithread=True,
                targetAlignedPixels=True,
                xRes=referenceTransform[1],
                yRes=referenceTransform[5],
                srcSRS=inputProjection,
                dstSRS=referenceProjection,
                srcNodata=inputNodataValue,
                dstNodata=inputNodataValue,
                warpOptions=options,
                resampleAlg=translateResampleMethod(method),
                callback=progressbar,
            )
        except:
            raise RuntimeError("Error while Warping.") from None

        gdal.PopErrorHandler()

        # Check if warped was successfull.
        if warpSuccess == 0:         # GDAL returns 0 for warnings.
            print('Warping completed with warnings. Check your result.')
        elif warpSuccess is None:    # GDAL returns None for errors.
            raise RuntimeError("Warping completed unsuccesfully.") from None

        if outputFormat == 'MEM':
            return destination
        else:
            return os.path.abspath(outRaster)
    else:

        inputXSize = inputDataframe.RasterXSize
        inputYSize = inputDataframe.RasterYSize
        xRatio = inputPixelWidth / targetSize[0]
        yRatio = inputPixelHeight / targetSize[1]
        print(inputPixelWidth, targetSize[0])
        print(inputPixelHeight, targetSize[1])
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

        gdal.PushErrorHandler('CPLQuietErrorHandler')

        progressbar = progress_callback_quiet
        if quiet is False:
            print(f"Resampling raster:")
            progressbar = progress_callback

        try:
            warpSuccess = gdal.Warp(
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
                callback=progressbar,
            )

        except:
            raise RuntimeError("Error while Warping.") from None

        gdal.PopErrorHandler()

        # Check if warped was successfull.
        if warpSuccess == 0:         # GDAL returns 0 for warnings.
            print('Warping completed with warnings. Check your result.')
        elif warpSuccess is None:    # GDAL returns None for errors.
            raise RuntimeError("Warping completed unsuccesfully.") from None

        if outputFormat == 'MEM':
            return destination
        else:
            return os.path.abspath(outRaster)
