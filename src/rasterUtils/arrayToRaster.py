from osgeo import gdal, osr
import numpy as np
import numpy.ma as ma
import os
from helpers import datatypeIsFloat, numpyToGdalDatatypes
from progress import progress_callback, progress_callback_quiet


def arrayToRaster(array, outRaster=None, referenceRaster=None, outputFormat='MEM',
                  topLeft=None, pixelSize=None, projection=None, calcBandStats=True,
                  srcNoDataValue=None, resample=False, quiet=False):
    ''' Turns a numpy array into a gdal dataframe or exported
        as a raster. If no reference is specified, the following
        must be provided: topLeft coordinates, pixelSize such as:
        (10, 10), projection in proj4 format and output raster size.

        The datatype of the raster will match the datatype of the input
        numpy array.

        OBS: If WGS84 lat long is specified as the projection the pixel
        sizes must be in degrees.

    Args:
        inRaster (URL or GDAL.DataFrame): The raster to clip.

    **kwargs:
        referenceRaster (URL or GDAL.DataFrame): A reference
        raster from where to clip the extent of the inRaster.

        outRaster (URL): The name of the output raster. Only
        used when output format is not memory.

        outputFormat (String): Output of calculation. MEM is
        default, if outRaster is specified but MEM is selected,
        GTiff is used as outputformat.

        topLeft (List): The coordinates for the topleft corner
        of the raster. Such as: [600000, 520000].

        pixelSize (List): The width and height of the pixels in
        the destination raster. Such as: [10, 10].

        projection (String): A proj4 string matching the projection
        of the destination raster. Such as: "+proj=utm +zone=32
            +ellps=WGS84 +datum=WGS84 +units=m +no_defs".

        srcNoDataValue (Number): Overwrite the nodata value of
        the source raster.

        quiet (Bool): Do not show the progressbars for warping.

    Returns:
        If the output format is memory outpus a GDAL dataframe
        containing the data contained in the array. Otherwise a
        raster is created and the return value is the URL string
        pointing to the created raster.
    '''

    ''' **********************************************************
        STEP (1): Verify the input data.
        ********************************************************** '''

    # Is the output format correct?
    if outRaster is None and outputFormat != 'MEM':
        raise AttributeError("Either a reference raster or a cutline must be provided.")

    # If outRaster is specified, default to GTiff output format
    if outRaster is not None and outputFormat == 'MEM':
        outputFormat = 'GTiff'

    if outRaster is None:
        outRaster = 'ignored'

    if referenceRaster is None and (
        topLeft is None or
        pixelSize is None or
        projection is None
    ):
        raise AttributeError("If no referenceRaster is provided. topLeft, pixelSize, projection and rasterSize are all required.")

    if referenceRaster is not None and (
        topLeft is not None or
        pixelSize is not None or
        projection is not None
    ):
        print('WARNING: Only the values from the referenceRaster will be used.')

    ''' **********************************************************
        STEP (2): Setup local values and ready data.
    ********************************************************** '''

    # The data that will be written to the raster
    data = array if isinstance(array, np.ndarray) else np.array(array)
    # data = np.swapaxes(data, 0, 1)

    if data.ndim != 2:
        raise AttributeError("The input raster must be 2-dimensional")

    # Gather reference information
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
        referenceRasterXSize = referenceDataframe.RasterXSize
        referenceRasterYSize = referenceDataframe.RasterYSize
        referencePixelWidth = referenceTransform[1]
        referencePixelHeight = referenceTransform[5]
        referenceTopLeftX = referenceTransform[0]
        referenceTopLeftY = referenceTransform[3]
        referenceBand = referenceDataframe.GetRasterBand(1)
        referenceNoDataValue = referenceBand.GetNoDataValue()
        referenceDataType = numpyToGdalDatatypes(data.dtype)
    else:
        referenceRasterXSize = data.shape[0]
        referenceRasterYSize = data.shape[1]
        referencePixelWidth = pixelSize[0]
        referencePixelHeight = pixelSize[1]
        referenceTopLeftX = topLeft[0]
        referenceTopLeftY = topLeft[1]
        referenceNoDataValue = None,
        referenceDataType = numpyToGdalDatatypes(data.dtype)
        referenceTransform = [
            referenceTopLeftX, referencePixelWidth, 0,
            referenceTopLeftY, 0, -referencePixelHeight,
        ]
        referenceProjection = osr.SpatialReference()
        referenceProjection.ImportFromProj4(projection)
        referenceProjection.ExportToWkt()
        referenceProjection = str(referenceProjection)

    inputNoDataValue = srcNoDataValue if srcNoDataValue is not None else None

    # Ready the nodata values
    if ma.is_masked(data) is True:
        if inputNoDataValue is not None:
            ma.masked_equal(data, inputNoDataValue)
            data.set_fill_value(inputNoDataValue)
        inputNoDataValue = data.get_fill_value()
        data = data.filled()

    # Weird "double" issue with GDAL. Cast to float or int
    inputNoDataValue = float(inputNoDataValue)
    if (inputNoDataValue).is_integer() is True:
        inputNoDataValue = int(inputNoDataValue)

    # If the output is not memory, set compression options.
    options = []
    if outputFormat != 'MEM':
        if datatypeIsFloat(referenceDataType) is True:
            predictor = 3  # Float predictor
        else:
            predictor = 2  # Integer predictor
        options = ['COMPRESS=DEFLATE', f'PREDICTOR={predictor}', 'NUM_THREADS=ALL_CPUS']

    ''' **********************************************************
        STEP (3): The business logic.
    ********************************************************** '''
    if resample is False:
        destinationDriver = gdal.GetDriverByName(outputFormat)
        destinationName = outRaster
        destinationOptions = options
    else:
        destinationName = 'ignored'
        destinationDriver = gdal.GetDriverByName('MEM')
        destinationOptions = []

    destination = destinationDriver.Create(destinationName, data.shape[1], data.shape[0], 1, referenceDataType, destinationOptions)

    # Test if the scale is correct
    if data.shape[0] == referenceRasterXSize and data.shape[1] == referenceRasterYSize:
        destination.SetGeoTransform(referenceTransform)
    else:
        destinationPixelWidth = (referenceRasterXSize / data.shape[1]) * referencePixelWidth
        destinationPixelHeight = (referenceRasterYSize / data.shape[0]) * referencePixelHeight
        destination.SetGeoTransform([
            referenceTopLeftX,
            destinationPixelWidth,
            0,
            referenceTopLeftY,
            0,
            destinationPixelHeight,
        ])

    destination.SetProjection(referenceProjection)
    destinationBand = destination.GetRasterBand(1)
    destinationBand.WriteArray(data)

    if inputNoDataValue is not None:
        destinationBand.SetNoDataValue(inputNoDataValue)

    if resample is True and (data.shape[0] != referenceRasterXSize or data.shape[1] != referenceRasterYSize):
        resampledDestinationDriver = gdal.GetDriverByName(outputFormat)
        resampledDestination = resampledDestinationDriver.Create(outRaster, referenceRasterXSize, referenceRasterYSize, 1, referenceDataType, options)
        resampledDestination.SetGeoTransform(referenceTransform)
        resampledDestination.SetProjection(referenceProjection)

        progressbar = progress_callback_quiet
        if quiet is False:
            print(f"Warping input array:")
            progressbar = progress_callback

        gdal.PushErrorHandler('CPLQuietErrorHandler')

        try:
            warpSuccess = gdal.Warp(
                resampledDestination,
                destination,
                format=outputFormat,
                xRes=referenceRasterXSize,
                yRes=referenceRasterYSize,
                srcSRS=referenceProjection,
                multithread=True,
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

        # Clear memory
        destination = None

        if calcBandStats is True:
            resampledDestination.GetRasterBand(1).GetStatistics(0, 1)

        # Return the resampled destination raster
        if outputFormat != 'MEM':
            resampledDestination.FlushCache()
            resampledDestination = None
            return os.path.abspath(outRaster)
        else:
            return resampledDestination

    if calcBandStats is True:
        destinationBand.GetStatistics(0, 1)

    # Return the destination raster
    if outputFormat != 'MEM':
        destination.FlushCache()
        destination = None
        return os.path.abspath(outRaster)
    else:
        return destination
