from osgeo import gdal, ogr, gdalconst
import numpy as np
import numpy.ma as ma
import os
from utils.progress import progress_callback, progress_callback_quiet
from rasterUtils.gdalHelpers import getExtent, getIntersection, createClipGeoTransform, datatypeIsFloat


# Author: CFI
# LastUpdate: 12-02-2019

def clipRaster(inRaster, outRaster=None, referenceRaster=None, cutline=None, cutlineAllTouch=False,
               cropToCutline=True, filled=False, inRasterBand=1, srcNoDataValue=None, dstNoDataValue=None,
               quiet=False, compressed=False, align=False, outputFormat='MEM'):
    ''' Clips a raster by either a reference raster, a cutline or both.

    Args:
        inRaster (URL or GDAL.DataFrame): The raster to turn into a Numpy array.

    **kwargs:
        cutline (URL or OGR.DataFrame): A geometry used to cut the inRaster.
        cutlineAllTouch (Bool): Should all pixels that touch be included?
            False is only centroids.
        compressed (Bool): Should the returned data be flattened to 1D?
            If a masked array is compressed, nodata-values will be removed
            from the return array.
        filled (Bool): Should they array be filled with the nodata value
            contained in the mask.
        inRasterBand (Bool): The number of the band in the raster to turn into
            an array.
        srcNoDataValue (Number): Overwrite the nodata value of the source raster.
        dstNoDataValue (Number): Set a new nodata for the output array.
        quiet (Bool): Suppresses GDAL error messages.

    Returns:
        if MEM is selected as outputFormat a GDAL dataframe is returned,
        otherwise a URL reference to the created raster is returned

    Raises:
        AttributeError: If non-memory outputFormat is selected without outRaster specified.
        AttributeError: If inRaster is invalid or unreadable by GDAL.
        AttributeError: If none of either reference or cutline are provided.
        RuntimeError: If errors are encountered during warping.
    '''

    # Is the output format correct?
    if outRaster is None and outputFormat is not 'MEM':
        raise AttributeError("Either a reference raster or a cutline must be provided.")

    # If outRaster is specified, default to GTiff output format
    if outRaster is not None and outputFormat is 'MEM':
        outputFormat = 'GTiff'

    # Are none of either reference raster or cutline provided?
    if referenceRaster is None and cutline is None:
        raise AttributeError("Either a reference raster or a cutline must be provided.")

    # Read the supplied raster
    inputDataframe = gdal.Open(inRaster)

    # Throw error if GDAL cannot open the raster
    if inputDataframe is None:
        raise AttributeError(f"Unable to parse the input raster: {inRaster}")

    # Read the requested band. The NoDataValue might be: None.
    inputBand = inputDataframe.GetRasterBand(inRasterBand)

    # Read the attributes of the inputdataframe
    inputTransform = inputDataframe.GetGeoTransform()
    inputProjection = inputDataframe.GetProjection()
    inputProjectionRef = inputDataframe.GetProjectionRef()
    # Needed to ensure compatability with multiband rasters
    inputBandCount = inputDataframe.RasterCount

    # Get the nodata-values from either the raster or the function parameters
    inputNodataValue = inputBand.GetNoDataValue()
    if inputNodataValue is None and srcNoDataValue is not None:
        inputNodataValue = srcNoDataValue

    # If the destination nodata value is not set - set it to the input nodata value
    if dstNoDataValue is None:
        dstNoDataValue = inputNodataValue

    ''' GDAL throws a warning whenever warpOptions are based to a function
        that has the 'MEM' format. However, it is necessary to do so because
        of the cutlineAllTouch feature.'''
    gdal.PushErrorHandler('CPLQuietErrorHandler')

    # Test the cutline
    if cutline is not None:
        # Test the cutline. This adds a tiny overhead, but is usefull to ensure
        # that the error messages are easy to understand.
        cutlineGeometry = ogr.Open(cutline)

        # Check if cutline was read properly.
        if cutlineGeometry is 0:         # GDAL returns 0 for warnings.
            print('Geometry read with warnings. Check your result.')
        elif cutlineGeometry is None:    # GDAL returns None for errors.
            raise RuntimeError("It was not possible to read the cutline geometry.") from None

        # Free the memory again.
        cutlineGeometry = None

    # Empty options to pass to GDAL.Warp. Since output is memory no compression
    # options are passed.
    options = []
    if cutlineAllTouch is True:
        options.append('CUTLINE_ALL_TOUCHED=TRUE')

    creationOptions = []
    if outputFormat is not 'MEM':
        if datatypeIsFloat(inputBand.DataType) is True:
            predictor = 3
        else:
            predictor = 2
        creationOptions = ['COMPRESS=DEFLATE', f'PREDICTOR={predictor}', 'NUM_THREADS=ALL_CPUS']

    # Create a GDAL driver to create dataframes in the right outputFormat
    driver = gdal.GetDriverByName(outputFormat)

    destinationName = outRaster if outRaster is not None else 'ignored'

    if referenceRaster is not None:

        # Read the reference raster
        referenceDataframe = gdal.Open(referenceRaster)

        # Throw error if GDAL cannot open the raster
        if referenceDataframe is None:
            raise AttributeError(f"Unable to parse the reference raster: {referenceRaster}")

        # Read the attributes of the referenceDataframe
        referenceTransform = referenceDataframe.GetGeoTransform()
        referenceProjection = referenceDataframe.GetProjection()
        inputExtent = getExtent(inputDataframe)

        # If the projections are the same there is no need to reproject.
        if inputProjection == referenceProjection:
            referenceExtent = getExtent(referenceDataframe)
        else:
            progressbar = progress_callback_quiet
            if quiet is False:
                print(f"Reprojecting reference raster:")
                progressbar = progress_callback

            try:
                # Reproject the reference to match the input before cutting by extent
                reprojectedReferenceDataframe = gdal.Warp(
                    'ignored',
                    referenceDataframe,
                    format='MEM',
                    srcSRS=referenceProjection,
                    dstSRS=inputProjection,
                    multithread=True,
                    callback=progressbar,
                )
            except:
                raise RuntimeError("Error while Warping.") from None

            # Check if warped was successfull.
            if reprojectedReferenceDataframe is 0:         # GDAL returns 0 for warnings.
                print('Warping completed with warnings. Check your result.')
            elif reprojectedReferenceDataframe is None:    # GDAL returns None for errors.
                raise RuntimeError("Warping completed unsuccesfully.") from None

            referenceExtent = getExtent(reprojectedReferenceDataframe)

        # Calculate the bounding boxes and test intersection
        intersection = getIntersection(inputExtent, referenceExtent)

        # If they dont intersect, throw error
        if intersection is False:
            raise RuntimeError("The reference raster did not intersec the input raster") from None

        # Calculates the GeoTransform and rastersize from an extent and a geotransform
        clippedTransform = createClipGeoTransform(inputTransform, intersection)

        # Create the raster to serve as the destination for the warp
        destinationDataframe = driver.Create(
            destinationName,                    # Location of the saved raster, ignored if driver is memory.
            clippedTransform['RasterXSize'],    # Dataframe width in pixels (e.g. 1920px).
            clippedTransform['RasterYSize'],    # Dataframe height in pixels (e.g. 1280px).
            inputBandCount,                     # The number of bands required.
            inputBand.DataType,                 # Datatype of the destination
            creationOptions,                    # Compressions options for non-memory output.
        )

        destinationDataframe.SetGeoTransform(clippedTransform['Transform'])
        destinationDataframe.SetProjection(inputProjection)

        if inputNodataValue is not None:
            for i in range(inputBandCount):
                destinationBand = destinationDataframe.GetRasterBand(i + 1)
                destinationBand.SetNoDataValue(dstNoDataValue)
                destinationBand.FlushCache()

        progressbar = progress_callback_quiet
        if quiet is False:
            print(f"Clipping input raster:")
            progressbar = progress_callback

        try:
            if cutline is None:
                warped = gdal.Warp(
                    destinationDataframe,
                    inputDataframe,
                    format=outputFormat,
                    targetAlignedPixels=align,
                    xRes=inputTransform[1],
                    yRes=inputTransform[5],
                    multithread=True,
                    srcNodata=inputNodataValue,
                    dstNodata=dstNoDataValue,
                    callback=progressbar,
                )
            else:
                warped = gdal.Warp(
                    destinationDataframe,
                    inputDataframe,
                    format=outputFormat,
                    targetAlignedPixels=align,
                    xRes=inputTransform[1],
                    yRes=inputTransform[5],
                    multithread=True,
                    srcNodata=inputNodataValue,
                    dstNodata=dstNoDataValue,
                    callback=progressbar,
                    cutlineDSName=cutline,
                    cropToCutline=cropToCutline,
                    warpOptions=options,
                )
        except:
            raise RuntimeError("Error while Warping.") from None

        # Check if warped was successfull.
        if warped is 0:         # GDAL returns 0 for warnings.
            print('Warping completed with warnings. Check your result.')
        elif warped is None:    # GDAL returns None for errors.
            raise RuntimeError("Warping completed unsuccesfully.") from None

    # If a cutline is requested, a new DataFrame with the cut raster is created in memory
    elif cutline is not None:

        # Create the raster to serve as the destination for the warp
        destinationDataframe = driver.Create(
            destinationName,                # Ignored as destination is memory.
            inputDataframe.RasterXSize,     # Dataframe width in pixels (e.g. 1920px).
            inputDataframe.RasterYSize,     # Dataframe height in pixels (e.g. 1280px).
            inputBandCount,                 # The number of bands required.
            inputBand.DataType,             # Datatype of the destination
            creationOptions,                # Compressions options for non-memory output.
        )
        destinationDataframe.SetGeoTransform(inputTransform)
        destinationDataframe.SetProjection(inputProjection)
        if inputNodataValue is not None:
            for i in range(inputBandCount):
                destinationBand = destinationDataframe.GetRasterBand(i + 1)
                destinationBand.SetNoDataValue(dstNoDataValue)
                destinationBand.FlushCache()

        progressbar = progress_callback_quiet
        if quiet is False:
            print(f"Warping the raster:")
            progressbar = progress_callback

        try:
            warped = gdal.Warp(
                destinationDataframe,
                inputDataframe,
                format=outputFormat,
                targetAlignedPixels=align,
                xRes=inputTransform[1],
                yRes=inputTransform[5],
                multithread=True,
                srcNodata=inputNodataValue,
                dstNodata=dstNoDataValue,
                callback=progressbar,
                cutlineDSName=cutline,
                cropToCutline=cropToCutline,
                warpOptions=options,
            )
        except:
            raise RuntimeError("Error while Warping.") from None

        # Check if warped was successfull.
        if warped is 0:         # GDAL returns 0 for warnings.
            print('Warping completed with warnings. Check your result.')
        elif warped is None:    # GDAL returns None for errors.
            raise RuntimeError("Warping completed unsuccesfully.") from None

    # Reenable the normal ErrorHandler.
    gdal.PopErrorHandler()

    # Close datasets again to free memory.
    inputDataframe = None
    destinationDataframe = None
    referenceDataframe = None
    reprojectedReferenceDataframe = None

    if outRaster is not None:
        warped = None
        return os.path.abspath(outRaster)
    else:
        return warped
