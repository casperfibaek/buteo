from osgeo import gdal, ogr, gdalconst
import numpy as np
import numpy.ma as ma
from utils.progress import progress_callback, progress_callback_quiet


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
    
    print(inputTransform)
    exit()


    # Get the nodata-values from either the raster or the function parameters
    inputNodataValue = inputBand.GetNoDataValue()
    if inputNodataValue is None and srcNoDataValue is not None:
        inputNodataValue = srcNoDataValue

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

    # Create a GDAL driver to create dataframes in the right outputFormat
    driver = gdal.GetDriverByName(outputFormat)

    #
    #
    #

    if referenceRaster is not None:

        # Read the reference raster
        referenceDataframe = gdal.Open(referenceRaster)

        # Throw error if GDAL cannot open the raster
        if referenceDataframe is None:
            raise AttributeError(f"Unable to parse the reference raster: {referenceRaster}")

        # Read the attributes of the referenceDataframe
        referenceTransform = referenceDataframe.GetGeoTransform()
        referenceProjection = referenceDataframe.GetProjection()

        progressbar = progress_callback_quiet
        if quiet is False:
            print(f"Reprojection the reference raster:")
            progressbar = progress_callback

        # Reproject the reference to match the input before cutting by extent
        rpReferenceDataframe = gdal.Warp(
            'ignored',
            referenceDataframe,
            format='MEM',
            srcSRS=referenceProjection,
            dstSRC=inputProjection,
            multithread=True,
            callback=progressbar,
        )

        destinationName = outRaster if outRaster is not None else 'ignored'

        # Create the raster to serve as the destination for the warp
        destinationDataframe = driver.Create(
            destinationName,                # Location of the saved raster, Unused if driver is memory.
            inputDataframe.RasterXSize,     # Dataframe width in pixels (e.g. 1920px).
            inputDataframe.RasterYSize,     # Dataframe height in pixels (e.g. 1280px).
            inputBandCount,                 # The number of bands required.
            inputBand.DataType,             # Datatype of the destination
        )
        destinationDataframe.SetGeoTransform(inputTransform)
        destinationDataframe.SetProjection(inputProjection)

        progressbar = progress_callback_quiet
        if quiet is False:
            print(f"Warping the raster:")
            progressbar = progress_callback

        ''' GDAL throws a warning whenever warpOptions are based to a function
            that has the 'MEM' format. However, it is necessary to do so because
            of the cutlineAllTouch feature.
        '''
        gdal.PushErrorHandler('CPLQuietErrorHandler')

        try:
            if cutline is None:
                warped = gdal.Warp(
                    destinationDataframe,
                    origin,
                    format='MEM',
                    srcSRS=inputProjection,
                    dstSRC=referenceProjection,
                    multithread=True,
                    callback=progressbar,
                )
            else:
                warped = gdal.Warp(
                    destinationDataframe,
                    origin,
                    format='MEM',
                    cutlineDSName=cutline,
                    cropToCutline=True,
                    multithread=True,
                    targetAlignedPixels=True,
                    xRes=inputTransform[1],
                    yRes=inputTransform[5],
                    warpOptions=options,
                    callback=progressbar,
                )
        except:
            raise RuntimeError("Error while Warping.") from None


    # If a cutline is requested, a new DataFrame with the cut raster is created in memory
    elif cutline is not None:

        # Read the attributes of the inputdataframe
        inputTransform = inputDataframe.GetGeoTransform()
        inputProjection = inputDataframe.GetProjection()
        # Ensure compatability with multiband rasters
        inputBandCount = inputDataframe.RasterCount

        # Create a GDAL driver to enable the creation of rasters in memory.
        memoryDriver = gdal.GetDriverByName('MEM')

        # Create the raster to serve as the destination for the warp
        destinationDataframe = memoryDriver.Create(
            'ignored',                      # Unused as destination is memory.
            inputDataframe.RasterXSize,     # Dataframe width in pixels (e.g. 1920px).
            inputDataframe.RasterYSize,     # Dataframe height in pixels (e.g. 1280px).
            1,                              # The number of bands required.
            inputBand.DataType,             # Datatype of the destination
        )
        destinationDataframe.SetGeoTransform(inputTransform)
        destinationDataframe.SetProjection(inputProjection)

        ''' As the returned array can only hold one band; it is necessary to create
            a dataframe containing only one band from the input raster, should the
            input raster contain more than one band.'''
        if inputBandCount != 1:
            subsetDataframe = memoryDriver.Create(
                'ignored',                      # Unused as destination is memory.
                inputDataframe.RasterXSize,     # Dataframe width in pixels (e.g. 1920px).
                inputDataframe.RasterYSize,     # Dataframe height in pixels (e.g. 1280px).
                1,                              # The number of bands required.
                inputBand.DataType,             # Datatype of the destination
            )
            subsetDataframe.SetGeoTransform(inputTransform)
            subsetDataframe.SetProjection(inputProjection)

            # The new empty band matching the input raster band kwarg(inRasterBand=1)
            subsetBand = subsetDataframe.GetRasterBand(1)

            # Write the requested inputBand to the subset
            subsetDataframe.WriteArray(inputBand.ReadAsArray())

            # Set the origin to a subset band of the input raster
            origin = subsetDataframe
        else:
            # Subsets are not needed as the input only has one band.
            # Origin is then the input.
            origin = inputDataframe

        progressbar = progress_callback_quiet
        if quiet is False:
            print(f"Warping the raster:")
            progressbar = progress_callback

        ''' GDAL throws a warning whenever warpOptions are based to a function
            that has the 'MEM' format. However, it is necessary to do so because
            of the cutlineAllTouch feature.
        '''
        gdal.PushErrorHandler('CPLQuietErrorHandler')
        try:
            warped = gdal.Warp(
                destinationDataframe,
                origin,
                format='MEM',
                cutlineDSName=cutline,
                cropToCutline=True,
                multithread=True,
                targetAlignedPixels=True,
                xRes=inputTransform[1],
                yRes=inputTransform[5],
                warpOptions=options,
                callback=progressbar,
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

        # Create the data array from the destination dataframe
        data = ma.array(destinationDataframe.GetRasterBand(inRasterBand).ReadAsArray())

    # Close datasets again to free memory.
    inputDataframe = None
    destinationDataframe = None
    subsetDataframe = None
