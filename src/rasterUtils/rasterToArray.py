from osgeo import gdal, ogr, gdalconst
import numpy as np
import numpy.ma as ma
import sys
sys.path.append("..")
from utils.progress import progress_callback, progress_callback_quiet


# Author: CFI
# LastUpdate: 12-02-2019

# TODO: ADD REFERENCE OPTIONS
def rasterToArray(inRaster, reference=None, cutline=None, cutlineAllTouch=False, compressed=False,
                  filled=False, inRasterBand=1, srcNoDataValue=None, dstNoDataValue=None, quiet=False):
    ''' Turns a raster into an Numpy Array in memory.

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
        Array: A Numpy array representing the pixels in the dataset.
            A masked numpy array is returned if nodata is present in inRaster.

    Raises:
        AttributeError: If inRaster is invalid or unreadable by GDAL.
        RuntimeError: If errors are encountered during warping.
    '''

    # Read the supplied raster
    inputDataframe = gdal.Open(inRaster)

    # Throw error if GDAL cannot open the raster
    if inputDataframe is None:
        raise AttributeError(f"Unable to parse the input raster: {inRaster}")

    # Read the requested band. The NoDataValue might be: None.
    inputBand = inputDataframe.GetRasterBand(inRasterBand)

    # Get the nodata-values from either the raster or the function parameters
    inputNodataValue = inputBand.GetNoDataValue()
    if inputNodataValue is None and srcNoDataValue is not None:
        inputNodataValue = srcNoDataValue

    # Bare options to pass to GDAL.Warp. Since output is memory no compression
    # options are passed.
    options = []
    if cutlineAllTouch is True:
        options.append('CUTLINE_ALL_TOUCHED=TRUE')

    # If a cutline is requested, a new DataFrame with the cut raster is created
    # in memory
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
            print(f"Running raster to array:")
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
    else:
        # Data is simply the inputBand as a numpy array.
        data = ma.array(inputBand.ReadAsArray())

    # Close datasets again to free memory.
    inputDataframe = None
    destinationDataframe = None
    subsetDataframe = None

    # Nodata-value handling
    if inputNodataValue is None and dstNoDataValue is not None:
        data = ma.masked_equal(dstNoDataValue)

    data = ma.masked_equal(inputNodataValue)
    if dstNoDataValue:
        ma.set_fill_value(data, dstNoDataValue)

    if filled is True:
        data = data.filled()
    if compressed is True:
        data = data.compressed()

    return data
