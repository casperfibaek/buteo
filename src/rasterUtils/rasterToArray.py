from osgeo import gdal, gdalconst
import numpy as np
import numpy.ma as ma
import utils

# Author: CFI
# LastUpdate: 12-02-2019


def rasterToArray(inRaster, cutline=None, cutlineAllTouch=False, flatten=False, inRasterBand=1,
                  quiet=True):
    ''' Turns a raster into an Numpy Array in memory.

    Args:
        inRaster (URL or GDAL.DataFrame): The raster to turn into a Numpy array.

    **kwargs:
        cutline (URL or OGR.DataFrame): A geometry used to cut the inRaster.
        cutlineAllTouch (Bool): Should all pixels that touch be included?
            False is only centroids.
        flatten (Bool): Should the returned data be flattened to 1D?
            If a masked array is flattened, nodata-values will be removed
            from the return array.
        inRasterBand (Bool): The number of the band in the raster to turn into
            an array.
        quiet (Bool): Suppresses GDAL error messages.

    Returns:
        Array: A Numpy array representing the pixels in the dataset.
            A masked numpy array is returned if nodata is present in inRaster.

    Raises:
        AttributeError: If inRaster is invalid or unreadable by GDAL.
    '''

    # Read the supplied raster
    inputDataframe = gdal.Open(inRaster)

    # Throw error if GDAL cannot open the raster
    if inputDataframe is None:
        raise AttributeError(f"Unable to parse the input raster: {inRaster}")

    # Read the requested band. The NoDataValue might be: None.
    inputBand = inputDataframe.GetRasterBand(inRasterBand)
    inputNodataValue = inputBand.GetNoDataValue()

    # Bare options to pass to GDAL.Warp. Since output is memory no compression
    # options are passed.
    options = []
    if cutlineAllTouch is True:
        options.append('CUTLINE_ALL_TOUCHED=TRUE')

    # If a cutline is requested, a new DataFrame with the cut raster is created
    # in memory
    if cutline is not None:

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

        # Add the nodata-value if one is present in the supplied raster.
        if inputNodataValue is not None:
            destinationDataframe.SetNoDataValue(inputNodataValue)

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

            # Add the nodata-value if one is present in the supplied raster.
            if inputNodataValue is not None:
                subsetDataframe.SetNoDataValue(inputNodataValue)

            # Write the requested inputBand to the subset
            subsetDataframe.WriteArray(inputBand.ReadAsArray())

            # Set the origin to a subset band of the input raster
            origin = subsetDataframe
        else:
            # Subsets are not needed as the input only has one band.
            # Origin is then the input.
            origin = inputDataframe

        ''' GDAL throws a warning whenever warpOptions are based to a function
            that has the 'MEM' format. However, it is necessary to do so because
            of the cutlineAllTouch feature.
        '''
        if quiet is True:
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
                callback=progress_callback,
            )
        except:
            raise RuntimeError("Error while Warping.") from None

        # Check if warped was successfull.
        if warped != 1:
            raise RuntimeError("Warping completed unsuccesfully.") from None

        # Reenable the normal ErrorHandler.
        if quiet is True:
            gdal.PopErrorHandler()

        # Create the data array from the destination dataframe
        data = destinationDataframe.GetRasterBand(inRasterBand).ReadAsArray()
    else:
        # Data is simply the inputBand as a numpy array.
        data = inputBand.ReadAsArray()

    # Close datasets again to free memory.
    inputDataframe = None
    destinationDataframe = None
    subsetDataframe = None

    # Handle nodata
    if inputNodataValue is not None:
        data = ma.masked_equal(data, inputNodataValue)
        ma.set_fill_value(data, inputNodataValue)
        if flatten is True:
            return data.compressed()
    else:
        if flatten is True:
            return data.flatten()
    return data

test = rasterToArray(
    '../raster/S2B_MSIL2A_20180702T104019_N0208_R008_T32VNJ_20180702T150728.SAFE/GRANULE/L2A_T32VNJ_A006898_20180702T104021/IMG_DATA/R10m/T32VNJ_20180702T104019_B02_10m.jp2',
    cutline='../geometry/roses.geojson',
)
