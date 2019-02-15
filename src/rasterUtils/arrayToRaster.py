from osgeo import gdal, gdalconst
import numpy.ma as ma
import os
from gdalHelpers import datatypeIsFloat, translateMaxValues


def arrayToRaster(array, referenceRaster=None, outRaster=None, outputFormat='MEM',
                  topLeft=None, pixelSize=None, projection=None, rasterSize=None,
                  srcNoDataValue=None, dstNoDataValue=None, quiet=False):
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

        rasterSize (List): The size of the output in pixels. Such
        as: [5000, 6000] in pixels.

        srcNoDataValue (Number): Overwrite the nodata value of
        the source raster.

        dstNoDataValue (Number): Set a new nodata for the
        output array.

        quiet (Bool): Do not show the progressbars for warping.

    Returns:
        If the output format is memory outpus a GDAL dataframe
        containing the data contained in the array. Otherwise a
        raster is created and the return value is the URL string
        pointing to the created raster.
    '''

    # Is the output format correct?
    if outRaster is None and outputFormat != 'MEM':
        raise AttributeError("Either a reference raster or a cutline must be provided.")

    # If outRaster is specified, default to GTiff output format
    if outRaster is not None and outputFormat == 'MEM':
        outputFormat = 'GTiff'

    if referenceRaster is None and (
        topLeft is None or
        pixelSize is None or
        projection is None or
        rasterSize is None
    ):
        raise AttributeError("If no referenceRaster is provided. topLeft, pixelSize, projection and rasterSize are all required.")

    if referenceRaster is not None and (
        topLeft is not None or
        pixelSize is not None or
        projection is not None or
        rasterSize is not None
    ):
        print('WARNING: Only the values from the referenceRaster will be used.')

    options = []
    if outputFormat != 'MEM':
        if datatypeIsFloat(referenceBand.DataType) is True:
            predictor = 3
        else:
            predictor = 2
        options = ['COMPRESS=DEFLATE', f'PREDICTOR={predictor}', 'NUM_THREADS=ALL_CPUS']

    # If nodata value is above max value, reset to max value.
    # 1) Ready the datasource using the nodata values
    if ma.is_masked(array) is True:
        fillValue = array.get_fill_value()
        data = array.filled()
    else:
        # Handle nodata
        print('handle nodata')

    # Create the destination frame
    referenceDataframe = gdal.Open(referenceRaster, gdalconst.GA_ReadOnly)
    referenceTransform = referenceDataframe.GetGeoTransform()
    referenceProjection = referenceDataframe.GetProjection()
    referenceBand = referenceDataframe.GetRasterBand(1)

    driver = gdal.GetDriverByName(outputFormat)
    destination = driver.Create(outRaster, referenceDataframe.RasterXSize, referenceDataframe.RasterYSize, 1, referenceBand.DataType, options)
    destination.SetGeoTransform(referenceTransform)
    destination.SetProjection(referenceProjection)
    destinationBand = destination.GetRasterBand(1)

    # Figure out if it is necessary to warp the raster
    # Warp and resample if the shape of the input raster does not math the destination raster

    # Fill or warp the destination
    destinationBand.WriteArray(array.filled())
    destinationBand.SetNoDataValue(fillValue)   # OBS on nodata

    # Return the raster
    if outputFormat != 'MEM':
        destinationBand.FlushCache()
        return os.path.abspath(outRaster)
    else:
        return destination
