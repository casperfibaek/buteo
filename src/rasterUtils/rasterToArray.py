from osgeo import gdal
import numpy as np
import numpy.ma as ma
from clipRaster import clipRaster
from helpers import numpyFillValues


def rasterToArray(inRaster, referenceRaster=None, cutline=None, cutlineAllTouch=False,
                  cropToCutline=True, compressed=False, bandToClip=1, srcNoDataValue=None,
                  filled=False, fillValue=None, quiet=False, calcBandStats=True, align=True):
    ''' Turns a raster into an Numpy Array in memory. Only
        supports for one band to be turned in to an array.

    Args:
        inRaster (URL or GDAL.DataFrame): The raster to clip.

    **kwargs:
        referenceRaster (URL or GDAL.DataFrame): A reference
        raster from where to clip the extent of the inRaster.

        cutline (URL or OGR.DataFrame): A geometry used to cut
        the inRaster.

        cutlineAllTouch (Bool): Should all pixels that touch
        the cutline be included? False is only pixel centroids
        that fall within the geometry.

        cropToCutline (Bool): Should the output raster be
        clipped to the extent of the cutline geometry.

        compressed (Bool): Should the returned data be flattened
        to 1D? If a masked array is compressed, nodata-values
        will be removed from the return array.

        filled (Bool): Should they array be filled with the
        nodata value contained in the mask.

        bandToClip (Number): A specific band in the input raster
        the be turned to an array.

        srcNoDataValue (Number): Overwrite the nodata value of
        the source raster.

        fillValue (Number): Set a number for the fill value of
        the output masked array. Defaults to the max value for
        the datatype.

        quiet (Bool): Do not show the progressbars.

        align (Bool): Align the output pixels with the pixels
        in the input.

        outputFormat (String): Output of calculation. MEM is
        default, if outRaster is specified but MEM is selected,
        GTiff is used as outputformat.

    Returns:
        Returns a numpy masked array with the raster data.
    '''

    # Verify inputs
    if not isinstance(bandToClip, int):
        raise AttributeError("bandToClip must be provided and it must be an integer")

    # if outRaster is None:
    #     outRaster = 'ignored'

    # If there is no referenceRaster or Cutline defined it is
    # not necesarry to clip the raster.
    if referenceRaster is None and cutline is None:
        if isinstance(inRaster, gdal.Dataset):
            readiedRaster = inRaster
        else:
            readiedRaster = gdal.Open(inRaster)

        if readiedRaster is None:
            raise AttributeError(f"Unable to parse the input raster: {inRaster}")

    else:
        readiedRaster = clipRaster(
            inRaster,
            referenceRaster=referenceRaster,
            cutline=cutline,
            cutlineAllTouch=cutlineAllTouch,
            cropToCutline=cropToCutline,
            srcNoDataValue=srcNoDataValue,
            quiet=quiet,
            align=align,
            bandToClip=bandToClip,
            outputFormat='MEM',
            calcBandStats=calcBandStats,
        )

    # Read the inraster as an array
    rasterBand = readiedRaster.GetRasterBand(bandToClip)
    rasterNoDataValue = rasterBand.GetNoDataValue()
    rasterAsArray = rasterBand.ReadAsArray()

    # Create a numpy masked array that corresponds to the nodata
    # values in the inRaster
    if rasterNoDataValue is None:
        data = ma.array(rasterAsArray, fill_value=fillValue)
    else:
        data = ma.masked_equal(rasterAsArray, rasterNoDataValue)

    if fillValue is not None:
        ma.set_fill_value(data, fillValue)
    else:
        ma.set_fill_value(data, numpyFillValues(rasterAsArray.dtype))

    # Free memory
    rasterAsArray = None
    rasterBand = None
    readiedRaster = None

    if filled is True:
        data = data.filled()

    if compressed is True:
        data = data.compressed()

    return data
