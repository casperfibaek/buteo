import numpy as np
import numpy.ma as ma
from osgeo import gdal
from clip_raster import clip_raster
from utils import numpy_fill_values


def raster_to_array(in_raster, reference_raster=None, cutline=None, cutline_all_touch=False, cutlineWhere=None,
                    crop_to_cutline=True, compressed=False, band_to_clip=1, src_nodata=None,
                    filled=False, fill_value=None, quiet=False, calc_band_stats=True, align=True):
    ''' Turns a raster into an Numpy Array in memory. Only
        supports for one band to be turned in to an array.

    Args:
        in_raster (URL or GDAL.DataFrame): The raster to clip.

    **kwargs:
        reference_raster (URL or GDAL.DataFrame): A reference
        raster from where to clip the extent of the in_raster.

        cutline (URL or OGR.DataFrame): A geometry used to cut
        the in_raster.

        cutline_all_touch (Bool): Should all pixels that touch
        the cutline be included? False is only pixel centroids
        that fall within the geometry.

        crop_to_cutline (Bool): Should the output raster be
        clipped to the extent of the cutline geometry.

        compressed (Bool): Should the returned data be flattened
        to 1D? If a masked array is compressed, nodata-values
        will be removed from the return array.

        filled (Bool): Should they array be filled with the
        nodata value contained in the mask.

        band_to_clip (Number): A specific band in the input raster
        the be turned to an array.

        src_nodata (Number): Overwrite the nodata value of
        the source raster.

        fill_value (Number): Set a number for the fill value of
        the output masked array. Defaults to the max value for
        the datatype.

        quiet (Bool): Do not show the progressbars.

        align (Bool): Align the output pixels with the pixels
        in the input.

        output_format (String): Output of calculation. MEM is
        default, if outRaster is specified but MEM is selected,
        GTiff is used as output_format.

    Returns:
        Returns a numpy masked array with the raster data.
    '''

    # Verify inputs
    if not isinstance(band_to_clip, int):
        raise AttributeError("band_to_clip must be provided and it must be an integer")

    # If there is no reference_raster or Cutline defined it is
    # not necesarry to clip the raster.
    if reference_raster is None and cutline is None:
        if isinstance(in_raster, gdal.Dataset):
            readiedRaster = in_raster
        else:
            readiedRaster = gdal.Open(in_raster)

        if readiedRaster is None:
            raise AttributeError(f"Unable to parse the input raster: {in_raster}")

    else:
        readiedRaster = clip_raster(
            in_raster,
            reference_raster=reference_raster,
            cutline=cutline,
            cutline_all_touch=cutline_all_touch,
            cutlineWhere=None,
            crop_to_cutline=crop_to_cutline,
            src_nodata=src_nodata,
            quiet=quiet,
            align=align,
            band_to_clip=band_to_clip,
            output_format='MEM',
            calc_band_stats=calc_band_stats,
        )

    if readiedRaster is False:
        return False

    # Read the in_raster as an array
    rasterBand = readiedRaster.GetRasterBand(band_to_clip)
    rasterNoDataValue = rasterBand.GetNoDataValue()
    rasterAsArray = rasterBand.ReadAsArray()

    # Create a numpy masked array that corresponds to the nodata
    # values in the in_raster
    if rasterNoDataValue is None:
        data = ma.array(rasterAsArray, fill_value=fill_value)
    else:
        data = ma.masked_equal(rasterAsArray, rasterNoDataValue)

    if fill_value is not None:
        ma.set_fill_value(data, fill_value)
    else:
        ma.set_fill_value(data, numpy_fill_values(rasterAsArray.dtype))

    # Free memory
    rasterAsArray = None
    rasterBand = None
    readiedRaster = None

    if filled is True:
        data = data.filled()

    if compressed is True:
        data = data.compressed()

    return data
