"""
### Rasterize vectors. ###

Functions to rasterize vectors to rasters.

TODO:
    * Add support for projections
"""

# Standard library
import sys; sys.path.append("../../")

# External
from osgeo import gdal

# Internal
from buteo.utils import gdal_utils, gdal_enums
from buteo.vector import core_vector
from buteo.raster import core_raster



def rasterize_vector(
    vector,
    pixel_size,
    out_path=None,
    *,
    extent=None,
    all_touch=False,
    dtype="uint8",
    optim="raster",
    band=1,
    fill_value=0,
    nodata_value=None,
    check_memory=True,
    burn_value=1,
    attribute=None,
):
    """
    Rasterize a vector to a raster.

    ## Args:
    `vector` (_str_/_ogr.DataSource): The vector to rasterize. </br>
    `pixel_size` (_float_/_int_): The pixel size of the raster. </br>

    ## Kwargs:
    `out_path` (_str_/_None_): Path to output raster. (Default: **None**) </br>
    `extent` (_list_/_None_): Extent of raster. (Default: **None**) </br>
    `all_touch` (_bool_): All touch? (Default: **False**) </br>
    `dtype` (_str_): Data type of raster. (Default: **"uint8"**) </br>
    `optim` (_str_): Optimization for raster or vector? (Default: **"raster"**) </br>
    `band` (_int_): Band to rasterize. (Default: **1**) </br>
    `fill_value` (_int_/_float_): Fill value. (Default: **0**) </br>
    `nodata_value` (_int_/_float_/_None_): Nodata value. (Default: **None**) </br>
    `check_memory` (_bool_): Check memory? (Default: **True**) </br>
    `burn_value` (_int_/_float_): Value to burn. (Default: **1**) </br>
    `attribute` (_str_/_None_): Attribute to burn. (Default: **None**)

    ## Returns:
    (_str_): Path to output raster.
    """
    vector_fn = vector

    if out_path is None:
        out_path = gdal_utils.create_memory_path(
            gdal_utils.get_path_from_dataset(vector),
            add_uuid=True,
            suffix="_rasterized",
        )

    # Open the data source and read in the extent
    source_ds = core_vector._open_vector(vector_fn)
    source_meta = core_vector._vector_to_metadata(vector_fn)
    source_layer = source_ds.GetLayer()
    x_min, x_max, y_min, y_max = source_layer.GetExtent()

    if isinstance(pixel_size, (gdal.Dataset, str)):
        pixel_size_x = core_raster._open_raster(pixel_size).GetGeoTransform()[1]
        pixel_size_y = abs(core_raster._open_raster(pixel_size).GetGeoTransform()[5])
    elif isinstance(pixel_size, (int, float)):
        pixel_size_x = pixel_size
        pixel_size_y = pixel_size
    elif isinstance(pixel_size, (list, tuple)):
        pixel_size_x, pixel_size_y = pixel_size
    else:
        raise Exception("pixel_size must be either a gdal.Dataset or a tuple of (x, y)")

    # Create the destination data source
    x_res = int((x_max - x_min) / pixel_size_x)
    y_res = int((y_max - y_min) / pixel_size_y)

    if extent is not None:
        extent_vector = core_vector._vector_to_metadata(extent)
        extent_dict = extent_vector["extent_dict"]
        x_res = int((extent_dict["right"] - extent_dict["left"]) / pixel_size_x)
        y_res = int((extent_dict["top"] - extent_dict["bottom"]) / pixel_size_y)
        x_min = extent_dict["left"]
        y_max = extent_dict["top"]

    if check_memory is False:
        gdal.SetConfigOption("CHECK_DISK_FREE_SPACE", "FALSE")

    try:
        target_ds = gdal.GetDriverByName("GTiff").Create(
            out_path,
            x_res,
            y_res,
            1,
            gdal_enums.translate_str_to_gdal_dtype(dtype),
        )
    finally:
        gdal.SetConfigOption("CHECK_DISK_FREE_SPACE", "TRUE")

    if target_ds is None:
        raise Exception("Unable to rasterize.")

    target_ds.SetGeoTransform((x_min, pixel_size_x, 0, y_max, 0, -pixel_size_y))
    target_ds.SetProjection(source_meta["projection_wkt"])

    band = target_ds.GetRasterBand(1)

    if nodata_value is not None:
        band.SetNoDataValue(nodata_value)
    else:
        band.Fill(fill_value)

    options = []
    if all_touch is True:
        options.append("ALL_TOUCHED=TRUE")
    else:
        options.append("ALL_TOUCHED=FALSE")

    if optim == "raster":
        options.append("OPTIM=RASTER")
    elif optim == "vector":
        options.append("OPTIM=VECTOR")
    else:
        options.append("OPTIM=AUTO")

    if attribute is None:
        gdal.RasterizeLayer(
            target_ds,
            [1],
            source_layer,
            burn_values=[burn_value],
            options=options,
        )
    else:
        options.append(f"ATTRIBUTE={attribute}")
        gdal.RasterizeLayer(
            target_ds, [1], source_layer, options=gdal_utils.default_creation_options(options)
        )

    return out_path
