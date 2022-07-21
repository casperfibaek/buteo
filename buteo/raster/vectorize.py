"""
### Vectorize rasters. ###

Module to turn rasters into vector representations.
"""

# Standard library
import sys; sys.path.append("../../")

# External
from osgeo import gdal

# Internal
from buteo.utils import core_utils, gdal_utils
from buteo.raster import core_raster



def _vectorize_raster(
    raster,
    *,
    out_path=None,
    band=1,
):
    """ Internal. """
    meta = core_raster._raster_to_metadata(raster)
    opened = core_raster._open_raster(raster)
    src_band = opened.GetRasterBand(band)

    projection = meta["projection_osr"]

    if out_path is None:
        out_path = gdal_utils.create_memory_path(
            out_path,
            suffix="_vectorized",
            add_uuid=True,
        )

    driver = gdal_utils.path_to_driver_vector(out_path)

    datasource = driver.CreateDataSource(out_path)
    layer = datasource.CreateLayer(out_path, srs=projection)

    try:
        gdal.Polygonize(src_band, None, layer, 0)
    except:
        raise RuntimeError(f"Error while vectorizing raster. {raster}") from None

    return out_path


def vectorize_raster(
    raster,
    *,
    out_path=None,
    band=1,
    prefix="",
    suffix="",
    add_uuid=False,
    overwrite=True,
):
    """
    Vectorizes a raster by turning it into polygons per unique value. Works
    best on integer rasters.

    ## Args:
    `raster` (_str_/_list_/_gdal.Dataset_): The raster(s) to vectorize.

    ## Kwargs:
    `out_path` (_str_/_list_/_None_): The path(s) to save the vectorized raster(s) to. (Default: **None**) </br>
    `band` (_int_): The band to vectorize. (Default: **1**) </br>
    `prefix` (_str_): The prefix to be added to the output raster name. (Default: **""**) </br>
    `suffix` (_str_): The suffix to be added to the output raster name. (Default: **""**) </br>
    `add_uuid` (_bool_): If True, a unique identifier will be added to the output raster name. (Default: **False**) </br>

    ## Returns:
    (_str_/_list_): The path(s) to the vectorized raster(s).
    """
    core_utils.type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    core_utils.type_check(out_path, [str, [str], None], "out_path")
    core_utils.type_check(band, [int], "band")
    core_utils.type_check(prefix, [str], "prefix")
    core_utils.type_check(suffix, [str], "suffix")
    core_utils.type_check(add_uuid, [bool], "add_uuid")
    core_utils.type_check(overwrite, [bool], "overwrite")

    raster_list = core_utils.ensure_list(raster)
    assert gdal_utils.is_raster_list(raster_list), f"Invalid raster in raster list: {raster_list}"

    path_list = gdal_utils.create_output_path_list(
        raster_list,
        out_path=out_path,
        overwrite=overwrite,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
    )

    vectorized_rasters = []
    for index, in_raster in enumerate(raster_list):
        vectorized_rasters.append(
            _vectorize_raster(
                in_raster,
                out_path=path_list[index],
                band=band,
            )
        )

    if isinstance(raster, list):
        return vectorized_rasters

    return vectorized_rasters[0]
