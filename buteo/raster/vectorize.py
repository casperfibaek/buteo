"""
### Vectorize rasters. ###

Module to turn rasters into vector representations.
"""

# Standard library
import sys; sys.path.append("../../")

# External
from osgeo import gdal

# Internal
from buteo.utils import utils_base, utils_gdal
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
        out_path = utils_gdal.create_memory_path(
            out_path,
            suffix="_vectorized",
            add_uuid=True,
        )

    driver = utils_gdal._get_vector_driver_from_path(out_path)

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
    utils_base.type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    utils_base.type_check(out_path, [str, [str], None], "out_path")
    utils_base.type_check(band, [int], "band")
    utils_base.type_check(prefix, [str], "prefix")
    utils_base.type_check(suffix, [str], "suffix")
    utils_base.type_check(add_uuid, [bool], "add_uuid")
    utils_base.type_check(overwrite, [bool], "overwrite")

    raster_list = utils_base._get_variable_as_list(raster)
    assert utils_gdal._check_is_raster_list(raster_list), f"Invalid raster in raster list: {raster_list}"

    path_list = utils_gdal.create_output_path_list(
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
