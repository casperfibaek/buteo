"""
### Vectorize rasters. ###

Module to turn rasters into vector representations.
"""

# Standard library
import sys; sys.path.append("../../")
from typing import Union, Optional, List

# External
from osgeo import gdal, ogr

# Internal
from buteo.utils import (
    utils_base,
    utils_gdal,
    utils_path,
    utils_io
)
from buteo.raster import core_raster



def _raster_vectorize(
    raster: Union[str, gdal.Dataset],
    out_path: Optional[str] = None,
    band: int = 1,
):
    """ Internal. """
    meta = core_raster._get_basic_metadata_raster(raster)
    opened = core_raster._raster_open(raster)
    src_band = opened.GetRasterBand(band)

    projection = meta["projection_osr"]

    if out_path is None:
        out_path = utils_path._get_temp_filepath("vectorized_raster.gpkg")
    else:
        if not utils_path._check_is_valid_output_filepath(out_path, overwrite=True):
            raise ValueError(f"out_path is not a valid output path: {out_path}")

    utils_path._delete_if_required(out_path, overwrite=True)

    driver_name = utils_gdal._get_vector_driver_name_from_path(out_path)
    driver = ogr.GetDriverByName(driver_name)

    datasource = driver.CreateDataSource(out_path)
    layer = datasource.CreateLayer(out_path, srs=projection)

    try:
        gdal.Polygonize(src_band, None, layer, 0)
    except:
        raise RuntimeError(f"Error while vectorizing raster. {raster}") from None
    finally:
        datasource.FlushCache()
        src_band = None
        datasource = None

    return out_path


def raster_vectorize(
    raster: Union[str, gdal.Dataset, List],
    out_path: Optional[str] = None,
    band: int = 1,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    add_timestamp: bool = False,
    overwrite: bool = True,
) -> Union[str, List[str]]:
    """
    Vectorizes a raster by turning it into polygons per unique value. Works
    best on integer rasters.

    Parameters
    ----------
    raster : Union[str, gdal.Dataset, List]
        The raster(s) to vectorize.

    out_path : Optional[str], optional
        The path(s) to save the vectorized raster(s) to. Default: None
    
    band : int, optional
        The band to vectorize. Default: 1

    prefix : str, optional
        The prefix to be added to the output raster name. Default: ""

    suffix : str, optional
        The suffix to be added to the output raster name. Default: ""

    add_uuid : bool, optional
        If True, a unique identifier will be added to the output raster name. Default: False

    add_timestamp : bool, optional
        If True, a timestamp will be added to the output raster name. Default: False

    overwrite : bool, optional
        If True, the output raster will be overwritten if it already exists. Default: True

    Returns
    -------
    Union[str, List]
        The path(s) to the vectorized raster(s).
    """
    utils_base._type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    utils_base._type_check(out_path, [str, [str], None], "out_path")
    utils_base._type_check(band, [int], "band")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(add_timestamp, [bool], "add_timestamp")
    utils_base._type_check(overwrite, [bool], "overwrite")

    input_is_list = isinstance(raster, list)

    input_list = utils_io._get_input_paths(raster, "raster")
    output_list = utils_io._get_output_paths(
        input_list,
        out_path,
        overwrite=overwrite,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        add_timestamp=add_timestamp,
        change_ext="gpkg",
    )

    utils_path._delete_if_required_list(output_list, overwrite)

    vectorized_rasters = []
    for idx, in_raster in enumerate(input_list):
        vectorized_rasters.append(
            _raster_vectorize(
                in_raster,
                out_path=output_list[idx],
                band=band,
            )
        )

    if input_is_list:
        return vectorized_rasters

    return vectorized_rasters[0]
