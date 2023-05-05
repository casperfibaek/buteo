"""
### Vectorize rasters. ###

Module to turn rasters into vector representations.
"""

# Standard library
import sys; sys.path.append("../../")
from typing import Union, Optional, List

# External
from osgeo import gdal

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
        out_path = utils_path._get_temp_filepath("vectorized_raster.shp", add_uuid=True)

    driver = utils_gdal._get_vector_driver_name_from_path(out_path)

    datasource = driver.CreateDataSource(out_path)
    layer = datasource.CreateLayer(out_path, srs=projection)

    try:
        gdal.Polygonize(src_band, None, layer, 0)
    except:
        raise RuntimeError(f"Error while vectorizing raster. {raster}") from None

    return out_path


def raster_vectorize(
    raster: Union[str, gdal.Dataset, List],
    out_path: Optional[str] = None,
    band: int = 1,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
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
    utils_base._type_check(overwrite, [bool], "overwrite")

    raster_list = utils_base._get_variable_as_list(raster)
    assert utils_gdal._check_is_raster_list(raster_list), f"Invalid raster in raster list: {raster_list}"

    path_list = utils_io._get_output_paths(
        raster_list,
        out_path,
        add_uuid=add_uuid or out_path is None,
        prefix=prefix,
        suffix=suffix,
        overwrite=overwrite,
    )

    vectorized_rasters = []
    for index, in_raster in enumerate(raster_list):
        vectorized_rasters.append(
            _raster_vectorize(
                in_raster,
                out_path=path_list[index],
                band=band,
            )
        )

    if isinstance(raster, list):
        return vectorized_rasters

    return vectorized_rasters[0]
