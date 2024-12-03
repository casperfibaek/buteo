# Standard library
import os
from typing import Union, Optional, List, Dict, Any, Callable, Tuple

# External
from osgeo import ogr, gdal, osr

# Internal
from buteo.utils import (
    utils_io,
    utils_base,
    utils_gdal,
    utils_bbox,
    utils_path,
    utils_projection,
)



def vector_to_extent(
    vector: Union[str, ogr.DataSource],
    out_path: Optional[str] = None,
    *,
    latlng: bool = False,
    overwrite: bool = True,
) -> str:
    """Converts a vector to a vector file with the extent as a polygon.

    Parameters
    ----------
    vector : str or ogr.DataSource
        The vector to convert.

    out_path : str, optional
        The path to save the extent to. If None, the extent is saved in memory. Default: None.

    latlng : bool, optional
        If True, the extent is returned in latlng coordinates. If false,
        the projection of the vector is used. Default: False.

    overwrite : bool, optional
        If True, the output file is overwritten if it exists. Default: True.

    Returns
    -------
    str
        The path to the extent.
    """
    utils_base._type_check(vector, [str, ogr.DataSource], "vector")
    utils_base._type_check(out_path, [str, type(None)], "out_path")
    utils_base._type_check(latlng, [bool], "latlng")
    utils_base._type_check(overwrite, [bool], "overwrite")

    if out_path is None:
        out_path = utils_path._get_temp_filepath("temp_extent.gpkg", add_timestamp=True, add_uuid=True)

    if not utils_path._check_is_valid_output_filepath(out_path):
        raise ValueError(f"Invalid output path: {out_path}")

    metadata = _get_basic_metadata_vector(vector)

    if latlng:
        extent = metadata["bounds_latlng"]
    else:
        extent = metadata["bounds_vector"]

    extent = ogr.CreateGeometryFromWkt(extent, metadata["projection_osr"])

    driver_name = utils_gdal._get_driver_name_from_path(out_path)

    driver = ogr.GetDriverByName(driver_name)
    extent_ds = driver.CreateDataSource(out_path)
    extent_layer = extent_ds.CreateLayer("extent", metadata["projection_osr"], ogr.wkbPolygon)
    extent_feature = ogr.Feature(extent_layer.GetLayerDefn())
    extent_feature.SetGeometry(extent)
    extent_layer.CreateFeature(extent_feature)

    extent_ds = None
    extent_layer = None
    extent_feature = None

    return out_path