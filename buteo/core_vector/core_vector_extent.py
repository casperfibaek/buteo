""" Module for converting a vector to an extent polygon. """

# Standard library
from typing import Union, Optional

# External
from osgeo import ogr

# Internal
from buteo.utils import (
    utils_base,
    utils_gdal,
    utils_path,
)
from buteo.core_vector.core_vector_info import get_metadata_vector



def vector_to_extent(
    vector: Union[str, ogr.DataSource],
    out_path: Optional[str] = None,
    layer_name_or_id: Optional[Union[str, int]] = None,
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

    layer_name_or_id : int, optional
        The layer to use. Default: 0.

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

    metadata = get_metadata_vector(vector, layer_name_or_id=layer_name_or_id)

    if latlng:
        extent = metadata["layers"][0]["bounds_latlng"]
    else:
        extent = metadata["layers"][0]["bounds"]

    extent = ogr.CreateGeometryFromWkt(extent, metadata["layers"][0]["projection_osr"])

    driver_name = utils_gdal._get_driver_name_from_path(out_path)

    driver = ogr.GetDriverByName(driver_name)
    extent_ds = driver.CreateDataSource(out_path)
    extent_layer = extent_ds.CreateLayer("extent", metadata["layers"][0]["projection_osr"], ogr.wkbPolygon)
    extent_feature = ogr.Feature(extent_layer.GetLayerDefn())
    extent_feature.SetGeometry(extent)
    extent_layer.CreateFeature(extent_feature)

    extent_ds = None
    extent_layer = None
    extent_feature = None

    return out_path
