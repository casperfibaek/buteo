"""### Convert geometry composition. ###

Convert geometries from multiparts and singleparts and vice versa.
"""

import sys; sys.path.append("../")

# Standard library
from typing import Union, Optional, List

# External
from osgeo import ogr

wkbMBit = 0x40000000


# Internal
from buteo.utils import (
    utils_base,
    utils_gdal,
    utils_path,
    utils_io,
)
from buteo.core_vector.core_vector_read import _open_vector, _vector_get_layer


def _parse_geom_type(geom_type: int) -> dict[str, Union[int, str, bool]]:
    """Parse the geometry type to a 2D or 3D type."""
    if geom_type == 1:
        return {"type": "point", "multi": False, "3D": False, "M": False, "number": 1}
    elif geom_type == 2:
        return {"type": "linestring", "multi": False, "3D": False, "M": False, "number": 2}
    elif geom_type == 3:
        return {"type": "polygon", "multi": False, "3D": False, "M": False, "number": 3}
    elif geom_type == 4:
        return {"type": "point", "multi": True, "3D": False, "M": False, "number": 4}
    elif geom_type == 5:
        return {"type": "linestring", "multi": True, "3D": False, "M": False, "number": 5}
    elif geom_type == 6:
        return {"type": "polygon", "multi": True, "3D": False, "M": False, "number": 6}
    elif geom_type == 7:
        return {"type": "geometrycollection", "multi": False, "3D": False, "M": False, "number": 7}
    elif geom_type == 1001:
        return {"type": "point", "multi": False, "3D": True, "M": False, "number": 1001}
    elif geom_type == 1002:
        return {"type": "linestring", "multi": False, "3D": True, "M": False, "number": 1002}
    elif geom_type == 1003:
        return {"type": "polygon", "multi": False, "3D": True, "M": False, "number": 1003}
    elif geom_type == 1004:
        return {"type": "point", "multi": True, "3D": True, "M": False, "number": 1004}
    elif geom_type == 1005:
        return {"type": "linestring", "multi": True, "3D": True, "M": False, "number": 1005}
    elif geom_type == 1006:
        return {"type": "polygon", "multi": True, "3D": True, "M": False, "number": 1006}
    elif geom_type == 1007:
        return {"type": "geometrycollection", "multi": False, "3D": True, "M": False, "number": 1007}
    elif geom_type == 2001:
        return {"type": "point", "multi": False, "3D": False, "M": True, "number": 2001}
    elif geom_type == 2002:
        return {"type": "linestring", "multi": False, "3D": False, "M": True, "number": 2002}
    elif geom_type == 2003:
        return {"type": "polygon", "multi": False, "3D": False, "M": True, "number": 2003}
    elif geom_type == 2004:
        return {"type": "point", "multi": True, "3D": False, "M": True, "number": 2004}
    elif geom_type == 2005:
        return {"type": "linestring", "multi": True, "3D": False, "M": True, "number": 2005}
    elif geom_type == 2006:
        return {"type": "polygon", "multi": True, "3D": False, "M": True, "number": 2006}
    elif geom_type == 2007:
        return {"type": "geometrycollection", "multi": False, "3D": False, "M": True, "number": 2007}
    elif geom_type == 3001:
        return {"type": "point", "multi": False, "3D": True, "M": True, "number": 3001}
    elif geom_type == 3002:
        return {"type": "linestring", "multi": False, "3D": True, "M": True, "number": 3002}
    elif geom_type == 3003:
        return {"type": "polygon", "multi": False, "3D": True, "M": True, "number": 3003}
    elif geom_type == 3004:
        return {"type": "point", "multi": True, "3D": True, "M": True, "number": 3004}
    elif geom_type == 3005:
        return {"type": "linestring", "multi": True, "3D": True, "M": True, "number": 3005}
    elif geom_type == 3006:
        return {"type": "polygon", "multi": True, "3D": True, "M": True, "number": 3006}
    elif geom_type == 3007:
        return {"type": "geometrycollection", "multi": False, "3D": True, "M": True, "number": 3007}
    else:
        raise ValueError(f"Invalid geometry type: {geom_type}")


def _geom_conversions_required(source_geom_dict, target_geom_dict):
    """ Determine which geometry conversions are required to convert between two geometries. """
    operations = []
    if source_geom_dict["3D"] and not target_geom_dict["3D"]:
        operations.append("flatten_to_2D")

    if not source_geom_dict["3D"] and target_geom_dict["3D"]:
        operations.append("promote_to_3D")

    if source_geom_dict["multi"] and not target_geom_dict["multi"]:
        operations.append("explode_to_singleparts")

    if not source_geom_dict["multi"] and target_geom_dict["multi"]:
        operations.append("aggregate_to_multiparts")

    if source_geom_dict["m"] and not target_geom_dict["m"]:
        operations.append("remove_m")

    if not source_geom_dict["m"] and target_geom_dict["m"]:
        operations.append("add_m")

    if source_geom_dict["type"] != target_geom_dict["type"]:
        raise ValueError("Cannot convert between different geometry types.")

    return operations


def vector_convert_geometry(
    vector: Union[str, ogr.DataSource],
    multigeometry: Optional[bool] = None,
    z: Optional[bool] = None,
    m: Optional[bool] = None,
    output_path: Optional[str] = None,
    layer_name_or_id: Union[str, int] = 0,
    z_attribute: Optional[str] = None,
    m_attribute: Optional[str] = None,
    *,
    prefix: str = "",
    suffix: str = "",
    overwrite: bool = False,
) -> str:
    """ Convert the geometry of a vector to a different subtype.
    
    Convert between multiparts and singleparts, 2D and 3D, and with or without M values.

    Parameters
    ----------
    vector : str or ogr.DataSource
        The vector to convert.
    multigeometry : bool, optional
        If True, the output vector will be multiparts. If False it will be singleparts. Default: None
    z : bool, optional
        If True, the output vector will be 3D. Default: None
    m : bool, optional
        If True, the output vector will have M (measure) values. Default: None
    output_path : str, optional
        The output path. Default: None (in-memory is created)
    layer_name_or_id : str or int, optional
        The name or index of the layer to convert. Default: 0
    z_attribute : str, optional
        The name of the attribute to use for Z values. If None, 0.0 is inserted Default: None
    m_attribute : str, optional
        The name of the attribute to use for M values. If None, 0.0 is inserted Default: None
    prefix : str, optional
        Prefix to add to output path. Default: ""
    suffix : str, optional
        Suffix to add to output path. Default: ""
    overwrite : bool, optional
        If True, overwrites existing files. Default: False

    Returns
    -------
    str
        The path to the converted vector
    """
    utils_base._type_check(vector, [str, ogr.DataSource], "vector")
    utils_base._type_check(multigeometry, [type(None), bool], "multigeometry")
    utils_base._type_check(z, [type(None), bool], "z")
    utils_base._type_check(m, [type(None), bool], "m")
    utils_base._type_check(output_path, [type(None), str], "output_path")
    utils_base._type_check(layer_name_or_id, [str, int], "layer_name_or_id")
    utils_base._type_check(z_attribute, [type(None), str], "z_attribute")
    utils_base._type_check(m_attribute, [type(None), str], "m_attribute")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(overwrite, [bool], "overwrite")

    ds = _open_vector(vector, writeable=False)
    layer = _vector_get_layer(ds, layer_name_or_id)[0]

    if not isinstance(layer, ogr.Layer):
        raise ValueError("Could not open the layer.")
    
    target_dict = {

    }

    return ""


if __name__ == "__main__":
    # Create a 2D vector with three points using WKT
    point_wkts_2d = ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)"]
    driver = ogr.GetDriverByName("ESRI Shapefile")
    path_2d = "/vsimem/points_2d.shp"
    ds_2d = driver.CreateDataSource(path_2d)
    layer_2d = ds_2d.CreateLayer("layer_2d", geom_type=ogr.wkbPoint)
    for wkt in point_wkts_2d:
        geometry = ogr.CreateGeometryFromWkt(wkt)
        feature = ogr.Feature(layer_2d.GetLayerDefn())
        feature.SetGeometry(geometry)
        layer_2d.CreateFeature(feature)
        feature = None

    # Create a similar 3D vector
    point_wkts_3d = ["POINT Z (0 0 0)", "POINT Z (1 1 1)", "POINT Z (2 2 2)"]
    path_3d = "/vsimem/points_3d.shp"
    ds_3d = driver.CreateDataSource(path_3d)
    layer_3d = ds_3d.CreateLayer("layer_3d", geom_type=ogr.wkbPoint25D)
    for wkt in point_wkts_3d:
        geometry = ogr.CreateGeometryFromWkt(wkt)
        feature = ogr.Feature(layer_3d.GetLayerDefn())
        feature.SetGeometry(geometry)
        layer_3d.CreateFeature(feature)
        feature = None



    import pdb; pdb.set_trace()
