""" ### Metadata functions for vector layers. ### """

# Standard library
from typing import Union, Dict, Any, Optional
from warnings import warn
import os

# External
from osgeo import ogr, osr

# Internal
from buteo.utils import (
    utils_base,
    utils_gdal,
    utils_bbox,
    utils_projection,
    utils_translate,
    utils_path,
)
from buteo.core_vector.core_vector_read import _open_vector, _vector_get_layer



def _parse_geom_type(geom_type: int) -> dict[str, Union[int, str, bool]]:
    """Parse the geometry type to a 2D or 3D type."""

    if not utils_translate._check_geom_is_geomtype(geom_type):
        if not utils_translate._check_geom_is_wkbgeom(geom_type):
            raise ValueError(f"Invalid geometry type: {geom_type}")

        geom_type = utils_translate._convert_wkb_to_geomtype(geom_type)

    geom_type_fullname = ogr.GeometryTypeToName(geom_type)
    geom_type_complex = utils_translate._convert_geomtype_to_wkb(geom_type)

    geom_dict = {}
    if geom_type == 1:
        geom_dict = {"type": "point", "multi": False, "3D": False, "M": False, "number_simple": 1}
    elif geom_type == 2:
        geom_dict = {"type": "linestring", "multi": False, "3D": False, "M": False, "number_simple": 2}
    elif geom_type == 3:
        geom_dict = {"type": "polygon", "multi": False, "3D": False, "M": False, "number_simple": 3}
    elif geom_type == 4:
        geom_dict = {"type": "point", "multi": True, "3D": False, "M": False, "number_simple": 4}
    elif geom_type == 5:
        geom_dict = {"type": "linestring", "multi": True, "3D": False, "M": False, "number_simple": 5}
    elif geom_type == 6:
        geom_dict = {"type": "polygon", "multi": True, "3D": False, "M": False, "number_simple": 6}
    elif geom_type == 7:
        geom_dict = {"type": "geometrycollection", "multi": False, "3D": False, "M": False, "number_simple": 7}
    elif geom_type == 1001:
        geom_dict = {"type": "point", "multi": False, "3D": True, "M": False, "number_simple": 1001}
    elif geom_type == 1002:
        geom_dict = {"type": "linestring", "multi": False, "3D": True, "M": False, "number_simple": 1002}
    elif geom_type == 1003:
        geom_dict = {"type": "polygon", "multi": False, "3D": True, "M": False, "number_simple": 1003}
    elif geom_type == 1004:
        geom_dict = {"type": "point", "multi": True, "3D": True, "M": False, "number_simple": 1004}
    elif geom_type == 1005:
        geom_dict = {"type": "linestring", "multi": True, "3D": True, "M": False, "number_simple": 1005}
    elif geom_type == 1006:
        geom_dict = {"type": "polygon", "multi": True, "3D": True, "M": False, "number_simple": 1006}
    elif geom_type == 1007:
        geom_dict = {"type": "geometrycollection", "multi": False, "3D": True, "M": False, "number_simple": 1007}
    elif geom_type == 2001:
        geom_dict = {"type": "point", "multi": False, "3D": False, "M": True, "number_simple": 2001}
    elif geom_type == 2002:
        geom_dict = {"type": "linestring", "multi": False, "3D": False, "M": True, "number_simple": 2002}
    elif geom_type == 2003:
        geom_dict = {"type": "polygon", "multi": False, "3D": False, "M": True, "number_simple": 2003}
    elif geom_type == 2004:
        geom_dict = {"type": "point", "multi": True, "3D": False, "M": True, "number_simple": 2004}
    elif geom_type == 2005:
        geom_dict = {"type": "linestring", "multi": True, "3D": False, "M": True, "number_simple": 2005}
    elif geom_type == 2006:
        geom_dict = {"type": "polygon", "multi": True, "3D": False, "M": True, "number_simple": 2006}
    elif geom_type == 2007:
        geom_dict = {"type": "geometrycollection", "multi": False, "3D": False, "M": True, "number_simple": 2007}
    elif geom_type == 3001:
        geom_dict = {"type": "point", "multi": False, "3D": True, "M": True, "number_simple": 3001}
    elif geom_type == 3002:
        geom_dict = {"type": "linestring", "multi": False, "3D": True, "M": True, "number_simple": 3002}
    elif geom_type == 3003:
        geom_dict = {"type": "polygon", "multi": False, "3D": True, "M": True, "number_simple": 3003}
    elif geom_type == 3004:
        geom_dict = {"type": "point", "multi": True, "3D": True, "M": True, "number_simple": 3004}
    elif geom_type == 3005:
        geom_dict = {"type": "linestring", "multi": True, "3D": True, "M": True, "number_simple": 3005}
    elif geom_type == 3006:
        geom_dict = {"type": "polygon", "multi": True, "3D": True, "M": True, "number_simple": 3006}
    elif geom_type == 3007:
        geom_dict = {"type": "geometrycollection", "multi": False, "3D": True, "M": True, "number_simple": 3007}
    else:
        raise ValueError(f"Invalid geometry type: {geom_type}")

    geom_dict.update({
        "number_complex": geom_type_complex,
        "fullname": geom_type_fullname
    })

    return geom_dict


def _get_basic_info_vector(
    datasource: ogr.DataSource,
    layer_name_or_id: Union[str, int] = 0,
) -> Dict[str, Any]:
    """Get basic information from a ogr.DataSource.

    Parameters
    ----------
    datasource : ogr.DataSource
        The ogr DataSource to extract information from.
    layer_name_or_id : Union[str, int], optional
        The layer name or index to extract information from, default: 0.

    Returns
    -------
    Dict[str, Any]
        Basic vector information including layers, features, geometry types, and projections.

    Raises
    ------
    ValueError
        If the dataset is invalid or cannot be read.
    """
    utils_base._type_check(datasource, [ogr.DataSource], "datasource")
    utils_base._type_check(layer_name_or_id, [str, int], "layer_name_or_id")

    layer_list = _vector_get_layer(datasource, layer_name_or_id)

    if len(layer_list) == 0:
        raise ValueError("Could not read layer from dataset.")

    layer = layer_list[0]
    layer_name = layer.GetName()

    projection_osr = layer.GetSpatialRef()

    if projection_osr is None:
        warn("Dataset has no projection defined. Using default projection.")
        projection_osr = utils_projection._get_default_projection_osr()

    projection_wkt = projection_osr.ExportToWkt()

    feature_count = layer.GetFeatureCount()
    geom_type = layer.GetGeomType()

    feature_count = layer.GetFeatureCount()
    fid_column = layer.GetFIDColumn()
    geom_column = layer.GetGeometryColumn()

    geom_dict = _parse_geom_type(geom_type)

    layer = None
    layer_list = None

    return {
        "layer_name": layer_name,
        "feature_count": feature_count,
        "geom_type_int": geom_dict["number_simple"],
        "geom_type_int_complex": geom_dict["number_complex"],
        "geom_type_name": geom_dict["type"],
        "geom_type_fullname": geom_dict["fullname"],
        "geom_multi": geom_dict["multi"],
        "geom_3d": geom_dict["3D"],
        "geom_m": geom_dict["M"],
        "projection_wkt": projection_wkt,
        "projection_osr": projection_osr,
        "column_geom": geom_column,
        "column_fid": fid_column,
    }



def _get_bounds_info_vector(
    datasource: ogr.DataSource,
    projection_osr: osr.SpatialReference,
    layer_name_or_id: Union[str, int] = 0,
) -> Dict[str, Any]:
    """Extract bounds and coordinate information from datasource.

    Parameters
    ----------
    datasource : ogr.DataSource
        The ogr datasource to process
    projection_osr : osr.SpatialReference
        The source projection
    layer_name_or_id : Union[str, int], optional
        The layer name or index to extract information from, default: 0.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing bounds information

    Raises
    ------
    ValueError
        If bounds computation fails
    """
    utils_base._type_check(datasource, [ogr.DataSource], "datasource")
    utils_base._type_check(layer_name_or_id, [str, int], "layer_name_or_id")
    utils_base._type_check(projection_osr, [osr.SpatialReference], "projection_osr")

    layer_list = _vector_get_layer(datasource, layer_name_or_id)

    if len(layer_list) == 0:
        raise ValueError("Could not read layer from dataset.")

    layer = layer_list[0]
    bbox = layer.GetExtent()

    default_projection = utils_projection._get_default_projection_osr()

    if utils_projection._check_projections_match(projection_osr, default_projection):
        bbox_latlng = bbox
    else:
        bbox_latlng = utils_projection.reproject_bbox(bbox, projection_osr, default_projection)

    centroid_x = (bbox[0] + bbox[1]) / 2
    centroid_y = (bbox[2] + bbox[3]) / 2
    centroid = (centroid_x, centroid_y)

    centroid_latlng_x = (bbox_latlng[0] + bbox_latlng[1]) / 2
    centroid_latlng_y = (bbox_latlng[2] + bbox_latlng[3]) / 2
    centroid_latlng = (centroid_latlng_x, centroid_latlng_y)

    bounds = utils_bbox._get_geom_from_bbox(bbox)
    bounds_latlng = utils_bbox._get_geom_from_bbox(bbox_latlng)

    return {
        "bbox": bbox,
        "bbox_latlng": bbox_latlng,
        "bounds": bounds.ExportToWkt(),
        "bounds_latlng": bounds_latlng.ExportToWkt(),
        "centroid": centroid,
        "centroid_latlng": centroid_latlng,
        "area_bbox": bounds.GetArea(),
        "area_latlng": bounds_latlng.GetArea(),
        "x_min": bbox[0],
        "x_max": bbox[1],
        "y_min": bbox[2],
        "y_max": bbox[3],
    }


def get_metadata_vector(
    datasource: Union[str, ogr.DataSource],
    layer_name_or_id: Optional[Union[str, int]] = None,
) -> Dict[str, Any]:
    """ Get metadata from a vector

    Parameters
    ----------
    datasource : Union[str, ogr.DataSource]
        The datasource to extract metadata from
    layer_name_or_id : Optional[Union[str, int]], optional
        The layer name or index to extract information from, default: None.
        If None, processes all the layers.
    
    Returns
    -------
    Dict[str, Any]
        Metadata dictionary containing information about the vector.
    """
    utils_base._type_check(datasource, [str, ogr.DataSource], "datasource")
    utils_base._type_check(layer_name_or_id, [str, int, None], "layer_name_or_id")

    ds = _open_vector(datasource, writeable=False)
    description = ds.GetDescription()
    path = utils_path._get_unix_path(description) if description != "" else "in_memory.mem"

    layers = _vector_get_layer(ds, layer_name_or_id)

    if len(layers) == 0:
        raise ValueError("Could not read layer from dataset.")

    metadata = {
        "path": path,
        "basename": os.path.basename(path),
        "name": os.path.splitext(os.path.basename(path))[0],
        "folder": os.path.dirname(path),
        "ext": os.path.splitext(path)[1],
        "in_memory": utils_gdal._check_is_dataset_in_memory(ds),
        "driver": ds.GetDriver().GetName(),
        "layer_count": len(layers),
    }

    metadata_layers = []
    for idx, _ in enumerate(layers):
        layer = layers[idx]
        layer_metadata = {}
        layer_defn = layer.GetLayerDefn()

        field_count = layer_defn.GetFieldCount()
        field_names = []
        field_types = []
        field_types_ogr = []

        for field_index in range(field_count):
            field_defn = layer_defn.GetFieldDefn(field_index)
            field_names.append(field_defn.GetName())
            field_type = field_defn.GetType()
            field_types_ogr.append(field_type)
            field_types.append(field_defn.GetFieldTypeName(field_type))

        meta_info = _get_basic_info_vector(ds, idx)
        meta_bounds = _get_bounds_info_vector(ds, meta_info["projection_osr"], idx)

        layer_metadata["field_count"] = field_count
        layer_metadata["field_names"] = field_names
        layer_metadata["field_types"] = field_types
        layer_metadata["field_types_ogr"] = field_types_ogr

        layer_metadata.update(meta_bounds)
        layer_metadata.update(meta_info)

        metadata_layers.append(layer_metadata)

    metadata["layers"] = metadata_layers

    return metadata
