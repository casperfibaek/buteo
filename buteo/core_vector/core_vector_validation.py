"""Module for vector validation functions."""

# Standard library
from typing import Union, Optional

# External
from osgeo import ogr

# Internal
from buteo.utils import utils_base
from buteo.core_vector.core_vector_read import open_vector, _vector_get_layers



def check_vector_has_invalid_geometry(
    vector: Union[str, ogr.DataSource],
    *,
    layer_name_or_id: Optional[Union[str, int]] = None,
    allow_empty: bool = False,
) -> bool:
    """Checks if a vector has invalid geometry.

    Parameters
    ----------
    vector : str or ogr.DataSource
        A path to a vector or an OGR datasource
    layer_name_or_id : str or int, optional
        The name or index of the layer to check. Default: None (checks all layers)
    allow_empty : bool, optional
        If True, empty geometries are considered valid. Default: False

    Returns
    -------
    bool
        True if the vector has invalid geometry, False if all geometries are valid
    """
    utils_base._type_check(vector, [str, ogr.DataSource], "vector")
    utils_base._type_check(layer_name_or_id, [type(None), str, int], "layer_name_or_id")
    utils_base._type_check(allow_empty, [bool], "allow_empty")

    ref = open_vector(vector)
    layers = _vector_get_layers(ref, layer_name_or_id)

    if len(layers) == 0:
        raise ValueError("No layers found in vector")

    for layer in layers:
        if layer.GetGeomType() == ogr.wkbNone: # if it is a table layer, skip it
            continue

        for feature in layer:
            geom = feature.GetGeometryRef()
            if geom is None:
                if not allow_empty:
                    return True
            elif not geom.IsValid():
                return True

    return False


def vector_fix_geometry(
    vector: Union[str, ogr.DataSource],
    *,
    layer_name_or_id: Optional[Union[str, int]] = None,
) -> bool:
    """Attempts to fix invalid geometries in a vector.

    Parameters
    ----------
    vector : str or ogr.DataSource
        A path to a vector or an OGR datasource
    layer_name_or_id : str or int, optional
        The name or index of the layer to fix. Default: None (fixes all layers)

    Returns
    -------
    bool
        True if all geometries are now valid, False if some remain invalid
    """
    utils_base._type_check(vector, [str, ogr.DataSource], "vector")
    utils_base._type_check(layer_name_or_id, [type(None), str, int], "layer_name_or_id")

    ref = open_vector(vector)
    layers = _vector_get_layers(ref, layer_name_or_id)

    for layer in layers:
        for feature in layer:
            geom = feature.GetGeometryRef()
            if geom and not geom.IsValid():
                fixed_geom = geom.MakeValid()
                if fixed_geom and fixed_geom.IsValid():
                    feature.SetGeometry(fixed_geom)
                    layer.SetFeature(feature)

    # Flush changes
    ref.FlushCache()
    ref = None

    return True
