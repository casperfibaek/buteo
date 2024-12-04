"""### Basic IO functions for working with vectors. ###

The basic module for interacting with vector data

    * More attribute functions
    * Repair vector functions
    * Sanity checks
    * Joins (by attribute, location, summary, etc..)
    * Union, Erase, ..
    * Sampling functions
    * Vector intersects, etc..

"""
# Standard library
from typing import Union, Optional, List

# External
from osgeo import ogr, gdal

# Internal
from buteo.utils import (
    utils_base,
    utils_path,
)



def open_vector(
    vector: Union[str, ogr.DataSource],
    *,
    writeable: bool = False,
) -> ogr.DataSource:
    """Opens a vector in read or write mode.

    Parameters
    ----------
    vector : str or ogr.DataSource
        A path to a vector or an OGR datasource
    writeable : bool, optional
        If True, opens in write mode. Default: False

    Returns
    -------
    ogr.DataSource
        The opened vector dataset

    Raises
    ------
    TypeError
        If vector is not str or ogr.DataSource
    ValueError
        If vector path doesn't exist or file cannot be opened
    """
    utils_base._type_check(vector, [str, ogr.DataSource], "vector")
    utils_base._type_check(writeable, [bool], "writeable")

    if isinstance(vector, ogr.DataSource):
        return vector

    if not utils_path._check_file_exists(vector):
        raise ValueError(f"Input vector does not exist: {vector}")

    if vector.startswith("/vsizip/"):
        writeable = False

    gdal.PushErrorHandler("CPLQuietErrorHandler")
    dataset = ogr.Open(vector, 1 if writeable else 0)
    gdal.PopErrorHandler()

    if dataset is None:
        raise ValueError(f"Could not open vector: {vector}")

    return dataset


def _vector_get_layers(
    vector: ogr.DataSource,
    layer_name_or_id: Optional[Union[str, int]] = None,
) -> List[ogr.Layer]:
    """ Helper function to get layers from a vector. """
    if layer_name_or_id is not None:
        if isinstance(layer_name_or_id, int):
            layer = vector.GetLayer(layer_name_or_id)
            if layer is None:
                raise ValueError(f"Layer with index {layer_name_or_id} does not exist")
            layers = [layer]
        else:
            layer = vector.GetLayerByName(layer_name_or_id)
            if layer is None:
                raise ValueError(f"Layer with name '{layer_name_or_id}' does not exist")
            layers = [layer]
    else:
        layers = [vector.GetLayer(i) for i in range(vector.GetLayerCount())]

    return layers


def check_vector_has_geometry(
    vector: Union[str, ogr.DataSource],
    *,
    layer_name_or_id: Optional[Union[str, int]] = None,
) -> bool:
    """Checks if a vector has geometry.

    Parameters
    ----------
    vector : str or ogr.DataSource
        A path to a vector or an OGR datasource
    layer_name_or_id : str or int, optional
        The name or index of the layer to check. Default: None (checks all layers)

    Returns
    -------
    bool
        True if all vector layers have geometry columns, False otherwise
    """
    utils_base._type_check(vector, [str, ogr.DataSource], "vector")
    utils_base._type_check(layer_name_or_id, [type(None), str, int], "layer_name_or_id")

    ref = open_vector(vector)
    layers = _vector_get_layers(ref, layer_name_or_id)

    if len(layers) == 0:
        raise ValueError("No layers found in vector")

    for layer in layers:
        if layer.GetGeomType() == ogr.wkbNone:
            return False

        if layer.GetFeatureCount() == 0:
            return False

        feature = layer.GetNextFeature()
        if feature is None or feature.GetGeometryRef() is None:
            return False

    return True


def check_vector_has_attributes(
    vector: Union[str, ogr.DataSource],
    *,
    layer_name_or_id: Optional[Union[str, int]] = None,
    attributes: Optional[Union[str, List[str]]] = None,
) -> bool:
    """Checks if a vector has attributes.

    Parameters
    ----------
    vector : str or ogr.DataSource
        A path to a vector or an OGR datasource
    layer_name_or_id : str or int, optional
        The name or index of the layer to check. Default: None (checks all layers)
    attributes : str or List[str], optional
        Specific attribute(s) to check for. Default: None

    Returns
    -------
    bool
        True if all layers have the specified attributes, False otherwise
    """
    utils_base._type_check(vector, [str, ogr.DataSource], "vector")
    utils_base._type_check(layer_name_or_id, [type(None), str, int], "layer_name_or_id")
    utils_base._type_check(attributes, [type(None), str, list], "attributes")

    if attributes is not None and isinstance(attributes, str):
        attributes = [attributes]

    ref = open_vector(vector)
    layers = _vector_get_layers(ref, layer_name_or_id)

    if len(layers) == 0:
        raise ValueError("No layers found in vector")

    for layer in layers:
        defn = layer.GetLayerDefn()
        field_count = defn.GetFieldCount()

        if field_count == 0:
            return False

        if attributes is not None:
            field_names = [defn.GetFieldDefn(i).GetName() for i in range(field_count)]
            if not all(attr in field_names for attr in attributes):
                return False

    return True


def check_vector_has_crs(
    vector: Union[str, ogr.DataSource],
    *,
    layer_name_or_id: Optional[Union[str, int]] = None,
) -> bool:
    """Checks if a vector has a defined coordinate reference system.

    Parameters
    ----------
    vector : str or ogr.DataSource
        A path to a vector or an OGR datasource
    layer_name_or_id : str or int, optional
        The name or index of the layer to check. Default: None (checks all layers)

    Returns
    -------
    bool
        True if all vector layers have a CRS defined, False otherwise
    """
    utils_base._type_check(vector, [str, ogr.DataSource], "vector")
    utils_base._type_check(layer_name_or_id, [type(None), str, int], "layer_name_or_id")

    ref = open_vector(vector)
    layers = _vector_get_layers(ref, layer_name_or_id)

    if len(layers) == 0:
        raise ValueError("No layers found in vector")

    for layer in layers:
        spatial_ref = layer.GetSpatialRef()
        if spatial_ref is None or spatial_ref.ExportToWkt() == '':
            return False

    return True


def check_vector_has_multiple_layers(
    vector: Union[str, ogr.DataSource],
) -> bool:
    """Checks if a vector has multiple layers.

    Parameters
    ----------
    vector : str or ogr.DataSource
        A path to a vector or an OGR datasource

    Returns
    -------
    bool
        True if the vector has multiple layers, False otherwise
    """
    utils_base._type_check(vector, [str, ogr.DataSource], "vector")

    ref = open_vector(vector)
    layer_count = ref.GetLayerCount()
    ref = None

    return layer_count > 1


def check_vector_is_geometry_type(
    vector: Union[str, ogr.DataSource],
    geometry_type: Union[str, List[str]],
    *,
    layer_name_or_id: Optional[Union[str, int]] = None,
) -> bool:
    """Checks if a vector has a specific geometry type.

    Parameters
    ----------
    vector : str or ogr.DataSource
        A path to a vector or an OGR datasource
    geometry_type : str or List[str]
        The geometry type to check for, or multiple types. ex. ['POINT', 'LINESTRING'] or 'POLYGON'
    layer_name_or_id : str or int, optional
        The name or index of the layer to check. Default: None (checks all layers)

    Returns
    -------
    bool
        True if all vector layers have the specified geometry type, False otherwise

    Notes
    -----
    Valid geometry types are:
        'POINT'
        'LINE STRING'
        'POLYGON'
        'MULTIPOINT'
        'MULTILINESTRING'
        'MULTIPOLYGON'
        'GEOMETRYCOLLECTION'
        '3D POINT'
        '3D LINE STRING'
        '3D POLYGON'
        '3D MULTIPOINT'
        '3D MULTILINESTRING'
        '3D MULTIPOLYGON'
        '3D GEOMETRYCOLLECTION'
        'NONE'
    """
    utils_base._type_check(vector, [str, ogr.DataSource], "vector")
    utils_base._type_check(geometry_type, [str, [str]], "geometry_type")
    utils_base._type_check(layer_name_or_id, [type(None), str, int], "layer_name_or_id")

    if isinstance(geometry_type, str):
        geometry_type = [geometry_type]

    if not all(isinstance(geom, str) for geom in geometry_type):
        raise ValueError("geometry_type must be a string or list of strings")

    valid_geometry_types = [
        'POINT',
        'LINE STRING',
        'POLYGON',
        'MULTIPOINT',
        'MULTILINESTRING',
        'MULTIPOLYGON',
        'GEOMETRYCOLLECTION',
        '3D POINT',
        '3D LINE STRING',
        '3D POLYGON',
        '3D MULTIPOINT',
        '3D MULTILINESTRING',
        '3D MULTIPOLYGON',
        '3D GEOMETRYCOLLECTION',
        'NONE',
    ]

    # convert all strings to uppercase
    geometry_type = [geom.upper() for geom in geometry_type]

    if not all(geom in valid_geometry_types for geom in geometry_type):
        raise ValueError("Invalid geometry type provided, must be one of: " + ", ".join(valid_geometry_types))

    ref = open_vector(vector)
    layers = _vector_get_layers(ref, layer_name_or_id)

    if len(layers) == 0:
        raise ValueError("No layers found in vector")

    for layer in layers:
        layer_defn = layer.GetLayerDefn()
        layer_geom_type = ogr.GeometryTypeToName(layer_defn.GetGeomType())

        if not isinstance(layer_geom_type, str) or layer_geom_type.upper() not in geometry_type:
            return False

    return True


def check_vector_is_point_type(
    vector: Union[str, ogr.DataSource],
    *,
    layer_name_or_id: Optional[Union[str, int]] = None,
) -> bool:
    """Checks if a vector has a point geometry type.

    Parameters
    ----------
    vector : str or ogr.DataSource
        A path to a vector or an OGR datasource
    layer_name_or_id : str or int, optional
        The name or index of the layer to check. Default: None (checks all layers)

    Returns
    -------
    bool
        True if all vector layers have a point geometry type, False otherwise

    Notes
    -----
    Valid point geometry types are:
        'POINT'
        '3D POINT'
        'MULTIPOINT'
        '3D MULTIPOINT'
    """
    return check_vector_is_geometry_type(
        vector,
        ["POINT", "3D POINT", "MULTIPOINT", "3D MULTIPOINT"],
        layer_name_or_id=layer_name_or_id,
    )


def check_vector_is_line_type(
    vector: Union[str, ogr.DataSource],
    *,
    layer_name_or_id: Optional[Union[str, int]] = None,
) -> bool:
    """Checks if a vector has a line geometry type.

    Parameters
    ----------
    vector : str or ogr.DataSource
        A path to a vector or an OGR datasource
    layer_name_or_id : str or int, optional
        The name or index of the layer to check. Default: None (checks all layers)

    Returns
    -------
    bool
        True if all vector layers have a line geometry type, False otherwise

    Notes
    -----
    Valid line geometry types are:
        'LINE STRING'
        '3D LINE STRING'
        'MULTILINESTRING'
        '3D MULTILINESTRING'
    """
    return check_vector_is_geometry_type(
        vector,
        ["LINE STRING", "3D LINE STRING", "MULTILINESTRING", "3D MULTILINESTRING"],
        layer_name_or_id=layer_name_or_id,
    )


def check_vector_is_polygon_type(
    vector: Union[str, ogr.DataSource],
    *,
    layer_name_or_id: Optional[Union[str, int]] = None,
) -> bool:
    """Checks if a vector has a polygon geometry type.

    Parameters
    ----------
    vector : str or ogr.DataSource
        A path to a vector or an OGR datasource
    layer_name_or_id : str or int, optional
        The name or index of the layer to check. Default: None (checks all layers)

    Returns
    -------
    bool
        True if all vector layers have a polygon geometry type, False otherwise
    
    Notes
    -----
    Valid polygon geometry types are:
        'POLYGON'
        '3D POLYGON'
        'MULTIPOLYGON'
        '3D MULTIPOLYGON'
    """
    return check_vector_is_geometry_type(
        vector,
        ["POLYGON", "3D POLYGON", "MULTIPOLYGON", "3D MULTIPOLYGON"],
        layer_name_or_id=layer_name_or_id,
    )
