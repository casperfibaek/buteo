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



def _open_vector(
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
        True if the vector has a geometry column, False otherwise
    """
    utils_base._type_check(vector, [str, ogr.DataSource], "vector")
    utils_base._type_check(layer_name_or_id, [type(None), str, int], "layer_name_or_id")

    ref = _open_vector(vector)

    if layer_name_or_id is None:
        layer_count = ref.GetLayerCount()
        for idx in range(layer_count):
            layer = ref.GetLayer(idx)
            if layer.GetGeomType() == ogr.wkbNone:
                continue

            if layer.GetFeatureCount() == 0:
                continue

            feature = layer.GetNextFeature()
            if feature is not None and feature.GetGeometryRef() is not None:
                return True
        return False
    else:
        layer = ref.GetLayer(layer_name_or_id) if isinstance(layer_name_or_id, int) else ref.GetLayerByName(layer_name_or_id)
        if layer is None:
            return False

        if layer.GetGeomType() == ogr.wkbNone:
            return False

        if layer.GetFeatureCount() == 0:
            return False

        feature = layer.GetNextFeature()
        return feature is not None and feature.GetGeometryRef() is not None


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
        True if the vector has attributes, False otherwise
    """
    utils_base._type_check(vector, [str, ogr.DataSource], "vector")
    utils_base._type_check(layer_name_or_id, [type(None), str, int], "layer_name_or_id")
    utils_base._type_check(attributes, [type(None), str, list], "attributes")

    ref = _open_vector(vector)

    if layer_name_or_id is None:
        layer_count = ref.GetLayerCount()
        for idx in range(layer_count):
            layer = ref.GetLayer(idx)
            defn = layer.GetLayerDefn()
            field_count = defn.GetFieldCount()

            if field_count == 0:
                continue

            if attributes is not None:
                if isinstance(attributes, str):
                    attributes = [attributes]

                field_names = [defn.GetFieldDefn(i).GetName() for i in range(field_count)]
                if all(attr in field_names for attr in attributes):
                    return True
            else:
                return True
        return False
    else:
        layer = ref.GetLayer(layer_name_or_id) if isinstance(layer_name_or_id, int) else ref.GetLayerByName(layer_name_or_id)
        if layer is None:
            return False

        defn = layer.GetLayerDefn()
        field_count = defn.GetFieldCount()

        if field_count == 0:
            return False

        if attributes is not None:
            if isinstance(attributes, str):
                attributes = [attributes]

            field_names = [defn.GetFieldDefn(i).GetName() for i in range(field_count)]
            return all(attr in field_names for attr in attributes)

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
        True if the vector has a CRS defined, False otherwise
    """
    utils_base._type_check(vector, [str, ogr.DataSource], "vector")
    utils_base._type_check(layer_name_or_id, [type(None), str, int], "layer_name_or_id")

    ref = _open_vector(vector)

    if layer_name_or_id is None:
        layer_count = ref.GetLayerCount()
        for idx in range(layer_count):
            layer = ref.GetLayer(idx)
            spatial_ref = layer.GetSpatialRef()
            if spatial_ref is None or spatial_ref.ExportToWkt() == '':
                return False
        return True if layer_count > 0 else False
    else:
        layer = ref.GetLayer(layer_name_or_id) if isinstance(layer_name_or_id, int) else ref.GetLayerByName(layer_name_or_id)
        if layer is None:
            return False

        spatial_ref = layer.GetSpatialRef()
        return spatial_ref is not None and not spatial_ref.ExportToWkt() == ''


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

    ref = _open_vector(vector)

    if not allow_empty:
        return check_vector_has_geometry(vector, layer_name_or_id=layer_name_or_id)

    if layer_name_or_id is None:
        layer_count = ref.GetLayerCount()
        for idx in range(layer_count):
            layer = ref.GetLayer(idx)
            if layer.GetFeatureCount() == 0 and not allow_empty:
                return True

            for feature in layer:
                geom = feature.GetGeometryRef()
                if geom is None:
                    return True
                if not geom.IsValid():
                    return True
        return False
    else:
        layer = ref.GetLayer(layer_name_or_id) if isinstance(layer_name_or_id, int) else ref.GetLayerByName(layer_name_or_id)
        if layer is None:
            return True

        if layer.GetFeatureCount() == 0 and not allow_empty:
            return True

        for feature in layer:
            geom = feature.GetGeometryRef()
            if geom is None:
                return True
            if not geom.IsValid():
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

    ref = _open_vector(vector, writeable=True)

    if layer_name_or_id is None:
        layer_count = ref.GetLayerCount()
        all_valid = True
        for idx in range(layer_count):
            layer = ref.GetLayer(idx)
            layer.ResetReading()
            for feature in layer:
                geom = feature.GetGeometryRef()
                if geom is None:
                    continue

                if not geom.IsValid():
                    fixed_geom = geom.Buffer(0)
                    if fixed_geom is not None and fixed_geom.IsValid():
                        feature.SetGeometry(fixed_geom)
                        layer.SetFeature(feature)

            layer.ResetReading()
            if check_vector_has_invalid_geometry(ref, layer_name_or_id=idx):
                all_valid = False
        return all_valid
    else:
        layer = ref.GetLayer(layer_name_or_id) if isinstance(layer_name_or_id, int) else ref.GetLayerByName(layer_name_or_id)
        if layer is None:
            return False

        layer.ResetReading()
        for feature in layer:
            geom = feature.GetGeometryRef()
            if geom is None:
                continue

            if not geom.IsValid():
                fixed_geom = geom.Buffer(0)
                if fixed_geom is not None and fixed_geom.IsValid():
                    feature.SetGeometry(fixed_geom)
                    layer.SetFeature(feature)

        layer.ResetReading()
        return not check_vector_has_invalid_geometry(ref, layer_name_or_id=layer_name_or_id)
