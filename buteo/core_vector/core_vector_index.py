""" ### Index functions for vector layers. ### """

# Standard library
from typing import Union

# External
from osgeo import ogr


from buteo.core_vector.core_vector_read import _open_vector, _vector_get_layer



def _check_if_vector_supports_index(vector: Union[str, ogr.DataSource]) -> bool:
    """
    Checks if a vector format supports spatial indexes.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource]
        The vector to check.

    Returns
    -------
    bool
        True if the format supports spatial indexes, False otherwise.
    """
    ds = _open_vector(vector)
    driver = ds.GetDriver()
    driver_name = driver.GetName()

    if driver_name in ["GPKG"]:
        return True

    return False


def _check_if_vector_is_flatgeobuf(vector: Union[str, ogr.DataSource]) -> bool:
    """
    Checks if a vector is in the FlatGeobuf format.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource]
        The vector to check.

    Returns
    -------
    bool
        True if the vector is in the FlatGeobuf format, False otherwise.
    """
    ds = _open_vector(vector)
    driver = ds.GetDriver()
    driver_name = driver.GetName()

    if driver_name in ["FlatGeobuf"]:
        return True

    return False


def check_vector_has_index(
    vector: Union[str, ogr.DataSource],
    layer_name_or_id: Union[str, int] = 0,
) -> bool:
    """
    Checks if a vector has a spatial index.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource]
        The vector to check.
    layer_name_or_id : Union[str, int], optional
        The layer name or index to check, default: 0.

    Returns
    -------
    bool
        True if the vector has a spatial index, False otherwise.

    Raises
    ------
    ValueError
        If the vector cannot be read.
    RuntimeError
        If the index cannot be read or determined.
    """
    if _check_if_vector_is_flatgeobuf(vector):
        return True

    if not _check_if_vector_supports_index(vector):
        return False

    ds = _open_vector(vector)
    layer = _vector_get_layer(ds, layer_name_or_id)[0]

    if not isinstance(layer, ogr.Layer):
        raise ValueError("Could not open layer.")

    layer_name = layer.GetName()
    geom_name = layer.GetGeometryColumn() or "geom"

    try:
        sql = f"SELECT HasSpatialIndex('{layer_name}', '{geom_name}')"
        sql_request = ds.ExecuteSQL(sql, dialect="SQLITE")
        sql_result = sql_request.GetNextFeature().GetField(0)

        ds.ReleaseResultSet(sql_request)

    # if this raises "Undefined function 'HasSpatialIndex' used." the data format does not support spatial indexes
    except RuntimeError:
        return False

    ds = None
    layer = None
    sql_request = None

    return sql_result == 1


def vector_create_index(
    vector: Union[str, ogr.DataSource],
    layer_name_or_id: Union[str, int] = 0,
    overwrite: bool = False,
) -> bool:
    """
    Creates a spatial index for a vector layer.

    If the index already exists, and overwrite is True, it will be overwritten.
    If the data format does not support spatial indexes, False will be returned.
    If the index creation fails, False will be returned.
    If the index is created successfully, True will be returned.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource]
        The vector to create the index for.
    layer_name_or_id : Union[str, int], optional
        The layer name or index to create the index for, default: 0.
    overwrite : bool, optional
        If an existing index should be overwritten, default: False.

    Returns
    -------
    bool
        True if the index was created, False otherwise.
    """
    if _check_if_vector_is_flatgeobuf(vector):
        return True

    if not _check_if_vector_supports_index(vector):
        return False

    ds = _open_vector(vector, writeable=True)
    layer = _vector_get_layer(ds, layer_name_or_id)[0]

    if not isinstance(layer, ogr.Layer):
        raise ValueError("Could not open layer.")

    if check_vector_has_index(ds, layer_name_or_id):
        if not overwrite:
            return True
        else:
            vector_delete_index(ds, layer_name_or_id)

    layer_name = layer.GetName()
    geom_name = layer.GetGeometryColumn() or "geom"

    try:
        sql = f"SELECT CreateSpatialIndex('{layer_name}', '{geom_name}')"
        sql_request = ds.ExecuteSQL(sql, dialect="SQLITE")
        sql_result = sql_request.GetNextFeature().GetField(0)
        ds.ReleaseResultSet(sql_request)
    except RuntimeError:
        return False

    ds, layer, sql_request = (None, None, None)

    return sql_result == 1


def vector_delete_index(
    vector: Union[str, ogr.DataSource],
    layer_name_or_id: Union[str, int] = 0,
) -> bool:
    """
    Deletes a spatial index from a vector layer.

    If the index does not exist, False will be returned.
    If the data format does not support spatial indexes, False will be returned.
    If the index deletion fails, False will be returned.
    If the index is deleted successfully, True will be returned.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource]
        The vector to delete the index from.
    layer_name_or_id : Union[str, int], optional
        The layer name or index to delete the index from, default: 0.

    Returns
    -------
    bool
        True if the index was deleted, False otherwise.
    """
    if not _check_if_vector_supports_index(vector):
        return False

    ds = _open_vector(vector, writeable=True)
    layer = _vector_get_layer(ds, layer_name_or_id)[0]

    if not isinstance(layer, ogr.Layer):
        raise ValueError("Could not open layer.")

    if check_vector_has_index(ds, layer_name_or_id) is False:
        return True

    layer_name = layer.GetName()
    geom_name = layer.GetGeometryColumn() or "geom"

    try:
        sql = f"SELECT DisableSpatialIndex('{layer_name}', '{geom_name}')"
        sql_request = ds.ExecuteSQL(sql, dialect="SQLITE")
        sql_result = sql_request.GetNextFeature().GetField(0)

        ds.ReleaseResultSet(sql_request)

    # if this raises "no such spatial index" the index does not exist
    except RuntimeError:
        return False

    ds = None
    layer = None
    sql_request = None

    return sql_result == 1
