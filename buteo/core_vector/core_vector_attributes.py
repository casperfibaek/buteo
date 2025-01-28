""" Module for vector attribute operations. """
# Standard library
from typing import Union, Optional, List, Any, Tuple

# External
from osgeo import ogr

# Internal
from buteo.utils import (
    utils_io,
    utils_base,
)
from buteo.core_vector.core_vector_info import get_metadata_vector
from buteo.core_vector.core_vector_read import _open_vector


# TODO: Use a convert to CSV trick.

def _vector_get_attribute_table(
    vector: Union[str, ogr.DataSource],
    process_layer: int = 0,
    include_fids: bool = False,
    include_geometry: bool = False,
    include_attributes: bool = True,
) -> Tuple[List[str], List[List[Any]]]:
    """Get the attribute table(s) of a vector.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource, List[str, ogr.DataSource]]
        Vector layer(s) or path(s) to vector layer(s).

    process_layer : int, optional
        The layer to process. Default: 0 (first layer).

    include_fids : bool, optional
        If True, will include the FID column. Default: False.

    include_geometry : bool, optional
        If True, will include the geometry column. Default: False.

    include_attributes : bool, optional
        If True, will include the attribute columns. Default: True.

    Returns
    -------
    attribute_table : Dict[str, Any]
        The attribute table(s) of the vector(s).
    """
    assert isinstance(vector, (str, ogr.DataSource)), "vector must be a string or an ogr.DataSource object."
    assert isinstance(process_layer, int), "process_layer must be an integer."
    assert isinstance(include_fids, bool), "include_fids must be a boolean."
    assert isinstance(include_geometry, bool), "include_geometry must be a boolean."
    assert isinstance(include_attributes, bool), "include_attributes must be a boolean."

    ref = _open_vector(vector)
    metadata = get_metadata_vector(ref)

    attribute_table_header = metadata["layers"][process_layer]["field_names"]
    attribute_table = []

    layer = ref.GetLayer(process_layer)
    layer.ResetReading()
    while True:
        feature = layer.GetNextFeature()

        if feature is None:
            break

        attributes = [feature.GetFID()]

        for field_name in attribute_table_header:
            attributes.append(feature.GetField(field_name))

        if include_geometry:
            geom_defn = feature.GetGeometryRef()
            attributes.append(geom_defn.ExportToIsoWkt())

        attribute_table.append(attributes)

    attribute_table_header.insert(0, "fid")

    if include_geometry:
        attribute_table_header.append("geom")

    ref = None
    layer = None

    return attribute_table_header, attribute_table


def vector_get_attribute_table(
    vector: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
    process_layer: int = 0,
    include_fids: bool = False,
    include_geometry: bool = False,
    include_attributes: bool = True,
) -> Union[Tuple[List[str], List[List[Any]]], List[Tuple[List[str], List[List[Any]]]]]:
    """Get the attribute table(s) of a vector.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource, List[str, ogr.DataSource]]
        Vector layer(s) or path(s) to vector layer(s).

    process_layer : int, optional
        The layer to process. Default: 0 (first layer).

    include_fids : bool, optional
        If True, will include the FID column. Default: False.

    include_geometry : bool, optional
        If True, will include the geometry column. Default: False.

    include_attributes : bool, optional
        If True, will include the attribute columns. Default: True.

    return_header : bool, optional
        If True, will return the header. Default: True.

    Returns
    -------
    attribute_table : Dict[str, Any]
        The attribute table(s) of the vector(s).
    """
    utils_base._type_check(vector, [str, ogr.DataSource, [str, ogr.DataSource]], "vector")
    utils_base._type_check(process_layer, [int], "process_layer")
    utils_base._type_check(include_fids, [bool], "include_fids")
    utils_base._type_check(include_geometry, [bool], "include_geometry")
    utils_base._type_check(include_attributes, [bool], "include_attributes")

    input_is_list = isinstance(vector, list)
    in_paths = utils_io._get_input_paths(vector, "vector") # type: ignore

    output_attributes = []
    output_headers = []
    for in_vector in in_paths:
        header, table = _vector_get_attribute_table(
            in_vector,
            process_layer=process_layer,
            include_fids=include_fids,
            include_geometry=include_geometry,
            include_attributes=include_attributes,
        )
        output_headers.append(header)
        output_attributes.append(table)

    if input_is_list:
        return output_headers, output_attributes

    return output_headers[0], output_attributes[0]


def vector_add_field(
    vector: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
    field_name: str,
    field_type: str,
) -> Union[str, List[str]]:
    """Adds a field to a vector in place.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]]
        Vector layer(s) or path(s) to vector layer(s).

    field : str
        The name of the field to add.

    field_type : str
        The type of the field to add.
        `['int', 'integer', 'float', 'double', 'string', 'date', 'datetime', 'time', 'binary', 'intlist', 'integerlist', 'floatlist', 'doublelist', 'stringlist', 'datelist', 'datetimelist', 'timelist', 'binarylist']`

    Returns
    -------
    Union[str, List[str]]
        The path(s) to the output vector file(s).
    """
    assert isinstance(vector, (str, ogr.DataSource)), "vector must be a string or an ogr.DataSource object."
    assert isinstance(field_name, str), "field must be a string."
    assert isinstance(field_type, str), "field_type must be a valid vector type."

    if field_type in ['int', 'integer']:
        field_type = ogr.OFTInteger
    elif field_type in ['float', 'double']:
        field_type = ogr.OFTReal
    elif field_type in ['string']:
        field_type = ogr.OFTString
    elif field_type in ['date', 'datetime', 'time']:
        field_type = ogr.OFTDateTime
    elif field_type in ['binary']:
        field_type = ogr.OFTBinary
    elif field_type in ['intlist', 'integerlist']:
        field_type = ogr.OFTIntegerList
    elif field_type in ['floatlist', 'doublelist']:
        field_type = ogr.OFTRealList
    elif field_type in ['stringlist']:
        field_type = ogr.OFTStringList
    else:
        raise ValueError(f"Invalid field_type: {field_type}")

    input_is_list = isinstance(vector, list)

    in_paths = utils_io._get_input_paths(vector, "vector")

    for _idx, in_vector in enumerate(in_paths):
        ref = _open_vector(in_vector, writeable=True)

        layers = ref.GetLayerCount()

        for layer_index in range(layers):
            layer = ref.GetLayer(layer_index)
            layer.ResetReading()

            layer.CreateField(ogr.FieldDefn(field_name, field_type))

            layer.SyncToDisk()

        ref = None

    if input_is_list:
        return in_paths

    return in_paths[0]


def vector_set_attribute_table(
    vector: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
    header: List[str],
    attribute_table: List[List[Any]],
    match: Optional[str] = 'fid',
) -> Union[str, List[str]]:
    """Sets the attribute table of a vector in place.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]]
        Vector layer(s) or path(s) to vector layer(s).

    header : List[str]
        The header of the attributes to update in the table.

    attribute_table : List[List[Any]]
        The attributes to update in the table.

    match : str, optional
        The field to match on for updates. Default: 'fid'.

    Returns
    -------
    Union[str, List[str]]
        The path(s) to the output vector file(s).
    """
    assert isinstance(vector, (str, ogr.DataSource)), "vector must be a string or an ogr.DataSource object."
    assert isinstance(header, list), "header must be a list."
    assert isinstance(attribute_table, list), "attribute_table must be a list."
    assert len(header) == len(attribute_table[0]), "header and attribute_table must have the same number of columns."
    assert isinstance(match, (str, type(None))), "match must be a string or None."

    if match is not None:
        assert match in header, "match must be in header."

    match_idx = header.index(match) if match is not None else -1

    input_is_list = isinstance(vector, list)
    in_paths = utils_io._get_input_paths(vector, "vector")

    for in_vector in in_paths:
        ds = _open_vector(in_vector, writeable=True)
        layers = ds.GetLayerCount()

        for layer_index in range(layers):
            current_layer = ds.GetLayer(layer_index)
            current_layer.ResetReading()

            layer_defn = current_layer.GetLayerDefn()
            field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]

            # Create missing fields
            for field in header:
                if field not in field_names and field not in ('fid', match):
                    current_layer.CreateField(ogr.FieldDefn(field, ogr.OFTString))

            def update_feature(feat, fields, values):
                if feat is None:
                    return
                try:
                    for field_idx, field in enumerate(fields):
                        if field in (match, 'fid'):
                            continue
                        feat.SetField(field, values[field_idx])
                    current_layer.SetFeature(feat)
                except Exception:
                    pass

            if match is not None:
                for row in attribute_table:
                    try:
                        feat = current_layer.GetFeature(int(row[match_idx]))
                        if feat is not None:
                            update_feature(feat, header, row)
                    except (ValueError, TypeError):
                        continue
            else:
                for idx, feat in enumerate(current_layer):
                    if idx < len(attribute_table):
                        update_feature(feat, header, attribute_table[idx])

            current_layer.SyncToDisk()

        ds = None

    if input_is_list:
        return in_paths

    return in_paths[0]


def vector_delete_fields(
    vector: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
    fields: List[str],
) -> Union[str, List[str]]:
    """Deletes fields from a vector in place.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]]
        Vector layer(s) or path(s) to vector layer(s).

    fields : List[str]
        The fields to delete.

    Returns
    -------
    Union[str, List[str]]
        The path(s) to the output vector file(s).
    """
    assert isinstance(vector, (str, ogr.DataSource)), "vector must be a string or an ogr.DataSource object."
    assert isinstance(fields, list), "fields must be a list."

    input_is_list = isinstance(vector, list)
    in_paths = utils_io._get_input_paths(vector, "vector")

    for in_vector in in_paths:
        ref = _open_vector(in_vector, writeable=True)

        layers = ref.GetLayerCount()

        for layer_index in range(layers):
            layer = ref.GetLayer(layer_index)
            layer.ResetReading()

            layer_defn = layer.GetLayerDefn()
            field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]

            for field in fields:
                if field in field_names:
                    field_idx = layer.FindFieldIndex(field, 1)
                    layer.DeleteField(field_idx)

            layer.SyncToDisk()

        ref = None

    if input_is_list:
        return in_paths

    return in_paths[0]
