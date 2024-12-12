# Standard library
from typing import Union, Optional,Callable

# External
from osgeo import ogr

# Internal
from buteo.utils import (
    utils_base,
    utils_gdal,
    utils_path,
    utils_io,
)

from buteo.core_vector.core_vector_info import get_metadata_vector
from buteo.core_vector.core_vector_read import _open_vector, _vector_get_layer



def vector_filter_by_function(
    vector: Union[ogr.DataSource, str],
    out_path: Optional[str] = None,
    inplace: bool = False,
    filter_function_attr: Optional[Callable] = None,
    filter_function_geom: Optional[Callable] = None,
    layer_name_or_id: Union[str, int] = 0,
    prefix: str = "",
    suffix: str = "",
    overwrite: bool = False,
) -> str:
    """
    Filters a vector based on functions that return True or False for each feature.

    Parameters
    ----------
    vector : ogr.DataSource or str
        The vector to filter.
    out_path : str, optional
        The output path. Default: None (in-memory is created)
    inplace : bool, optional
        If True, the input vector is modified. Default: False
    filter_function_attr : function, optional
        A function that takes a key-val dictionary of fieldnames and values and returns True or False.
        Default: None
    filter_function_geom : function, optional
        A function that takes a geometry object and returns True or False.
        Default: None
    layer_name_or_id : str or int, optional
        The name or index of the layer to filter. Default: 0
    prefix : str, optional
        Prefix to add to output path. Default: ""
    suffix : str, optional
        Suffix to add to output path. Default: ""
    overwrite : bool, optional
        If True, overwrites existing files. Default: False

    Returns
    -------
    str
        The path to the filtered vector

    Examples
    --------
    >>> from buteo.core_vector.core_vector_filter import vector_filter_by_function
    >>> vector_filter_by_function("input.shp", "output.shp", filter_function_attr=lambda x: x["value"] > 10)
    "output.shp"
    """
    utils_base._type_check(vector, [str, ogr.DataSource], "vector")
    utils_base._type_check(out_path, [type(None), str], "out_path")
    utils_base._type_check(inplace, [bool], "inplace")
    utils_base._type_check(filter_function_attr, [type(None), type(utils_base._type_check)], "filter_function_attr")
    utils_base._type_check(filter_function_geom, [type(None), type(utils_base._type_check)], "filter_function_geom")
    utils_base._type_check(layer_name_or_id, [type(None), str, int], "layer_name_or_id")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(overwrite, [bool], "overwrite")

    if filter_function_attr is None and filter_function_geom is None:
        raise ValueError("At least one filter function must be provided.")

    if inplace and out_path is not None:
        raise ValueError("Cannot set inplace to True and provide an out_path.")

    ds = _open_vector(vector, writeable=True if inplace else False)
    layer = _vector_get_layer(ds, layer_name_or_id)[0]

    if not isinstance(layer, ogr.Layer):
        raise ValueError("Could not get layer from vector.")

    metadata = get_metadata_vector(vector, layer_name_or_id=layer_name_or_id)
    metadata_layer = metadata["layers"][0]

    field_names = metadata_layer["field_names"]

    if inplace:
        # loop features
        layer.ResetReading()
        feature_count = layer.GetFeatureCount()
        for _ in range(feature_count):
            keep = True

            feature = layer.GetNextFeature()

            # loop fields
            if filter_function_attr is not None:
                field_dict = {}
                for field_name in field_names:
                    field_dict[field_name] = feature.GetField(field_name)

                keep = filter_function_attr(field_dict)

            # loop geometry
            if filter_function_geom is not None:
                geom = feature.GetGeometryRef()
                keep = filter_function_geom(geom)

            if not keep:
                layer.DeleteFeature(feature.GetFID())

        layer.SyncToDisk()
        layer.ResetReading()
        layer = None

        ds.FlushCache()
        ds = None

        return utils_gdal._get_path_from_dataset(vector, "vector")

    if out_path is None:
        out_path = utils_path._get_temp_filepath(vector, prefix=prefix, suffix=suffix)

    utils_io._check_overwrite_policy([out_path], overwrite)
    utils_io._delete_if_required_list([out_path], overwrite)

    driver_name = utils_gdal._get_vector_driver_name_from_path(out_path)
    driver = ogr.GetDriverByName(driver_name)

    datasource_destination = driver.CreateDataSource(out_path)

    layer_destination = datasource_destination.CreateLayer(
        layer.GetName(),
        layer.GetSpatialRef(),
        layer.GetGeomType(),
    )

    layer_field_defn = layer.GetLayerDefn()
    for field_name in field_names:
        field_defn = layer_field_defn.GetFieldDefn(layer_field_defn.GetFieldIndex(field_name))
        layer_destination.CreateField(field_defn)

    layer.ResetReading()

    for _ in range(layer.GetFeatureCount()):
        feature = layer.GetNextFeature()

        if filter_function_attr is not None:
            field_dict = {}
            for field_name in field_names:
                field_dict[field_name] = feature.GetField(field_name)

            if not filter_function_attr(field_dict):
                continue

        if filter_function_geom is not None:
            geom = feature.GetGeometryRef()
            if not filter_function_geom(geom):
                continue

        layer_destination.CreateFeature(feature.Clone())

    layer_destination.SyncToDisk()
    layer_destination.ResetReading()
    layer_destination = None

    datasource_destination.FlushCache()
    datasource_destination = None

    return out_path
