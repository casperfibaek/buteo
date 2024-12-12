""" This module contains functions for resetting FIDs in a vector layer. """
# Standard library
from typing import Union, Optional

# External
from osgeo import ogr

# Internal
from buteo.utils import (
    utils_io,
    utils_base,
    utils_gdal,
)

from buteo.core_vector.core_vector_read import _open_vector, _vector_get_layer



def vector_reset_fids(
    vector: Union[str, ogr.DataSource],
    inplace: bool = False,
    out_path: Optional[str] = None,
    layer_name_or_id: Union[str, int] = 0,
    *,
    prefix: str = "",
    suffix: str = "",
    overwrite: bool = False,
) -> str:
    """
    Resets the FID Column of a vector layer to 0, 1, 2, 3, ...

    Parameters
    ----------
    vector : Union[str, ogr.DataSource]
        Vector layer or path to vector layer.
    inplace : bool, optional
        If True, will overwrite the input vector. Default: False.
    out_path : Optional[str], optional
        The path to the output vector. If None, an in-memory layer will be created. Default: None.
    layer_name_or_id : Union[str, int], optional
        The name or index of the layer to process. Default: 0.
    prefix : str, optional
        Prefix to add to the output vector. Default: "".
    suffix : str, optional
        Suffix to add to the output vector. Default: "".
    overwrite : bool, optional
        If True, will overwrite the output vector if it already exists. Default: False.

    Returns
    -------
    str
        Path to the output vector.
    """
    utils_base._type_check(vector, [str, ogr.DataSource], "vector")
    utils_base._type_check(inplace, [bool], "inplace")
    utils_base._type_check(out_path, [str, type(None)], "out_path")
    utils_base._type_check(layer_name_or_id, [str, int], "layer_name_or_id")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(overwrite, [bool], "overwrite")

    if inplace and out_path is not None:
        raise ValueError("Cannot set inplace=True and provide an out_path.")

    in_ds = _open_vector(vector, writeable=inplace)
    layer = _vector_get_layer(in_ds, layer_name_or_id)[0]
    feature_count = layer.GetFeatureCount()

    if not isinstance(layer, ogr.Layer):
        raise ValueError("Could not get layer from vector.")

    in_paths = utils_io._get_input_paths(vector, "vector")
    out_paths = utils_io._get_output_paths(in_paths, out_path, prefix=prefix, suffix=suffix) # type: ignore

    utils_io._check_overwrite_policy(out_paths, overwrite)
    utils_io._delete_if_required_list(out_paths, overwrite)

    in_path = in_paths[0]
    out_path = out_paths[0]

    if inplace:
        # Collect all features
        layer.ResetReading()
        features = []
        layer_defn = layer.GetLayerDefn()
        field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]

        for feature in layer:
            features.append(feature.Clone())

        # Delete all features from the layer
        for feature in features:
            layer.DeleteFeature(feature.GetFID())

        # Assign new FIDs and add features back to the layer
        for idx, feature in enumerate(features):
            if "fid" in field_names:
                feature.SetField("fid", str(idx))
            feature.SetFID(idx)
            layer.CreateFeature(feature)

        layer.ResetReading()
        layer.SyncToDisk()
        in_ds.FlushCache()
        in_ds = None

        return in_path

    driver_name = utils_gdal._get_vector_driver_name_from_path(out_path)
    driver = ogr.GetDriverByName(driver_name)

    if driver is None:
        raise ValueError(f"Could not find driver for {driver_name}")

    out_ds = driver.CreateDataSource(out_path)

    layer_defn = layer.GetLayerDefn()
    out_layer = out_ds.CreateLayer(
        layer.GetName(),
        layer.GetSpatialRef(),
        layer_defn.GetGeomType(),
    )

    for field_index in range(layer_defn.GetFieldCount()):
        field_defn = layer_defn.GetFieldDefn(field_index)
        out_layer.CreateField(field_defn)

    layer.ResetReading()
    fids = []

    for _ in range(feature_count):
        feature = layer.GetNextFeature()
        fids.append(feature.GetFID())

    layer.ResetReading()
    fids = sorted(fids)

    for _ in range(feature_count):
        feature = layer.GetNextFeature()
        current_fid = feature.GetFID()
        target_fid = fids.index(current_fid)

        feature.SetFID(target_fid)
        out_layer.CreateFeature(feature)

    out_layer.SyncToDisk()
    out_layer.ResetReading()
    out_layer = None

    out_ds.FlushCache()

    in_ds = None
    out_ds = None

    return out_path


def vector_create_attribute_from_fid(
    vector: Union[str, ogr.DataSource],
    inplace: bool = False,
    out_path: Optional[str] = None,
    layer_name_or_id: Union[str, int] = 0,
    *,
    attribute_name: str = "id",
    prefix: str = "",
    suffix: str = "",
    overwrite: bool = False,
) -> str:
    """Creates an attribute from the FID field in a vector.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource]
        Vector layer or path to vector layer.
    inplace : bool, optional
        If True, will overwrite the input vector. Default: False.
    out_path : Optional[str], optional
        The path to the output vector. If None, an in-memory layer will be created. Default: None.
    layer_name_or_id : Union[str, int], optional
        The name or index of the layer to process. Default: 0.
    attribute_name : str, optional
        The name of the attribute to create. Default: "id".
    prefix : str, optional
        Prefix to add to the output vector. Default: "".
    suffix : str, optional
        Suffix to add to the output vector. Default: "".
    overwrite : bool, optional
        If True, will overwrite the output vector if it already exists. Default: False.

    Returns
    -------
    str
        Path to the output vector.
    """
    utils_base._type_check(vector, [str, ogr.DataSource], "vector")
    utils_base._type_check(inplace, [bool], "inplace")
    utils_base._type_check(out_path, [str, type(None)], "out_path")
    utils_base._type_check(layer_name_or_id, [str, int], "layer_name_or_id")
    utils_base._type_check(attribute_name, [str], "attribute_name")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(overwrite, [bool], "overwrite")

    if inplace and out_path is not None:
        raise ValueError("Cannot set inplace=True and provide an out_path.")

    in_ds = _open_vector(vector, writeable=inplace)
    layer = _vector_get_layer(in_ds, layer_name_or_id)[0]

    if not isinstance(layer, ogr.Layer):
        raise ValueError("Could not get layer from vector.")

    in_paths = utils_io._get_input_paths(vector, "vector")
    out_paths = utils_io._get_output_paths(in_paths, out_path, prefix=prefix, suffix=suffix) # type: ignore

    utils_io._check_overwrite_policy(out_paths, overwrite)
    utils_io._delete_if_required_list(out_paths, overwrite)

    in_path = in_paths[0]
    out_path = out_paths[0]

    if inplace:
        layer_defn = layer.GetLayerDefn()
        field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]

        if attribute_name not in field_names:
            field = ogr.FieldDefn(attribute_name, ogr.OFTInteger)
            layer.CreateField(field)

        layer.ResetReading()
        for _ in range(layer.GetFeatureCount()):
            feature = layer.GetNextFeature()
            feature.SetField(attribute_name, feature.GetFID())
            layer.SetFeature(feature)

        layer.SyncToDisk()
        in_ds = None
        return in_path

    driver_name = utils_gdal._get_vector_driver_name_from_path(out_path)
    driver = ogr.GetDriverByName(driver_name)

    if driver is None:
        raise ValueError(f"Could not find driver for {driver_name}")

    out_ds = driver.CreateDataSource(out_path)

    layer_defn = layer.GetLayerDefn()
    out_layer = out_ds.CreateLayer(
        layer.GetName(),
        layer.GetSpatialRef(),
        layer_defn.GetGeomType(),
    )

    for field_index in range(layer_defn.GetFieldCount()):
        field_defn = layer_defn.GetFieldDefn(field_index)
        out_layer.CreateField(field_defn)

    field = ogr.FieldDefn(attribute_name, ogr.OFTInteger)
    out_layer.CreateField(field)

    layer.ResetReading()
    field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]
    if attribute_name in field_names:
        field_names.remove(attribute_name)

    for feature in layer:
        out_feature = ogr.Feature(out_layer.GetLayerDefn())
        out_feature.SetGeometry(feature.GetGeometryRef())
        out_feature.SetField(attribute_name, feature.GetFID())

        for field_name in field_names:
            out_feature.SetField(field_name, feature.GetField(field_name))

        out_layer.CreateFeature(out_feature)

    out_layer.SyncToDisk()
    in_ds = None
    out_ds = None

    return out_path
