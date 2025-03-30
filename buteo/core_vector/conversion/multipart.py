"""### Convert between multipart and singlepart geometries. ###"""

# Standard library
from typing import Union, Optional

# External
from osgeo import ogr

# Internal
from buteo.utils import (
    utils_base,
    utils_io,
)
from buteo.core_vector.core_vector_read import _open_vector, _vector_get_layer
from buteo.core_vector.core_vector_write import vector_create_empty_copy


def check_vector_is_multipart(
    vector: Union[str, ogr.DataSource],
    layer_name_or_id: Union[str, int] = 0,
) -> bool:
    """
    Checks if a vector is multipart. That if it contains features with multiple geometries.

    Parameters
    ----------
    vector : str or ogr.DataSource
        The vector to check.
    layer_name_or_id : str or int, optional
        The name or index of the layer to check. Default: 0

    Returns
    -------
    bool
        True if the vector is multipart, False otherwise.
    """
    utils_base._type_check(vector, [str, ogr.DataSource], "vector")
    utils_base._type_check(layer_name_or_id, [str, int], "layer_name_or_id")

    ds = _open_vector(vector, writeable=False)
    layer = _vector_get_layer(ds, layer_name_or_id)[0]

    if not isinstance(layer, ogr.Layer):
        raise ValueError("Could not open the layer.")

    layer.ResetReading()
    feature_count = layer.GetFeatureCount()

    for _i in range(feature_count):
        feature = layer.GetNextFeature()
        geom = feature.GetGeometryRef()
        if geom.GetGeometryCount() > 1:
            return True

    return False


def vector_multipart_to_singlepart(
    vector: Union[str, ogr.DataSource],
    layer_name_or_id: Union[str, int] = 0,
    output_path: Optional[str] = None,
    output_multitype: Optional[bool] = None,
    *,
    prefix: str = "",
    suffix: str = "",
    overwrite: bool = False,
) -> str:
    """
    Convert a multipart vector to a singlepart vector. That is, split features with multiple geometries into multiple features.
    Copies the attributes from the original feature to all the new features.

    Parameters
    ----------
    vector : str or ogr.DataSource
        The vector to convert.
    layer_name_or_id : str or int, optional
        The name or index of the layer to convert. Default: 0
    output_path : str, optional
        The output path. Default: None (in-memory is created)
    output_multitype : bool, optional
        If True, the output vector will be of the "multi"types. That is MultiPolygon, MultiPoint, etc. Default: None
        If False, the output vector will be of the "single"types. That is Polygon, Point, etc.
        If None, no changes. Default: None
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
    utils_base._type_check(layer_name_or_id, [str, int], "layer_name_or_id")
    utils_base._type_check(output_path, [type(None), str], "output_path")
    utils_base._type_check(output_multitype, [type(None), bool], "output_multitype")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(overwrite, [bool], "overwrite")

    in_path = utils_io._get_input_paths(vector, "vector")
    out_path = utils_io._get_output_paths(in_path, output_path, prefix=prefix, suffix=suffix) # type: ignore

    utils_io._check_overwrite_policy(out_path, overwrite)
    utils_io._delete_if_required_list(out_path, overwrite)

    in_path = in_path[0]
    out_path = out_path[0]

    src_ds = _open_vector(in_path, writeable=False)
    src_lyr = _vector_get_layer(src_ds, layer_name_or_id)[0]

    if not isinstance(src_lyr, ogr.Layer):
        raise ValueError("Could not open the layer.")

    try:
        dst_ds_path = vector_create_empty_copy(
            in_path,
            out_path,
            layer_names_or_ids=layer_name_or_id,
            prefix=prefix,
            suffix=suffix,
            overwrite=overwrite,
        )
        dst_ds = _open_vector(dst_ds_path, writeable=True)
        dst_lyr = _vector_get_layer(dst_ds, layer_name_or_id)[0]

        if not isinstance(dst_lyr, ogr.Layer):
            raise ValueError("Could not open the layer.")

        src_lyr.ResetReading()
        feature_count = src_lyr.GetFeatureCount()

        for _i in range(feature_count):
            src_feat = src_lyr.GetNextFeature()
            src_geom = src_feat.GetGeometryRef()

            geom_count = src_geom.GetGeometryCount()

            if geom_count == 1:
                dst_lyr.CreateFeature(src_feat)
                continue

            for j in range(geom_count):
                part = src_geom.GetGeometryRef(j)
                feat = ogr.Feature(dst_lyr.GetLayerDefn())
                feat.SetGeometry(part)
                feat.SetFrom(src_feat)
                dst_lyr.CreateFeature(feat)
                feat = None

        dst_lyr.SyncToDisk()
        dst_lyr = None
        dst_ds = None

    except Exception as e:
        utils_io._delete_file(out_path)
        raise ValueError("Could not convert the vector.") from e

    return out_path


def vector_singlepart_to_multipart(
    vector: Union[str, ogr.DataSource],
    layer_name_or_id: Union[str, int] = 0,
    output_path: Optional[str] = None,
    attribute: Optional[str] = None,
    *,
    prefix: str = "",
    suffix: str = "",
    overwrite: bool = False,
) -> str:
    """
    Convert a singlepart vector to a multipart vector. That is, merge features with the same attributes into a single feature with multiple geometries.

    Parameters
    ----------
    vector : str or ogr.DataSource
        The vector to convert.
    layer_name_or_id : str or int, optional
        The name or index of the layer to convert. Default: 0
    output_path : str, optional
        The output path. Default: None (in-memory is created)
    attribute : str, optional
        The attribute to use for merging features. If None, all features are merged. Default: None
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
    utils_base._type_check(layer_name_or_id, [str, int], "layer_name_or_id")
    utils_base._type_check(output_path, [type(None), str], "output_path")
    utils_base._type_check(attribute, [type(None), str], "attribute")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(overwrite, [bool], "overwrite")

    in_path = utils_io._get_input_paths(vector, "vector")
    out_path = utils_io._get_output_paths(in_path, output_path, prefix=prefix, suffix=suffix) # type: ignore

    utils_io._check_overwrite_policy(out_path, overwrite)
    utils_io._delete_if_required_list(out_path, overwrite)

    in_path = in_path[0]
    out_path = out_path[0]

    src_ds = _open_vector(in_path, writeable=False)
    src_lyr = _vector_get_layer(src_ds, layer_name_or_id)[0]

    if not isinstance(src_lyr, ogr.Layer):
        raise ValueError("Could not open the layer.")

    if attribute is not None:
        layer_defn = src_lyr.GetLayerDefn()
        field_index = layer_defn.GetFieldIndex(attribute)
        if field_index == -1:
            raise ValueError("The attribute does not exist in the layer.")

    try:
        dst_ds_path = vector_create_empty_copy(
            in_path,
            out_path,
            layer_names_or_ids=layer_name_or_id,
            prefix=prefix,
            suffix=suffix,
            overwrite=overwrite,
        )
        dst_ds = _open_vector(dst_ds_path, writeable=True)
        dst_lyr = _vector_get_layer(dst_ds, layer_name_or_id)[0]

        if not isinstance(dst_lyr, ogr.Layer):
            raise ValueError("Could not open the layer.")

        src_lyr.ResetReading()
        feature_count = src_lyr.GetFeatureCount()

        if attribute is None:
            features = {}
            for _i in range(feature_count):
                src_feat = src_lyr.GetNextFeature()
                geom = src_feat.GetGeometryRef()
                key_i = geom.ExportToWkt()
                if key_i in features:
                    features[key_i].append(src_feat)
                else:
                    features[key_i] = [src_feat]

            for key, feature_list in features.items():
                geom = ogr.CreateGeometryFromWkt(key)
                feat = ogr.Feature(dst_lyr.GetLayerDefn())
                feat.SetGeometry(geom)
                for src_feat in feature_list:
                    feat.SetFrom(src_feat)
                dst_lyr.CreateFeature(feat)
                feat = None

        else:
            features = {}
            for _i in range(feature_count):
                src_feat = src_lyr.GetNextFeature()
                key_i = src_feat.GetField(attribute)
                if key_i in features:
                    features[key_i].append(src_feat)
                else:
                    features[key_i] = [src_feat]

            for _j, features_list in features.items():
                geom = ogr.Geometry(ogr.wkbGeometryCollection)
                for src_feat in features_list:
                    geom.AddGeometry(src_feat.GetGeometryRef())
                feat = ogr.Feature(dst_lyr.GetLayerDefn())
                feat.SetGeometry(geom)
                for src_feat in features_list:
                    feat.SetFrom(src_feat)
                dst_lyr.CreateFeature(feat)
                feat = None

        dst_lyr.SyncToDisk()
        dst_lyr = None
        dst_ds = None

    except Exception as e:
        utils_io._delete_file(out_path)
        raise ValueError("Could not convert the vector.") from e

    return out_path
