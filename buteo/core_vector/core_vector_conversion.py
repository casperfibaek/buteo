"""### Convert geometry composition. ###

Convert geometries from multiparts and singleparts and vice versa.
"""

# Standard library
from typing import Union, Optional, List

# External
from osgeo import ogr

# Internal
from buteo.utils import (
    utils_base,
    utils_gdal,
    utils_path,
    utils_io,
    utils_translate
)
from buteo.core_vector.core_vector_read import _open_vector, _vector_get_layer
from buteo.core_vector.core_vector_write import vector_create_empty_copy, vector_create_copy
from buteo.core_vector.core_vector_info import _get_basic_info_vector



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


def _convert_multitype(input_geom: ogr.Geometry, multitype: bool) -> Union[ogr.Geometry, List[ogr.Geometry]]:
    geom_type = input_geom.GetGeometryType()
    if utils_translate._check_geom_is_wkbgeom(geom_type):
        geom_type = utils_translate._convert_wkb_to_geomtype(geom_type)

    if multitype:
        converted_type = utils_translate._convert_singletype_int_to_multitype_int(geom_type)
    else:
        converted_type = utils_translate._convert_multitype_int_to_singletype_int(geom_type)

    if geom_type == converted_type:
        return input_geom

    converted_type = utils_translate._convert_geomtype_to_wkb(converted_type)

    if geom_type == ogr.wkbGeometryCollection:
        return input_geom

    if multitype is True:
        new_geom = ogr.Geometry(converted_type)
        new_geom.AddGeometry(input_geom)

        return new_geom

    else:
        separated = []
        for i in range(input_geom.GetGeometryCount()):
            part = input_geom.GetGeometryRef(i)
            if part is not None:
                separated.append(part.Clone())

        return separated


def vector_change_multitype(
    vector: Union[str, ogr.DataSource],
    multitype: bool,
    layer_name_or_id: Union[str, int] = 0,
    output_path: Optional[str] = None,
    *,
    prefix: str = "",
    suffix: str = "",
    overwrite: bool = False,
) -> str:
    """
    This function changes the type of a vector. Either from Multi to Single or vice versa.
    Examples: MultiPolygon to Polygon, MultiPoint to Point. Point to MultiPoint, etc.

    Parameters
    ----------
    vector : str or ogr.DataSource
        The vector to convert.
    multitype : bool
        If True, the output vector will be of the "multi"types. That is MultiPolygon, MultiPoint, etc.
        If False, the output vector will be of the "single"types. That is Polygon, Point, etc.
    layer_name_or_id : str or int, optional
        The name or index of the layer to convert. Default: 0
    output_path : str, optional
        The output path. Default: None (in-memory is created)
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
    utils_base._type_check(multitype, [bool], "multitype")
    utils_base._type_check(layer_name_or_id, [str, int], "layer_name_or_id")
    utils_base._type_check(output_path, [type(None), str], "output_path")
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

    curr_multi = _get_basic_info_vector(src_ds, layer_name_or_id)["geom_multi"]

    if curr_multi == multitype:
        return in_path

    if multitype is True:
        target_geom_type = utils_translate._convert_singletype_int_to_multitype_int(src_lyr.GetGeomType())
    else:
        target_geom_type = utils_translate._convert_multitype_int_to_singletype_int(src_lyr.GetGeomType())

    if utils_translate._check_geom_is_geomtype(target_geom_type):
        target_geom_type = utils_translate._convert_geomtype_to_wkb(target_geom_type)

    try:
        dst_ds_path = vector_create_empty_copy(
            in_path,
            out_path,
            layer_names_or_ids=layer_name_or_id,
            geom_type=target_geom_type,
            prefix=prefix,
            suffix=suffix,
            overwrite=overwrite,
        )
        dst_ds = _open_vector(dst_ds_path, writeable=True)
        dst_lyr = _vector_get_layer(dst_ds, layer_name_or_id)[0]
        dst_lyr_defn = dst_lyr.GetLayerDefn()

        if not isinstance(dst_lyr, ogr.Layer):
            raise ValueError("Could not open the layer.")

        src_lyr.ResetReading()
        feature_count = src_lyr.GetFeatureCount()

        for _i in range(feature_count):
            src_feat = src_lyr.GetNextFeature()
            src_geom = src_feat.GetGeometryRef()

            geom_list = _convert_multitype(src_geom, multitype)

            if not isinstance(geom_list, list):
                geom_list = [geom_list]

            for geom in geom_list:
                feat = ogr.Feature(dst_lyr_defn)
                feat.SetGeometry(geom)

                # Copy only the attributes from the source feature
                for fld_idx in range(src_feat.GetFieldCount()):
                    feat.SetField(fld_idx, src_feat.GetField(fld_idx))

                dst_lyr.CreateFeature(feat)
                feat = None

        dst_lyr.SyncToDisk()
        dst_lyr = None
        dst_ds = None

    except Exception as e:
        utils_io._delete_file(out_path)
        raise ValueError("Could not convert the vector.") from e

    return out_path


def vector_change_dimensionality(
    vector: Union[str, ogr.DataSource],
    z: Optional[bool] = None,
    m: Optional[bool] = None,
    layer_name_or_id: Union[str, int] = 0,
    output_path: Optional[str] = None,
    z_attribute: Optional[str] = None,
    m_attribute: Optional[str] = None,
    *,
    prefix: str = "",
    suffix: str = "",
    overwrite: bool = False,
) -> str:
    """
    Change the dimensionality of a vector. That is, add or remove Z and M values.

    Parameters
    ----------
    vector : str or ogr.DataSource
        The vector to convert.
    z : bool, optional
        If True, the output vector will be 3D.
        If False, the output vector will be 2D.
        If None, no changes. Default: None
    m : bool, optional
        If True, the output vector will have M (measure) values.
        If False, the output vector will not have M values.
        If None, no changes. Default: None
    layer_name_or_id : str or int, optional
        The name or index of the layer to convert. Default: 0
    output_path : str, optional
        The output path. Default: None (in-memory is created)
    z_attribute : str, optional
        The name of the attribute to use for Z values. If None, 0.0 is inserted. Default: None
    m_attribute : str, optional
        The name of the attribute to use for M values. If None, 0.0 is inserted. Default: None
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
    utils_base._type_check(z, [type(None), bool], "z")
    utils_base._type_check(m, [type(None), bool], "m")
    utils_base._type_check(layer_name_or_id, [str, int], "layer_name_or_id")
    utils_base._type_check(output_path, [type(None), str], "output_path")
    utils_base._type_check(z_attribute, [type(None), str], "z_attribute")
    utils_base._type_check(m_attribute, [type(None), str], "m_attribute")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(overwrite, [bool], "overwrite")

    in_path = utils_io._get_input_paths(vector, "vector")
    out_path = utils_io._get_output_paths(in_path, output_path, prefix=prefix, suffix=suffix)  # type: ignore

    utils_io._check_overwrite_policy(out_path, overwrite)
    utils_io._delete_if_required_list(out_path, overwrite)

    in_path = in_path[0]
    out_path = out_path[0]

    src_ds = _open_vector(in_path, writeable=False)
    src_lyr = _vector_get_layer(src_ds, layer_name_or_id)[0]
    layer_defn = src_lyr.GetLayerDefn()

    curr_geom_type = layer_defn.GetGeomType()
    if not utils_translate._check_geom_is_geomtype(curr_geom_type):
        curr_geom_type = utils_translate._convert_wkb_to_geomtype(curr_geom_type)

    hasZ = curr_geom_type in [1001, 1002, 1003, 1004, 1005, 1006, 3001, 3002, 3003, 3004, 3005, 3006]

    target_geom_type = curr_geom_type
    if z is False and hasZ:
        target_geom_type -= 1000
    elif z is True and not hasZ:
        target_geom_type += 1000

    hasM = curr_geom_type in [2001, 2002, 2003, 2004, 2005, 2006, 3001, 3002, 3003, 3004, 3005, 3006]

    if m is False and hasM:
        target_geom_type -= 2000
    elif m is True and not hasM:
        target_geom_type += 2000

    if target_geom_type == curr_geom_type:
        return in_path

    target_geom_type = utils_translate._convert_geomtype_to_wkb(target_geom_type) if utils_translate._check_geom_is_geomtype(target_geom_type) else target_geom_type

    if not isinstance(src_lyr, ogr.Layer):
        raise ValueError("Could not open the layer.")

    if z_attribute is not None:
        field_index = layer_defn.GetFieldIndex(z_attribute)
        if field_index == -1:
            raise ValueError("The z_attribute does not exist in the layer.")

    if m_attribute is not None:
        field_index = layer_defn.GetFieldIndex(m_attribute)
        if field_index == -1:
            raise ValueError("The m_attribute does not exist in the layer.")

    # try:
    dst_ds_path = vector_create_empty_copy(
        in_path,
        out_path,
        layer_names_or_ids=layer_name_or_id,
        geom_type=target_geom_type,
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

        if src_geom is None:
            continue

        geom_copy = src_geom.Clone()
        geom_copy.FlattenTo2D()

        if z is not None and z is True:
            if z_attribute is not None:
                z_value = src_feat.GetFieldAsDouble(z_attribute)
            else:
                z_value = 0.0
            geom_copy = _set_z_value_to_geometry(geom_copy, z_value)
        elif z is not None and z is False:
            geom_copy.FlattenTo2D()

        if m is not None and m is True:
            if m_attribute is not None:
                m_value = src_feat.GetFieldAsDouble(m_attribute)
            else:
                m_value = 0.0
            geom_copy = _set_m_value_to_geometry(geom_copy, m_value)
        elif m is not None and m is False:
            # Remove M values
            geom_copy.FlattenTo2D()

        feat = ogr.Feature(dst_lyr.GetLayerDefn())
        feat.SetFrom(src_feat)
        feat.SetGeometry(geom_copy)
        dst_lyr.CreateFeature(feat)
        feat = None

    dst_lyr.SyncToDisk()
    dst_lyr = None
    dst_ds = None

    # except Exception as e:
    #     utils_io._delete_file(out_path)
    #     raise ValueError("Could not convert the vector.") from e

    return out_path


def _set_z_value_to_geometry(
    geometry: ogr.Geometry,
    z_value: float,
) -> ogr.Geometry:
    """Set Z value for a geometry."""
    geom_type = geometry.GetGeometryType()
    if geom_type == ogr.wkbPoint or geom_type == ogr.wkbPointM:
        x = geometry.GetX()
        y = geometry.GetY()
        m = geometry.GetM() if hasattr(geometry, 'GetM') else None
        if m is not None:
            new_geom = ogr.Geometry(ogr.wkbPointZM)
            new_geom.AddPointZM(x, y, z_value, m)
        else:
            new_geom = ogr.Geometry(ogr.wkbPoint25D)
            new_geom.AddPoint(x, y, z_value)
        return new_geom
    elif geom_type in (ogr.wkbLineString, ogr.wkbLinearRing, ogr.wkbLineStringM):
        has_m = geom_type == ogr.wkbLineStringM
        if has_m:
            new_geom = ogr.Geometry(ogr.wkbLineStringZM)
        else:
            new_geom = ogr.Geometry(ogr.wkbLineString25D)

        for i in range(geometry.GetPointCount()):
            x, y = geometry.GetX(i), geometry.GetY(i)
            m = geometry.GetM(i) if has_m else None
            if m is not None:
                new_geom.AddPointZM(x, y, z_value, m)
            else:
                new_geom.AddPoint(x, y, z_value)

        if geom_type == ogr.wkbLinearRing:
            linear_ring = ogr.Geometry(ogr.wkbLinearRing)
            linear_ring.AssignSpatialReference(geometry.GetSpatialReference())
            linear_ring.AddGeometry(new_geom)
            return linear_ring
        return new_geom

    elif geom_type in (ogr.wkbPolygon, ogr.wkbPolygonM):
        has_m = geom_type == ogr.wkbPolygonM
        if has_m:
            new_geom = ogr.Geometry(ogr.wkbPolygonZM)
        else:
            new_geom = ogr.Geometry(ogr.wkbPolygon25D)

        for i in range(geometry.GetGeometryCount()):
            ring = geometry.GetGeometryRef(i)
            if has_m:
                ring_points = ogr.Geometry(ogr.wkbLineStringZM)
            else:
                ring_points = ogr.Geometry(ogr.wkbLineString25D)

            for j in range(ring.GetPointCount()):
                x, y = ring.GetX(j), ring.GetY(j)
                m = ring.GetM(j) if has_m else None
                if m is not None:
                    ring_points.AddPointZM(x, y, z_value, m)
                else:
                    ring_points.AddPoint(x, y, z_value)

            if has_m:
                new_ring = ogr.Geometry(ogr.wkbLinearRing | ogr.wkb25DBit | 0x40000000)
            else:
                new_ring = ogr.Geometry(ogr.wkbLinearRing | ogr.wkb25DBit)

            for j in range(ring_points.GetPointCount()):
                x, y = ring_points.GetX(j), ring_points.GetY(j)
                z = ring_points.GetZ(j)
                m = ring_points.GetM(j) if has_m else None
                if m is not None:
                    new_ring.AddPointZM(x, y, z, m)
                else:
                    new_ring.AddPoint(x, y, z)
            new_geom.AddGeometry(new_ring)
        return new_geom

    elif geom_type in (ogr.wkbMultiPoint, ogr.wkbMultiLineString, ogr.wkbMultiPolygon,
                      ogr.wkbGeometryCollection, ogr.wkbMultiPointM, ogr.wkbMultiLineStringM,
                      ogr.wkbMultiPolygonM, ogr.wkbGeometryCollectionM):
        has_m = geom_type & 0x40000000
        if has_m:
            new_type = geom_type | ogr.wkb25DBit | 0x40000000
        else:
            new_type = geom_type | ogr.wkb25DBit

        new_geom = ogr.Geometry(new_type)
        for i in range(geometry.GetGeometryCount()):
            sub_geom = geometry.GetGeometryRef(i)
            new_sub_geom = _set_z_value_to_geometry(sub_geom, z_value)
            new_geom.AddGeometry(new_sub_geom)
        return new_geom

    else:
        return geometry


# TODO: FIX - Causes errors
def _set_m_value_to_geometry(
    geometry: ogr.Geometry,
    m_value: float,
) -> ogr.Geometry:
    geom_type = geometry.GetGeometryType()

    if geom_type in (ogr.wkbPoint, ogr.wkbPoint25D):
        x, y = geometry.GetX(), geometry.GetY()
        z = geometry.GetZ() if geom_type == ogr.wkbPoint25D else 0.0
        new_type = ogr.wkbPointZM if geom_type == ogr.wkbPoint25D else ogr.wkbPointM
        new_geom = ogr.Geometry(new_type)
        new_geom.AddPointZM(x, y, z, m_value) if z else new_geom.AddPointM(x, y, m_value) # pylint: disable=expression-not-assigned

        return new_geom

    if geom_type in (ogr.wkbPolygon, ogr.wkbPolygon25D):
        has_z = geom_type == ogr.wkbPolygon25D
        new_geom = ogr.Geometry(ogr.wkbPolygonZM) if has_z else ogr.Geometry(ogr.wkbPolygonM)
        for i in range(geometry.GetGeometryCount()):
            ring = geometry.GetGeometryRef(i)
            if has_z:
                new_ring = ogr.Geometry(ogr.wkbLinearRing | ogr.wkb25DBit)
            else:
                new_ring = ogr.Geometry(ogr.wkbLinearRing)
            for j in range(ring.GetPointCount()):
                x, y, z = ring.GetPoint(j) if has_z else (*ring.GetPoint(j)[:2], 0.0)
                if has_z:
                    new_ring.AddPointZM(x, y, z, m_value)
                else:
                    new_ring.AddPointM(x, y, m_value)
            new_geom.AddGeometry(new_ring)
        return new_geom

    if geom_type in (ogr.wkbLineString, ogr.wkbLinearRing, ogr.wkbLineString25D):
        has_z = geom_type == ogr.wkbLineString25D
        is_ring = geom_type == ogr.wkbLinearRing

        if is_ring and has_z:
            new_type = ogr.wkbLinearRing | ogr.wkb25DBit | 0x40000000
        elif is_ring:
            new_type = ogr.wkbLinearRing | 0x40000000
        elif has_z:
            new_type = ogr.wkbLineStringZM
        else:
            new_type = ogr.wkbLineStringM

        new_geom = ogr.Geometry(new_type)
        for i in range(geometry.GetPointCount()):
            x, y, z = geometry.GetPoint(i) if has_z else (*geometry.GetPoint(i)[:2], 0.0)
            new_geom.AddPointZM(x, y, z, m_value) if has_z else new_geom.AddPointM(x, y, m_value) # pylint: disable=expression-not-assigned

        return new_geom

    multi_types = {
        ogr.wkbMultiPoint: ogr.wkbMultiPointM,
        ogr.wkbMultiLineString: ogr.wkbMultiLineStringM,
        ogr.wkbMultiPolygon: ogr.wkbMultiPolygonM,
        ogr.wkbGeometryCollection: ogr.wkbGeometryCollection
    }

    if geom_type in multi_types:
        new_geom = ogr.Geometry(multi_types[geom_type])
        for i in range(geometry.GetGeometryCount()):
            sub_geom = geometry.GetGeometryRef(i)
            new_sub_geom = _set_m_value_to_geometry(sub_geom, m_value)
            new_geom.AddGeometry(new_sub_geom)
        return new_geom

    return geometry


def vector_convert_geometry(
    vector: Union[str, ogr.DataSource],
    *,
    multitype: Optional[bool] = None,
    multipart: Optional[bool] = None,
    z: Optional[bool] = None,
    m: Optional[bool] = None,
    output_path: Optional[str] = None,
    layer_name_or_id: Union[str, int] = 0,
    z_attribute: Optional[str] = None,
    m_attribute: Optional[str] = None,
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
    multitype : bool, optional
        If True, the output vector will be of the "multi"types. That is MultiPolygon, MultiPoint, etc. Default: None
    multipart : bool, optional
        If True, the output vector will be multiparts. Features will be merged into a single feature with multiple geometries.
        The merged features will have the same attributes.
        If False, the output will be singleparts. If a multi feature is encountered, it will be split into multiple features.
        The split features will have the same attributes.
        If None, no changes. Default: None
    z : bool, optional
        If True, the output vector will be 3D. If None, no changes. Default: None
    m : bool, optional
        If True, the output vector will have M (measure) values. If None, no changes. Default: None
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
    utils_base._type_check(multitype, [type(None), bool], "multitype")
    utils_base._type_check(multipart, [type(None), bool], "multipart")
    utils_base._type_check(z, [type(None), bool], "z")
    utils_base._type_check(m, [type(None), bool], "m")
    utils_base._type_check(output_path, [type(None), str], "output_path")
    utils_base._type_check(layer_name_or_id, [str, int], "layer_name_or_id")
    utils_base._type_check(z_attribute, [type(None), str], "z_attribute")
    utils_base._type_check(m_attribute, [type(None), str], "m_attribute")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(overwrite, [bool], "overwrite")

    if multitype is False and multipart is True:
        raise ValueError("Cannot set both multitype to False and multipart to True.")

    in_path = utils_io._get_input_paths(vector, "vector")
    out_path = utils_io._get_output_paths(in_path, output_path, prefix=prefix, suffix=suffix) # type: ignore

    utils_io._check_overwrite_policy(out_path, overwrite)
    utils_io._delete_if_required_list(out_path, overwrite)

    in_path = in_path[0]
    out_path = out_path[0]

    ds = _open_vector(in_path, writeable=False)
    layer = _vector_get_layer(ds, layer_name_or_id)[0]

    if not isinstance(layer, ogr.Layer):
        raise ValueError("Could not open the layer.")

    meta = _get_basic_info_vector(ds, layer_name_or_id)

    # Check if conversion is necessary
    if multitype is True and meta["geom_multi"] is True:
        multitype = None
    elif multitype is False and meta["geom_multi"] is False:
        multitype = None

    if z is not None and z_attribute is None:
        if z is True and meta["geom_3d"] is True:
            z = None
        elif z is False and meta["geom_3d"] is False:
            z = None

    if m is not None and m_attribute is None:
        if m is True and meta["geom_m"] is True:
            m = None
        elif m is False and meta["geom_m"] is False:
            m = None

    # We don't check the other way, because not all the features are possibly multiparts
    if not check_vector_is_multipart(in_path, layer_name_or_id) and not multipart:
        multipart = None

    temp_path_1 = utils_path._get_temp_filepath("temp_path_1.fgb", add_timestamp=True, add_uuid=True)
    temp_path_2 = utils_path._get_temp_filepath("temp_path_2.fgb", add_timestamp=True, add_uuid=True)
    temp_path_3 = utils_path._get_temp_filepath("temp_path_3.fgb", add_timestamp=True, add_uuid=True)

    # First we convert the types:
    converted = in_path
    if multipart is True:
        converted = vector_singlepart_to_multipart(
            converted,
            layer_name_or_id=layer_name_or_id,
            output_path=temp_path_1,
            overwrite=overwrite,
        )
    elif multipart is False:
        converted = vector_multipart_to_singlepart(
            converted,
            layer_name_or_id=layer_name_or_id,
            output_path=temp_path_1,
            overwrite=overwrite,
        )

    # Then we convert the multitype
    if multitype is not None:
        converted = vector_change_multitype(
            converted,
            multitype,
            layer_name_or_id=layer_name_or_id,
            output_path=temp_path_2,
            overwrite=overwrite,
        )

    # Then we convert the dimensionality
    if z is not None or m is not None:
        converted = vector_change_dimensionality(
            converted,
            z=z,
            m=m,
            layer_name_or_id=layer_name_or_id,
            output_path=temp_path_3,
            z_attribute=z_attribute,
            m_attribute=m_attribute,
            overwrite=overwrite,
        )

    # Copy the final result to the output path
    output = vector_create_copy(
        converted,
        out_path=out_path,
        overwrite=overwrite,
    )


    utils_gdal.delete_dataset_if_in_memory_list([temp_path_1, temp_path_2, temp_path_3])

    if not isinstance(output, str):
        raise ValueError("Could not create the output vector.")

    return output
