"""### Convert vector geometry dimensionality (2D/3D). ###"""

# Standard library
from typing import Union, Optional

# External
from osgeo import ogr

# Internal
from buteo.utils import (
    utils_base,
    utils_io,
    utils_translate,
)
from buteo.core_vector.core_vector_read import _open_vector, _vector_get_layer
from buteo.core_vector.core_vector_write import vector_create_empty_copy


def _set_z_value_to_geometry(
    geometry: ogr.Geometry,
    z_value: float,
) -> ogr.Geometry:
    """
    Set Z value for a geometry.
    
    Parameters
    ----------
    geometry : ogr.Geometry
        The geometry to add Z values to.
    z_value : float
        The Z value to set.
        
    Returns
    -------
    ogr.Geometry
        The geometry with Z values added.
    """
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


def _set_m_value_to_geometry(
    geometry: ogr.Geometry,
    m_value: float,
) -> ogr.Geometry:
    """
    Set M value for a geometry.
    
    Parameters
    ----------
    geometry : ogr.Geometry
        The geometry to add M values to.
    m_value : float
        The M value to set.
        
    Returns
    -------
    ogr.Geometry
        The geometry with M values added.
    """
    geom_type = geometry.GetGeometryType()

    if geom_type in (ogr.wkbPoint, ogr.wkbPoint25D):
        x, y = geometry.GetX(), geometry.GetY()
        z = geometry.GetZ() if geom_type == ogr.wkbPoint25D else 0.0
        new_type = ogr.wkbPointZM if geom_type == ogr.wkbPoint25D else ogr.wkbPointM
        new_geom = ogr.Geometry(new_type)
        if z:
            new_geom.AddPointZM(x, y, z, m_value)
        else:
            new_geom.AddPointM(x, y, m_value)

        return new_geom

    if geom_type in (ogr.wkbPolygon, ogr.wkbPolygon25D):
        has_z = geom_type == ogr.wkbPolygon25D
        new_geom = ogr.Geometry(ogr.wkbPolygonZM) if has_z else ogr.Geometry(ogr.wkbPolygonM)
        for i in range(geometry.GetGeometryCount()):
            ring = geometry.GetGeometryRef(i)
            if has_z:
                new_ring = ogr.Geometry(ogr.wkbLinearRing | ogr.wkb25DBit | 0x40000000)
            else:
                new_ring = ogr.Geometry(ogr.wkbLinearRing | 0x40000000)
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
            if has_z:
                new_geom.AddPointZM(x, y, z, m_value)
            else:
                new_geom.AddPointM(x, y, m_value)

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

    except Exception as e:
        utils_io._delete_file(out_path)
        raise ValueError("Could not convert the vector.") from e

    return out_path
