"""### Convert single and multi geometry types. ###"""

# Standard library
from typing import Union, Optional, List

# External
from osgeo import ogr

# Internal
from buteo.utils import (
    utils_base,
    utils_io,
    utils_translate
)
from buteo.core_vector.core_vector_read import _open_vector, _vector_get_layer
from buteo.core_vector.core_vector_write import vector_create_empty_copy
from buteo.core_vector.core_vector_info import _get_basic_info_vector


def _convert_multitype(input_geom: ogr.Geometry, multitype: bool) -> Union[ogr.Geometry, List[ogr.Geometry]]:
    """
    Convert a geometry to a multitype or singletype.

    Parameters
    ----------
    input_geom : ogr.Geometry
        The geometry to convert.
    multitype : bool
        If True, convert to a multitype. If False, convert to a singletype.

    Returns
    -------
    Union[ogr.Geometry, List[ogr.Geometry]]
        The converted geometry, or a list of geometries if converting to singletype.
    """
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
