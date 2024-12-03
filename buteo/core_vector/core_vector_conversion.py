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
)
from buteo.vector import core_vector
from buteo.vector.metadata import _vector_to_metadata




def _singlepart_to_multipart(
    vector: Union[str, ogr.DataSource],
    out_path: Optional[str] = None,
    overwrite: bool = True,
    add_index: bool = True,
    process_layer: int = -1,
) -> str:
    """Internal."""
    assert isinstance(vector, (ogr.DataSource, str)), "Invalid input vector"
    assert utils_gdal._check_is_vector(vector), "Invalid input vector"

    if out_path is None:
        out_path = utils_path._get_temp_filepath(vector)

    assert utils_path._check_is_valid_output_filepath(out_path, overwrite=overwrite), "Invalid output path"

    ref = core_vector._vector_open(vector)

    out_format = utils_gdal._get_path_from_dataset(out_path, dataset_type="vector")
    driver = ogr.GetDriverByName(out_format)

    utils_io._delete_if_required(out_path, overwrite)

    metadata = _vector_to_metadata(ref)
    destination = driver.CreateDataSource(out_path)

    for index, layer_meta in enumerate(metadata["layers"]):
        if process_layer != -1 and index != process_layer:
            continue

        name = layer_meta["layer_name"]
        geom = layer_meta["column_geom"]

        sql = f"SELECT ST_Collect({geom}) AS geom FROM {name};"

        result = ref.ExecuteSQL(sql, dialect="SQLITE")
        destination.CopyLayer(result, name, ["OVERWRITE=YES"])

    if add_index:
        core_vector.vector_add_index(destination)

    destination.FlushCache()

    return out_path


def _multipart_to_singlepart(
    vector: Union[str, ogr.DataSource],
    out_path: Optional[str] = None,
    copy_attributes: bool = False,
    overwrite: bool = True,
    add_index: bool = True,
    process_layer: int = -1,
    verbose: bool = False,
) -> str:
    """Internal."""
    assert isinstance(vector, (ogr.DataSource, str)), "Invalid input vector"
    assert utils_gdal._check_is_vector(vector), "Invalid input vector"

    if out_path is None:
        out_path = utils_path._get_temp_filepath(vector)

    assert utils_path._check_is_valid_output_filepath(out_path, overwrite=overwrite), "Invalid output path"

    ref = core_vector._vector_open(vector)

    out_format = utils_gdal._get_path_from_dataset(out_path, dataset_type="vector")
    driver = ogr.GetDriverByName(out_format)

    utils_io._delete_if_required(out_path, overwrite)

    metadata = _vector_to_metadata(ref)

    utils_io._delete_if_required(out_path, overwrite)

    destination = driver.CreateDataSource(out_path)

    for index, layer_meta in enumerate(metadata["layers"]):
        if process_layer != -1 and index != process_layer:
            continue

        if verbose == 1:
            layer_name = layer_meta["layer_name"]
            print(f"Splitting layer: {layer_name}")

        target_unknown = False

        if layer_meta["geom_type_ogr"] == 4:  # MultiPoint
            target_type = 1  # Point
        elif layer_meta["geom_type_ogr"] == 5:  # MultiLineString
            target_type = 2  # LineString
        elif layer_meta["geom_type_ogr"] == 6:  # MultiPolygon
            target_type = 3  # Polygon
        elif layer_meta["geom_type_ogr"] == 1004:  # MultiPoint (z)
            target_type = 1001  # Point (z)
        elif layer_meta["geom_type_ogr"] == 1005:  # MultiLineString (z)
            target_type = 1002  # LineString (z)
        elif layer_meta["geom_type_ogr"] == 1006:  # MultiPolygon (z)
            target_type = 1003  # Polygon (z)
        elif layer_meta["geom_type_ogr"] == 2004:  # MultiPoint (m)
            target_type = 2001  # Point (m)
        elif layer_meta["geom_type_ogr"] == 2005:  # MultiLineString (m)
            target_type = 2002  # LineString (m)
        elif layer_meta["geom_type_ogr"] == 2006:  # MultiPolygon (m)
            target_type = 2003  # Polygon (m)
        elif layer_meta["geom_type_ogr"] == 3004:  # MultiPoint (zm)
            target_type = 3001  # Point (m)
        elif layer_meta["geom_type_ogr"] == 3005:  # MultiLineString (zm)
            target_type = 3002  # LineString (m)
        elif layer_meta["geom_type_ogr"] == 3006:  # MultiPolygon (zm)
            target_type = 3003  # Polygon (m)
        else:
            target_unknown = True
            target_type = layer_meta["geom_type_ogr"]

        destination_layer = destination.CreateLayer(
            layer_meta["layer_name"], layer_meta["projection_osr"], target_type
        )
        layer_defn = destination_layer.GetLayerDefn()
        field_count = layer_meta["field_count"]

        original_target = ref.GetLayerByIndex(index)
        feature_count = original_target.GetFeatureCount()

        if copy_attributes:
            first_feature = original_target.GetNextFeature()
            original_target.ResetReading()

            if verbose == 1:
                print("Creating attribute fields")

            for field_id in range(field_count):
                field_defn = first_feature.GetFieldDefnRef(field_id)

                fname = field_defn.GetName()
                ftype = field_defn.GetTypeName()
                fwidth = field_defn.GetWidth()
                fprecision = field_defn.GetPrecision()

                if ftype == "String" or ftype == "Date":
                    fielddefn = ogr.FieldDefn(fname, ogr.OFTString)
                    fielddefn.SetWidth(fwidth)
                elif ftype == "Real":
                    fielddefn = ogr.FieldDefn(fname, ogr.OFTReal)
                    fielddefn.SetWidth(fwidth)
                    fielddefn.SetPrecision(fprecision)
                else:
                    fielddefn = ogr.FieldDefn(fname, ogr.OFTInteger)

                destination_layer.CreateField(fielddefn)

        for _ in range(feature_count):
            feature = original_target.GetNextFeature()
            geom = feature.GetGeometryRef()

            if target_unknown:
                out_feat = ogr.Feature(layer_defn)
                out_feat.SetGeometry(geom)

                if copy_attributes:
                    for field_id in range(field_count):
                        values = feature.GetField(field_id)
                        out_feat.SetField(field_id, values)

                destination_layer.CreateFeature(out_feat)

            for geom_part in geom:
                out_feat = ogr.Feature(layer_defn)
                out_feat.SetGeometry(geom_part)

                if copy_attributes:
                    for field_id in range(field_count):
                        values = feature.GetField(field_id)
                        out_feat.SetField(field_id, values)

                destination_layer.CreateFeature(out_feat)

            if verbose:
                utils_base.progress(_, feature_count - 1, "Splitting.")

    if add_index:
        core_vector.vector_add_index(destination)

    return out_path


def vector_singlepart_to_multipart(
    vector: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
    out_path: Optional[str] = None,
    add_index: bool = True,
    process_layer: int = -1,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    overwrite: bool = True,
    allow_lists: bool = True,
) -> Union[str, List[str]]:
    """Converts a singlepart vector to multipart.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]]
        The vector(s) to convert.

    out_path : Optional[str], optional
        The path(s) to the output vector. If None a memory output is produced. Default: None

    add_index : bool, optional
        Add an geospatial index to the output vector. Default: True

    process_layer : int, optional
        The layer index to process. Default: -1

    prefix : str, optional
        The prefix to add to the layer name. Default: ""

    suffix : str, optional
        The suffix to add to the layer name. Default: ""

    add_uuid : bool, optional
        Add a UUID field to the output vector. Default: False

    overwrite : bool, optional
        Overwrite the output vector if it already exists. Default: True

    allow_lists : bool, optional
        Allow lists of vectors as input. Default: True

    Returns
    -------
    Union[str, List[str]]
        The path(s) to the output vector(s).
    """
    utils_base._type_check(vector, [str, ogr.DataSource, [str, ogr.DataSource]], "vector")
    utils_base._type_check(out_path, [str, [str], None], "out_path")
    utils_base._type_check(overwrite, [bool], "overwrite")
    utils_base._type_check(add_index, [bool], "add_index")
    utils_base._type_check(process_layer, [int], "process_layer")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(allow_lists, [bool], "allow_lists")

    if not allow_lists and isinstance(vector, list):
        raise ValueError("Vector cannot be a list when allow_lists is False.")

    vector_list = utils_base._get_variable_as_list(vector)

    assert utils_gdal._check_is_vector_list(vector_list), f"Vector is not a list of vectors. {vector_list}"

    path_list = utils_io._get_output_paths(
        vector_list,
        out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        overwrite=overwrite,
    )

    output = []
    for index, in_vector in enumerate(vector_list):
        output.append(
            _singlepart_to_multipart(
                in_vector,
                out_path=path_list[index],
                overwrite=overwrite,
                add_index=add_index,
                process_layer=process_layer,
            )
        )

    if isinstance(vector, list):
        return output[0]

    return output


def vector_multipart_to_singlepart(
    vector: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
    out_path: Optional[str] = None,
    overwrite: bool = True,
    add_index: bool = True,
    process_layer: int = -1,
    verbose: bool = False,
    prefix: str ="",
    suffix: str = "",
    add_uuid: bool = False,
    allow_lists: bool = True,
) -> Union[str, List[str]]:
    """Converts a multipart vector to singlepart.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]]
        The vector(s) to convert.

    out_path : Optional[str], optional
        The path(s) to the output vector. If None a memory output is produced. Default: None

    overwrite : bool, optional
        Overwrite the output vector if it already exists. Default: True

    add_index : bool, optional
        Add an geospatial index to the output vector. Default: True

    process_layer : int, optional
        The layer index to process. Default: -1

    verbose : bool, optional
        Print progress. Default: False

    prefix : str, optional
        The prefix to add to the layer name. Default: ""

    suffix : str, optional
        The suffix to add to the layer name. Default: ""

    add_uuid : bool, optional
        Add a UUID field to the output vector. Default: False

    allow_lists : bool, optional
        Allow lists of vectors as input. Default: True

    Returns
    -------
    Union[str, List[str]]
        The path(s) to the output vector(s).
    """
    utils_base._type_check(vector, [str, ogr.DataSource, [str, ogr.DataSource]], "vector")
    utils_base._type_check(out_path, [str, [str], None], "out_path")
    utils_base._type_check(overwrite, [bool], "overwrite")
    utils_base._type_check(add_index, [bool], "add_index")
    utils_base._type_check(process_layer, [int], "process_layer")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(allow_lists, [bool], "allow_lists")

    if not allow_lists and isinstance(vector, list):
        raise ValueError("Vector cannot be a list when allow_lists is False.")

    vector_list = utils_base._get_variable_as_list(vector)

    assert utils_gdal._check_is_vector_list(vector_list), f"Vector is not a list of vectors. {vector_list}"

    path_list = utils_io._get_output_paths(
        vector_list,
        out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        overwrite=overwrite,
    )

    output = []
    for index, in_vector in enumerate(vector_list):
        output.append(
            _multipart_to_singlepart(
                in_vector,
                out_path=path_list[index],
                overwrite=overwrite,
                add_index=add_index,
                process_layer=process_layer,
                verbose=verbose,
            )
        )

    if isinstance(vector, list):
        return output[0]

    return output
