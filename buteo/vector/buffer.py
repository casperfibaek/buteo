"""
### Clip vectors to other geometries ###

Clip vector files with other geometries. Can come from rasters or vectors.
"""

# Standard library
import sys; sys.path.append("../../")
import os
from typing import Union, Optional, List

# External
from osgeo import ogr

# Internal
from buteo.utils import utils_base, utils_path, utils_io, utils_gdal
from buteo.vector import core_vector


def _vector_buffer(
    vector: Union[str, ogr.DataSource],
    distance: Union[int, float, str],
    out_path: Optional[str] = None,
    in_place: bool = False,
    force_multipolygon: bool = True,
) -> str:
    """ Internal. """
    assert isinstance(vector, (str, ogr.DataSource)), "Invalid vector input."
    assert isinstance(distance, (int, float, str)), "Invalid distance input."
    assert isinstance(out_path, (str, type(None))), "Invalid output path input."

    if out_path is None:
        out_path = utils_path._get_temp_filepath(vector, suffix="_buffered", ext="gpkg")
    else:
        assert utils_path._check_is_valid_output_filepath(vector, out_path), "Invalid vector output path."

    read = core_vector.vector_open(vector)
    metadata = core_vector._get_basic_metadata_vector(read)

    if in_place:
        for l in metadata["layers"]:
            if l["geom_type"] not in ["Polygon", "MultiPolygon", "3D Polygon", "3D MultiPolygon"]:
                raise ValueError("Input vector must be polygonal to be buffered in-place.")
    else:
        driver = ogr.GetDriverByName(utils_gdal._get_driver_name_from_path(out_path))
        if driver is None:
            raise Exception("Driver for output buffer file is not available.")
        
        if os.path.exists(out_path):
            driver.DeleteDataSource(out_path)

        output_ds = driver.CreateDataSource(out_path)
        if output_ds is None:
            raise Exception(f"Could not create output file: {out_path}")

    if isinstance(distance, str):
        for layer_meta in metadata["layers"]:
            if distance not in layer_meta["field_names"]:
                raise AttributeError(f"Attribute ({distance}) is a string and not one of the vector field names.")

    for layer_meta in metadata["layers"]:
        layer_name = layer_meta["layer_name"]

        vector_layer_origin = read.GetLayer(layer_name)
        if in_place:
            vector_layer_destination = vector_layer_origin
        else:
            vector_layer_destination = output_ds.CreateLayer(layer_name, geom_type=ogr.wkbMultiPolygon, srs=vector_layer_origin.GetSpatialRef())
            vector_layer_origin_defn = vector_layer_origin.GetLayerDefn()

            for i in range(vector_layer_origin_defn.GetFieldCount()):
                vector_layer_destination.CreateField(vector_layer_origin_defn.GetFieldDefn(i))

        if isinstance(distance, str):
            if distance not in layer_meta["field_names"]:
                raise AttributeError(f"Attribute ({distance}) is a stirng and not one of the vector field names.")
            buffer_is_field = True
        else:
            buffer_is_field = False

        vector_layer_origin = read.GetLayer(layer_name)

        feature_count = vector_layer_origin.GetFeatureCount()

        vector_layer_origin.ResetReading()
        for _ in range(feature_count):
            feature = vector_layer_origin.GetNextFeature()

            if buffer_is_field:
                buffer_distance = feature.GetField(distance)
            else:
                buffer_distance = distance

            if feature is None:
                break

            feature_geom = feature.GetGeometryRef()

            if feature_geom is None:
                continue

            buffered_geometry = feature_geom.Buffer(buffer_distance)

            if force_multipolygon:
                if buffered_geometry.GetGeometryType() == ogr.wkbPolygon:
                    buffered_geometry = ogr.ForceToMultiPolygon(buffered_geometry)

            if in_place:
                try:
                    feature.SetGeometry(buffered_geometry)
                    vector_layer_origin.SetFeature(feature)
                except ValueError:
                    continue

            else:
                out_feature = ogr.Feature(vector_layer_destination.GetLayerDefn())
                out_feature.SetGeometry(buffered_geometry)

                for i in range(feature.GetFieldCount()):
                    out_feature.SetField(i, feature.GetField(i))
                
                vector_layer_destination.CreateFeature(out_feature)

        vector_layer_destination.GetExtent()
        vector_layer_destination.ResetReading()
        vector_layer_destination.SyncToDisk()

    read.FlushCache()
    read = None

    return out_path


def vector_buffer(
    vector: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
    distance: Union[int, float, str],
    out_path: Optional[Union[str, List[str]]] = None,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    add_timestamp: bool = False,
    in_place: bool = False,
    overwrite: bool = True,
) -> Union[str, List[str]]:
    """
    Buffers a vector with a fixed distance or an attribute.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource, List[str, ogr.DataSource]]
        Vector(s) to buffer.

    distance : Union[int, float, str]
        The distance to buffer with. If string, uses the attribute of that name.

    out_path : Optional[str], optional
        Output path. If None, memory vectors are created. Default: None

    prefix : str, optional
        Prefix to add to the output path. Default: ""

    suffix : str, optional
        Suffix to add to the output path. Default: ""

    add_uuid : bool, optional
        Add UUID to the output path. Default: False

    add_timestamp : bool, optional
        Add timestamp to the output path. Default: False
    
    in_place : bool, optional
        If True, overwrites the input vector. Default: False

    overwrite : bool, optional
        Overwrite output if it already exists. Default: True

    Returns
    -------
    Union[str, List[str]]
        Output path(s) of clipped vector(s).
    """
    utils_base._type_check(vector, [str, ogr.DataSource, [str, ogr.DataSource]], "vector")
    utils_base._type_check(distance, [int, float, str], "distance")
    utils_base._type_check(out_path, [str, None], "out_path")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(add_timestamp, [bool], "add_timestamp")
    utils_base._type_check(in_place, [bool], "in_place")
    utils_base._type_check(overwrite, [bool], "overwrite")

    input_is_list = isinstance(vector, list)
    input_data = utils_io._get_input_paths(vector, "vector")
    output_data = utils_io._get_output_paths(
        input_data,
        out_path,
        in_place=in_place,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        add_timestamp=add_timestamp,
        overwrite=overwrite,
    )

    if not in_place:
        utils_path._delete_if_required_list(output_data, overwrite)

    output = []
    for idx, in_vector in enumerate(input_data):
        output.append(
            _vector_buffer(
                in_vector,
                distance,
                out_path=output_data[idx],
                in_place=in_place,
            )
        )

    if input_is_list:
        return output

    return output[0]
