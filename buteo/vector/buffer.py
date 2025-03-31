"""### Buffer vector geometries. ###

Buffer vector geometries by a fixed distance or attribute field value.
"""

# Standard library
import os
from typing import Union, Optional, List

# External
from osgeo import ogr, gdal

# Internal
from buteo.utils import utils_base, utils_path, utils_io, utils_gdal
from buteo.core_vector.core_vector_read import open_vector as vector_open
from buteo.core_vector.core_vector_info import _get_basic_info_vector as _get_basic_metadata_vector


def _vector_buffer(
    vector: Union[str, ogr.DataSource],
    distance: Union[int, float, str],
    out_path: Optional[str] = None,
    in_place: bool = False,
    force_multipolygon: bool = True,
) -> str:
    """Internal buffer implementation."""
    assert isinstance(vector, (str, ogr.DataSource)), "Invalid vector input."
    assert isinstance(distance, (int, float, str)), "Invalid distance input."
    assert isinstance(out_path, (str, type(None))), "Invalid output path input."

    # Store original vector path for in-place operations
    original_path = None
    if isinstance(vector, str) and in_place:
        original_path = vector

    if out_path is None:
        out_path = utils_path._get_temp_filepath(vector, suffix="_buffered", ext="gpkg")
    else:
        assert utils_path._check_is_valid_output_filepath(out_path), "Invalid vector output path."

    read = vector_open(vector, writeable=in_place)
    # If open_vector returns a list (which it shouldn't in this case), take the first item
    if isinstance(read, list):
        read = read[0]
    metadata = _get_basic_metadata_vector(read)

    if in_place:
        if metadata["geom_type_name"] not in ["polygon"]:
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
        field_names = []
        layer = read.GetLayer(0)
        layer_defn = layer.GetLayerDefn()
        for i in range(layer_defn.GetFieldCount()):
            field_defn = layer_defn.GetFieldDefn(i)
            field_names.append(field_defn.GetName())
            
        if distance not in field_names:
            raise AttributeError(f"Attribute ({distance}) is a string and not one of the vector field names.")

    # Process layers in the vector
    layer_count = read.GetLayerCount()
    for i in range(layer_count):
        layer = read.GetLayer(i)
        layer_name = layer.GetName()

        vector_layer_origin = layer
        if in_place:
            vector_layer_destination = vector_layer_origin
        else:
            vector_layer_destination = output_ds.CreateLayer(layer_name, geom_type=ogr.wkbMultiPolygon, srs=vector_layer_origin.GetSpatialRef())
            vector_layer_origin_defn = vector_layer_origin.GetLayerDefn()

            for i in range(vector_layer_origin_defn.GetFieldCount()):
                vector_layer_destination.CreateField(vector_layer_origin_defn.GetFieldDefn(i))

        if isinstance(distance, str):
            buffer_is_field = True
        else:
            buffer_is_field = False

        vector_layer_origin = read.GetLayer(layer_name)

        feature_count = vector_layer_origin.GetFeatureCount()

        vector_layer_origin.ResetReading()
        for _ in range(feature_count):
            feature = vector_layer_origin.GetNextFeature()

            if feature is None:
                break

            if buffer_is_field:
                buffer_distance = feature.GetField(distance)
            else:
                buffer_distance = distance

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

    if read is not None:
        read.FlushCache()
        read = None

    # For in-place operations, return the original path instead of the potentially virtual path
    if in_place and original_path is not None:
        # Normalize slashes to match the test expectations
        return utils_path._get_unix_path(original_path) if os.name == 'nt' else original_path
        
    # Normalize slashes to match the test expectations
    return utils_path._get_unix_path(out_path) if os.name == 'nt' else out_path


def vector_buffer(
    vector: Union[str, ogr.DataSource, gdal.Dataset, List[Union[str, ogr.DataSource, gdal.Dataset]]],
    distance: Union[int, float, str],
    out_path: Optional[Union[str, List[str]]] = None,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    add_timestamp: bool = False,
    in_place: bool = False,
    force_multipolygon: bool = True,
    overwrite: bool = True,
) -> Union[str, List[str]]:
    """Buffers a vector with a fixed distance or an attribute.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource, gdal.Dataset, List[Union[str, ogr.DataSource, gdal.Dataset]]]
        Vector(s) to buffer.

    distance : Union[int, float, str]
        The distance to buffer with. If string, uses the attribute of that name.

    out_path : Optional[Union[str, List[str]]], optional
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
        
    force_multipolygon : bool, optional
        If True, forces output to be multipolygon geometry. Default: True

    overwrite : bool, optional
        Overwrite output if it already exists. Default: True

    Returns
    -------
    Union[str, List[str]]
        Output path(s) of buffered vector(s).
    """
    utils_base._type_check(vector, [str, ogr.DataSource, gdal.Dataset, [str, ogr.DataSource, gdal.Dataset]], "vector")
    utils_base._type_check(distance, [int, float, str], "distance")
    utils_base._type_check(out_path, [str, [str], None], "out_path")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(add_timestamp, [bool], "add_timestamp")
    utils_base._type_check(in_place, [bool], "in_place")
    utils_base._type_check(force_multipolygon, [bool], "force_multipolygon")
    utils_base._type_check(overwrite, [bool], "overwrite")

    input_is_list = isinstance(vector, list)
    in_paths = utils_io._get_input_paths(vector, "vector")

    # Handle output paths
    if out_path is None:
        # Create temp paths for each input
        out_paths = []
        for path in in_paths:
            temp_path = utils_path._get_temp_filepath(
                path,
                prefix=prefix,
                suffix=suffix,
                add_uuid=add_uuid,
                add_timestamp=add_timestamp,
            )
            out_paths.append(temp_path)
    elif isinstance(out_path, list):
        # Use provided output paths directly
        if len(out_path) != len(in_paths):
            raise ValueError("Number of output paths must match number of input paths")
        out_paths = out_path
    else:
        # Single output path for a single input
        if len(in_paths) > 1:
            raise ValueError("Single output path provided for multiple inputs")
        out_paths = [out_path]

    if not in_place:
        utils_io._check_overwrite_policy(out_paths, overwrite)
    utils_io._delete_if_required_list(out_paths, overwrite)

    output = []
    for idx, in_vector in enumerate(in_paths):
        output.append(
            _vector_buffer(
                in_vector,
                distance,
                out_path=out_paths[idx],
                in_place=in_place,
                force_multipolygon=force_multipolygon,
            )
        )

    if input_is_list:
        return output

    return output[0]
