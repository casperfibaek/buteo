"""
### Clip vectors to other geometries ###

Clip vector files with other geometries. Can come from rasters or vectors.
"""

# Standard library
import sys; sys.path.append("../../")
from typing import Union, Optional, List

# External
from osgeo import ogr

# Internal
from buteo.utils import utils_base, utils_path, utils_io
from buteo.vector import core_vector


# TODO: Do this with a vectorized approach - multicore.
def _vector_buffer(
    vector: Union[str, ogr.DataSource],
    distance: Union[int, float, str],
    out_path: Optional[str] = None,
    in_place: bool = False,
) -> str:
    """ Internal. """
    assert isinstance(vector, (str, ogr.DataSource)), "Invalid vector input."
    assert isinstance(distance, (int, float, str)), "Invalid distance input."
    assert isinstance(out_path, (str, type(None))), "Invalid output path input."
    if isinstance(distance, (int, float)):
        assert distance >= 0.0, "Distance must be positive."

    if out_path is None:
        out_path = utils_path._get_temp_filepath(vector, suffix="_buffered", ext="gpkg")
    else:
        assert utils_path._check_is_valid_output_filepath(vector, out_path), "Invalid vector output path."

    if not in_place:
        read = core_vector.vector_open(core_vector.vector_copy(vector, out_path))
    else:
        read = core_vector.vector_open(vector)

    metadata = core_vector._get_basic_metadata_vector(read)

    for layer_meta in metadata["layers"]:
        layer_name = layer_meta["layer_name"]
        vector_layer = read.GetLayer(layer_name)

        field_names = layer_meta["field_names"]

        if isinstance(distance, str):
            if distance not in field_names:
                raise ValueError(f"Attribute to buffer by not in vector field names: {distance} not in {field_names}")

        feature_count = vector_layer.GetFeatureCount()

        vector_layer.ResetReading()
        for _ in range(feature_count):
            feature = vector_layer.GetNextFeature()

            if isinstance(distance, str):
                buffer_distance = float(feature.GetField(distance))
            else:
                buffer_distance = float(distance)

            if feature is None:
                break

            feature_geom = feature.GetGeometryRef()

            if feature_geom is None:
                continue

            try:
                feature.SetGeometry(feature_geom.Buffer(buffer_distance))
                vector_layer.SetFeature(feature)
            except ValueError:
                continue

        vector_layer.GetExtent()
        vector_layer.ResetReading()
        vector_layer.SyncToDisk()

    read.FlushCache()
    read = None

    return out_path


def vector_buffer(
    vector: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
    distance: Union[int, float],
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
