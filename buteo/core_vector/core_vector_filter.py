# Standard library
import os
from typing import Union, Optional, List, Dict, Any, Callable, Tuple

# External
from osgeo import ogr, gdal, osr

# Internal
from buteo.utils import (
    utils_io,
    utils_base,
    utils_gdal,
    utils_bbox,
    utils_path,
    utils_projection,
)



def _vector_filter(
    vector: Union[ogr.DataSource, str],
    filter_function: Callable,
    out_path: Optional[str] = None,
    process_layer: int = -1,
    prefix: str = "",
    suffix: str = "",
    overwrite: bool = True,
) -> str:
    """Internal."""
    assert isinstance(vector, (str, ogr.DataSource)), "vector must be a string or an ogr.DataSource object."
    assert isinstance(filter_function, (type(lambda: True))), "filter_function must be a function."

    metadata = _get_basic_metadata_vector(vector)

    if out_path is None:
        out_path = utils_path._get_temp_filepath(vector, prefix=prefix, suffix=suffix)

    assert utils_path._check_is_valid_output_filepath(out_path, overwrite=overwrite), f"out_path is not a valid output path. {out_path}"

    projection = metadata["projection_osr"]

    driver_name = utils_gdal._get_vector_driver_name_from_path(out_path)
    driver = ogr.GetDriverByName(driver_name)

    datasource_destination = driver.CreateDataSource(out_path)
    datasource_original = _vector_open(vector)

    for i in range(metadata["layer_count"]):
        if process_layer != -1 and i != process_layer:
            continue

        features = metadata["layers"][i]["feature_count"]
        field_names = metadata["layers"][i]["field_names"]

        geom_type = metadata["layers"][i]["geom_type_ogr"]

        layer_original = datasource_original.GetLayer(i)
        layer_destination = datasource_destination.CreateLayer(layer_original.GetName(), projection, geom_type)

        for feature in range(features):
            feature = layer_original.GetNextFeature()

            field_values = []
            for field in field_names:
                field_values.append(feature.GetField(field))

            field_dict = {}
            for j, value in enumerate(field_values):
                field_dict[field_names[j]] = value

            if filter_function(field_dict):
                layer_destination.CreateFeature(feature.Clone())

        layer_destination.SyncToDisk()
        layer_destination.ResetReading()
        layer_destination = None

    return out_path


def vector_filter(
    vector: Union[ogr.DataSource, str, List[Union[ogr.DataSource, str]]],
    filter_function: Callable,
    out_path: Optional[Union[str, List[str]]] = None,
    process_layer: int = -1,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    overwrite: bool = True,
) -> Union[str, List[str]]:
    """Filters a vector using its attribute table and a function.

    Parameters
    ----------
    vector : Union[ogr.DataSource, str, List[Union[ogr.DataSource, str]]]
        A vector layer(s) or path(s) to a vector file.

    filter_function : Callable
        A function that takes a dictionary of attributes and returns a boolean.

    out_path : Optional[str], optional
        Path to the output vector file. If None, a memory vector will be created. default: None

    process_layer : int, optional
        The index of the layer to process. If -1, all layers will be processed. default: -1

    prefix : str, optional
        A prefix to add to the output vector file. default: ""

    suffix : str, optional
        A suffix to add to the output vector file. default: ""

    add_uuid : bool, optional
        If True, a uuid will be added to the output path. default: False

    overwrite : bool, optional
        If True, the output file will be overwritten if it already exists. default: True

    Returns
    -------
    Union[str, List[str]]
        The path(s) to the output vector file(s).
    """
    utils_base._type_check(vector, [str, ogr.DataSource, [str, ogr.DataSource]], "vector")
    utils_base._type_check(filter_function, [type(lambda: True)], "filter_function")
    utils_base._type_check(process_layer, [int], "process_layer")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(overwrite, [bool], "overwrite")

    input_is_list = isinstance(vector, list)
    in_paths = utils_io._get_input_paths(vector, "vector")
    out_paths = utils_io._get_output_paths(
        in_paths,
        out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        overwrite=overwrite,
        change_ext="gpkg",
    )

    utils_io._check_overwrite_policy(out_paths, overwrite)
    utils_io._delete_if_required_list(out_paths, overwrite)

    output = []
    for idx, in_vector in enumerate(in_paths):
        output.append(_vector_filter(
            in_vector,
            filter_function,
            out_path=out_paths[idx],
            process_layer=process_layer,
            prefix=prefix,
            suffix=suffix,
            overwrite=overwrite,
        ))

    if input_is_list:
        return output

    return output[0]


def vector_filter_layer(
    vector: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
    layer_name_or_idx: Union[str, int],
    out_path: Optional[Union[str, List[str]]] = None,
    prefix: str = "",
    suffix: str = "_layer",
    add_uuid: bool = False,
    overwrite: bool = True,
):
    """Filters a multi-layer vector source to a single layer.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource, List[str, ogr.DataSource]]
        Vector layer(s) or path(s) to vector layer(s).

    layer_name_or_idx : Union[str, int]
        The name or index of the layer to filter.

    out_path : Optional[str], optional
        The path to the output vector. If None, will create a new file in the same directory as the input vector. Default: None.

    prefix : str, optional
        Prefix to add to the output vector. Default: "".

    suffix : str, optional
        Suffix to add to the output vector. Default: "_layer".

    add_uuid : bool, optional
        If True, will add a UUID to the output vector. Default: False.

    overwrite : bool, optional
        If True, will overwrite the output vector if it already exists. Default: True.

    Returns
    -------
    out_path : str
        Path to the output vector.
    """
    input_is_list = isinstance(vector, list)

    in_paths = utils_io._get_input_paths(vector, "vector")
    out_paths = utils_io._get_output_paths(
        in_paths,
        out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        overwrite=overwrite,
        change_ext="gpkg",
    )

    utils_io._check_overwrite_policy(out_paths, overwrite)
    utils_io._delete_if_required_list(out_paths, overwrite)

    output = []
    for idx, in_vector in enumerate(in_paths):
        ref = _vector_open(in_vector)
        out_path = out_paths[idx]

        if isinstance(layer_name_or_idx, int):
            layer = ref.GetLayerByIndex(layer_name_or_idx)
        elif isinstance(layer_name_or_idx, str):
            layer = ref.GetLayer(layer_name_or_idx)
        else:
            raise RuntimeError("Wrong datatype for layer selection")

        driver_name = utils_gdal._get_vector_driver_name_from_path(out_path)
        driver = ogr.GetDriverByName(driver_name)

        destination = driver.CreateDataSource(out_path)
        destination.CopyLayer(layer, layer.GetName(), ["OVERWRITE=YES"])
        destination.FlushCache()

        destination = None
        ref = None

        output.append(out_path)

    if input_is_list:
        return output

    return output[0]
