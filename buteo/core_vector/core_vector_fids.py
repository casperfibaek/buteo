
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



def vector_reset_fids(
    vector: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
) -> Union[str, List[str]]:
    """Resets the FID column of a vector to 0, 1, 2, 3, ...

    Parameters
    ----------
    vector : Union[str, ogr.DataSource, List[str, ogr.DataSource]]
        Vector layer(s) or path(s) to vector layer(s).

    out_path : Optional[str], optional
        The path to the output vector. If None, will create a new file in the same directory as the input vector. Default: None.

    Returns
    -------
    out_path : str
        Path to the output vector.
    """
    assert isinstance(vector, (str, ogr.DataSource)), "vector must be a string or an ogr.DataSource object."

    input_is_list = isinstance(vector, list)

    in_paths = utils_io._get_input_paths(vector, "vector")

    output = []
    for idx, in_vector in enumerate(in_paths):
        ref = _vector_open(in_vector)

        layers = ref.GetLayerCount()

        for layer_index in range(layers):
            layer = ref.GetLayer(layer_index)
            layer.ResetReading()
            fids = []

            for feature in layer:
                fids.append(feature.GetFID())

            layer.ResetReading()
            fids = sorted(fids)

            layer_defn = layer.GetLayerDefn()
            field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]

            for feature in layer:
                current_fid = feature.GetFID()
                target_fid = fids.index(current_fid)

                # If there is a fid field, update it too.
                if "fid" in field_names:
                    feature.SetField("fid", str(target_fid))

                feature.SetFID(target_fid)
                layer.SetFeature(feature)

            layer.SyncToDisk()

        ref = None

        output.append(in_paths[idx])

    if input_is_list:
        return output

    return output[0]


def vector_create_fid_field(
    vector: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
):
    """Creates a FID field in a vector if it doesn't exist.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource, List[str, ogr.DataSource]]
        Vector layer(s) or path(s) to vector layer(s).

    Returns
    -------
        str: original vector path
    """
    assert isinstance(vector, (str, ogr.DataSource)), "vector must be a string or an ogr.DataSource object."

    input_is_list = isinstance(vector, list)

    in_paths = utils_io._get_input_paths(vector, "vector")

    output = []
    for idx, in_vector in enumerate(in_paths):
        ref = _vector_open(in_vector)

        layers = ref.GetLayerCount()

        for layer_index in range(layers):
            layer = ref.GetLayer(layer_index)
            layer.ResetReading()
            layer_defn = layer.GetLayerDefn()
            field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]

            if "fid" not in field_names:
                field = ogr.FieldDefn("fid", ogr.OFTInteger)
                layer.CreateField(field)

            layer.ResetReading()

            for idx, feature in enumerate(layer):
                feature.SetField("fid", idx)
                layer.SetFeature(feature)

            layer.SyncToDisk()

        ref = None

        output.append(in_paths[idx])

    if input_is_list:
        return output

    return output[0]


def vector_create_attribute_from_fid(
    vector: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
    attribute_name: str = "id",
):
    """Creates an attribute from the FID field in a vector.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource, List[str, ogr.DataSource]]
        Vector layer(s) or path(s) to vector layer(s).

    attribute_name : str, optional
        The name of the attribute to create. Default: "id"

    Returns
    -------
        str: original vector path
    """
    assert isinstance(vector, (str, ogr.DataSource)), "vector must be a string or an ogr.DataSource object."
    assert isinstance(attribute_name, str), "attribute_name must be a string."

    input_is_list = isinstance(vector, list)

    in_paths = utils_io._get_input_paths(vector, "vector")

    output = []
    for idx, in_vector in enumerate(in_paths):
        ref = _vector_open(in_vector)

        layers = ref.GetLayerCount()

        for layer_index in range(layers):
            layer = ref.GetLayer(layer_index)
            layer.ResetReading()
            layer_defn = layer.GetLayerDefn()
            field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]

            if "fid" in field_names and attribute_name not in field_names:
                field = ogr.FieldDefn(attribute_name, ogr.OFTInteger)
                layer.CreateField(field)

            layer.ResetReading()

            for feature in layer:
                feature.SetField(attribute_name, feature.GetFID())
                layer.SetFeature(feature)

            layer.SyncToDisk()

        ref = None

        output.append(in_paths[idx])

    if input_is_list:
        return output

    return output[0]
