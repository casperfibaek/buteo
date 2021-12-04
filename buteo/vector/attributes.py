import sys
import numpy as np
import pandas as pd
from osgeo import ogr
from typing import Union

sys.path.append("../../")

from buteo.utils import type_check
from buteo.vector.io import internal_vector_to_metadata, open_vector


def vector_get_attribute_table(
    vector: Union[str, ogr.DataSource],
    process_layer: int = 0,
    include_geom: bool = False,
) -> pd.DataFrame:
    type_check(vector, [str, ogr.DataSource], "vector")
    type_check(process_layer, [int], "process_layer")
    type_check(include_geom, [bool], "include_geom")

    ref = open_vector(vector)
    metadata = internal_vector_to_metadata(
        ref, process_layer=process_layer, create_geometry=False
    )

    attribute_table_header = None
    feature_count = None

    attribute_table_header = metadata["layers"][process_layer]["field_names"]
    feature_count = metadata["layers"][process_layer]["feature_count"]

    attribute_table = []

    layer = ref.GetLayer(process_layer)

    for _ in range(feature_count):
        feature = layer.GetNextFeature()
        attributes = [feature.GetFID()]

        for field_name in attribute_table_header:
            attributes.append(feature.GetField(field_name))

        if include_geom:
            geom_defn = feature.GetGeometryRef()
            attributes.append(geom_defn.ExportToIsoWkt())

        attribute_table.append(attributes)

    attribute_table_header.insert(0, "fid")

    if include_geom:
        attribute_table_header.append("geom")

    df = pd.DataFrame(attribute_table, columns=attribute_table_header)

    return df


def vector_get_fids(
    vector: Union[str, ogr.DataSource], process_layer: int = 0
) -> np.ndarray:
    type_check(vector, [str, ogr.DataSource], "vector")
    type_check(process_layer, [int], "process_layer")

    metadata = internal_vector_to_metadata(vector)
    features = metadata["layers"][0]["feature_count"]

    ref = open_vector(vector)
    layer = ref.GetLayer(process_layer)

    if layer is None:
        raise Exception(f"Requested a non-existing layer: layer_idx={process_layer}")

    fid_list = np.empty(features, dtype=int)

    for index in range(features):
        feature = layer.GetNextFeature()
        fid_list[index] = feature.GetFID()

    layer.ResetReading()

    return fid_list
