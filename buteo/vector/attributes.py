import sys; sys.path.append('../../')
from osgeo import ogr
from typing import Union
from buteo.utils import type_check
from buteo.vector.io import vector_to_metadata
from buteo.gdal_utils import vector_to_reference
import pandas as pd
import numpy as np


def vector_get_attribute_table(
    vector: Union[ogr.DataSource, str, list],
    process_layer: int=0,
    include_geom: bool=False,
) -> pd.DataFrame:
    type_check(vector, [ogr.DataSource, str, list], "vector")
    type_check(process_layer, [int], "process_layer")
    type_check(include_geom, [bool], "include_geom")

    ref = vector_to_reference(vector)
    metadata = vector_to_metadata(ref, process_layer=process_layer, latlng_and_footprint=False)

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
    vector: Union[ogr.DataSource, str, list],
) -> Union[np.ndarray, list]:
    type_check(vector, (ogr.DataSource, str, list), "vector")
    
    metadata = vector_to_metadata(vector, latlng_and_footprint=False)
    features = metadata["layers"][0]["feature_count"]
    
    ref = vector_to_reference(vector)
    layer = ref.GetLayer()

    fid_list = np.empty(features, dtype=int)

    for index in range(features):
        feature = layer.GetNextFeature()
        fid_list[index] = feature.GetFID()

    layer.ResetReading()

    return fid_list
