""" ### Metadata functions for vector layers. ### """

# Standard library
from typing import Union, List, Dict, Any
import os

# External
from osgeo import ogr, gdal, osr

# Internal
from buteo.utils import (
    utils_base,
    utils_gdal,
    utils_bbox,
)
from buteo.vector.core_vector import _vector_open



def _vector_to_metadata(
    vector: Union[str, ogr.DataSource, gdal.Dataset],
) -> Dict[str, Any]:
    """Internal."""
    assert isinstance(vector, (str, ogr.DataSource)), "vector must be a path or a DataSource"

    datasource = _vector_open(vector)

    vector_driver = datasource.GetDriver()

    path = datasource.GetDescription()
    basename = os.path.basename(path)
    split_path = os.path.split(basename)
    name = split_path[0]
    ext = split_path[1]

    driver = vector_driver.GetName()

    in_memory = utils_gdal._check_is_dataset_in_memory(datasource)

    layer_count = datasource.GetLayerCount()
    layers = []

    vector_bbox = None
    for layer_index in range(layer_count):
        layer = datasource.GetLayerByIndex(layer_index)

        x_min, x_max, y_min, y_max = layer.GetExtent()
        layer_bbox = [x_min, x_max, y_min, y_max]
        vector_bbox = layer_bbox

        layer_name = layer.GetName()

        column_fid = layer.GetFIDColumn()
        column_geom = layer.GetGeometryColumn()

        if column_geom == "":
            column_geom = "geom"

        feature_count = layer.GetFeatureCount()

        projection_osr = layer.GetSpatialRef()
        projection_wkt = layer.GetSpatialRef().ExportToWkt()

        if layer_index > 0:
            v_x_min, v_x_max, v_y_min, v_y_max = vector_bbox
            if x_min < v_x_min:
                vector_bbox[0] = x_min
            if x_max > v_x_max:
                vector_bbox[1] = x_max
            if y_min < v_y_min:
                vector_bbox[2] = y_min
            if y_max > v_y_max:
                vector_bbox[3] = y_max

        layer_defn = layer.GetLayerDefn()

        geom_type_ogr = layer_defn.GetGeomType()
        geom_type = ogr.GeometryTypeToName(layer_defn.GetGeomType())

        field_count = layer_defn.GetFieldCount()
        field_names = []
        field_types = []
        field_types_ogr = []

        for field_index in range(field_count):
            field_defn = layer_defn.GetFieldDefn(field_index)
            field_names.append(field_defn.GetName())
            field_type = field_defn.GetType()
            field_types_ogr.append(field_type)
            field_types.append(field_defn.GetFieldTypeName(field_type))

        layer_dict = {
            "layer_name": layer_name,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "column_fid": column_fid,
            "column_geom": column_geom,
            "feature_count": feature_count,
            "projection_wkt": projection_wkt,
            "projection_osr": projection_osr,
            "geom_type": geom_type,
            "geom_type_ogr": geom_type_ogr,
            "field_count": field_count,
            "field_names": field_names,
            "field_types": field_types,
            "field_types_ogr": field_types_ogr,
            "is_raster": False,
            "is_vector": True,
            "bbox": layer_bbox,
            "extent": layer_bbox,
        }

        layer_bboxes = utils_bbox._additional_bboxes(layer_bbox, projection_osr)

        for key, value in layer_bboxes.items():
            layer_dict[key] = value


        ## MOVE TO A SINGLE FUNCTION
        def get_bbox_as_vector_layer():
            return utils_bbox._get_vector_from_bbox(layer_bbox, projection_osr) # pylint: disable=cell-var-from-loop


        def get_bbox_as_vector_latlng_layer():
            projection_osr_latlng = osr.SpatialReference()
            projection_osr_latlng.ImportFromWkt('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]')
            # projection_osr_latlng.ImportFromEPSG(4326)

            return utils_bbox._get_vector_from_bbox(layer_dict["bbox_latlng"], projection_osr_latlng)  # pylint: disable=cell-var-from-loop


        layer_dict["get_bbox_vector"] = get_bbox_as_vector_layer
        layer_dict["get_bbox_vector_latlng"] = get_bbox_as_vector_latlng_layer

        layers.append(layer_dict)

    metadata = {
        "path": path,
        "basename": basename,
        "name": name,
        "ext": ext,
        "in_memory": in_memory,
        "projection_wkt": projection_wkt,
        "projection_osr": projection_osr,
        "driver": driver,
        "x_min": vector_bbox[0],
        "x_max": vector_bbox[1],
        "y_min": vector_bbox[2],
        "y_max": vector_bbox[3],
        "is_vector": True,
        "is_raster": False,
        "layer_count": layer_count,
        "layers": layers,
        "extent": vector_bbox,
        "bbox": vector_bbox,
    }

    vector_bboxes = utils_bbox._additional_bboxes(vector_bbox, projection_osr)

    for key, value in vector_bboxes.items():
        metadata[key] = value


    def get_bbox_as_vector():
        return utils_bbox._get_vector_from_bbox(vector_bbox, projection_osr) # pylint: disable=cell-var-from-loop


    def get_bbox_as_vector_latlng():
        projection_osr_latlng = osr.SpatialReference()
        projection_osr_latlng.ImportFromWkt('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]')
        # projection_osr_latlng.ImportFromEPSG(4326)

        return utils_bbox._get_vector_from_bbox(metadata["bbox_latlng"], projection_osr_latlng)  # pylint: disable=cell-var-from-loop


    metadata["get_bbox_vector"] = get_bbox_as_vector
    metadata["get_bbox_vector_latlng"] = get_bbox_as_vector_latlng

    return metadata


def vector_to_metadata(
    vector: Union[ogr.DataSource, str, List[Union[ogr.DataSource, str]]],
    allow_lists: bool = True,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Creates a dictionary with metadata about the vector layer.

    Parameters
    ----------
    vector : Union[ogr.DataSource, str, List[Union[ogr.DataSource, str]]]
        A vector layer(s) or path(s) to a vector file.

    allow_lists : bool, optional
        If True, vector can be a list of vector layers or paths. If False, vector must be a single vector layer or path. default: True

    Returns
    -------
    Union[Dict[str, Any], List[Dict[str, Any]]]
    """
    utils_base._type_check(vector, [str, ogr.DataSource, [str, ogr.DataSource]], "vector")
    utils_base._type_check(allow_lists, [bool], "allow_lists")

    if isinstance(vector, list) and not allow_lists:
        raise ValueError("The vector parameter cannot be a list when allow_lists is False.")

    vector_list = utils_base._get_variable_as_list(vector)

    if not utils_gdal._check_is_vector_list(vector_list):
        raise ValueError("The vector parameter must be a list of vector layers.")

    output = []

    for in_vector in vector_list:
        output.append(_vector_to_metadata(in_vector))

    if isinstance(vector, list):
        return output

    return output[0]
