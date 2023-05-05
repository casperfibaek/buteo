"""
### Basic IO functions for working with Vectprs ###

The basic module for interacting with vector data
"""

# TODO: More attribute functions
# TODO: Repair vector functions
# TODO: Sanity checks
# TODO: Joins (by attribute, location, summary, etc..)
# TODO: Union, Erase, ..
# TODO: Sampling functions
# TODO: Vector intersects, etc..

# Standard library
import sys; sys.path.append("../../")
from typing import Union, Optional, List, Dict, Any, Callable
import os

# External
import numpy as np
from osgeo import ogr, gdal, osr

# Internal
from buteo.utils import (
    utils_io,
    utils_base,
    utils_gdal,
    utils_bbox,
    utils_path,
)


def _open_vector(
    vector: Union[str, ogr.DataSource, gdal.Dataset],
    writeable: bool = True,
    allow_raster: bool = True,
) -> ogr.DataSource:
    """ Internal. """
    assert isinstance(vector, (str, ogr.DataSource)), "vector must be a path or a DataSource"

    opened = None
    if utils_gdal._check_is_vector(vector):
        if isinstance(vector, str):
            gdal.PushErrorHandler("CPLQuietErrorHandler")
            opened = ogr.Open(vector, gdal.GF_Write) if writeable else ogr.Open(vector, gdal.GF_Read)
            gdal.PopErrorHandler()
        elif isinstance(vector, ogr.DataSource):
            opened = vector
        else:
            raise RuntimeError(f"Could not read input vector: {vector}")

    elif allow_raster and utils_gdal._check_is_raster(vector):
        if isinstance(vector, str):
            gdal.PushErrorHandler("CPLQuietErrorHandler")
            opened = gdal.Open(vector, gdal.GF_Write) if writeable else ogr.Open(vector, gdal.GF_Read)
            gdal.PopErrorHandler()
        elif isinstance(vector, gdal.Dataset):
            opened = vector
        else:
            raise RuntimeError(f"Could not read input vector: {vector}")

        bbox = utils_bbox._get_bbox_from_geotransform(
            opened.GetGeoTransform(),
            opened.RasterXSize,
            opened.RasterYSize,
        )

        projection_wkt = opened.GetProjection()
        projection_osr = osr.SpatialReference()
        projection_osr.ImportFromWkt(projection_wkt)

        vector_bbox = utils_bbox._get_vector_from_bbox(bbox, projection_osr)
        opened = ogr.Open(vector_bbox, gdal.GF_Write) if writeable else ogr.Open(vector_bbox, gdal.GF_Read)

    else:
        raise RuntimeError(f"Could not read input vector: {vector}")

    if opened is None:
        raise RuntimeError(f"Could not read input vector: {vector}")

    return opened


def vector_open(
    vector: Union[str, ogr.DataSource, gdal.Dataset, List[Union[str, ogr.DataSource, gdal.Dataset]]],
    writeable: bool = True,
    allow_raster: bool = True,
    allow_lists: bool = True,
) -> Union[ogr.DataSource, List[ogr.DataSource]]:
    """
    Opens a vector to an ogr.Datasource class.

    Parameters
    ----------
    vector : str, ogr.DataSource, gdal.Dataset, list[str, ogr.DataSource, gdal.Dataset]
        The vector to open. If a raster is supplied the bounding box is opened as a vector.

    writeable : bool, optional
        If True, the vector is opened in write mode. Default: True

    allow_raster : bool, optional
        If True, a raster will be opened as a vector bounding box. Default: True
    
    allow_lists : bool, optional
        If True, the input can be a list of vectors. Default: True

    Returns
    -------
    ogr.DataSource, list[ogr.DataSource]
        The opened vector(s).
    """
    utils_base._type_check(vector, [str, ogr.DataSource, gdal.Dataset, [str, ogr.DataSource, gdal.Dataset]], "vector")
    utils_base._type_check(writeable, [bool], "writeable")

    if isinstance(vector, list) and not allow_lists:
        raise ValueError("Cannot open a list of vectors when allow_list is False.")

    vectors = utils_base._get_variable_as_list(vector)

    output = []
    for element in vectors:
        output.append(_open_vector(element, writeable=writeable, allow_raster=allow_raster))

    if isinstance(vector, list):
        return output

    return output[0]


def _vector_to_metadata(
    vector: Union[str, ogr.DataSource, gdal.Dataset],
) -> Dict[str, Any]:
    """ Internal. """
    assert isinstance(vector, (str, ogr.DataSource)), "vector must be a path or a DataSource"

    datasource = _open_vector(vector)

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
    """
    Creates a dictionary with metadata about the vector layer.

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


def _vector_filter(
    vector: Union[ogr.DataSource, str],
    filter_function: Callable,
    out_path: Optional[str] = None,
    process_layer: int = -1,
    prefix: str = "",
    suffix: str = "",
    overwrite: bool = True,
) -> str:
    """ Internal. """
    assert isinstance(vector, (str, ogr.DataSource)), "vector must be a string or an ogr.DataSource object."
    assert isinstance(filter_function, (type(lambda: True))), "filter_function must be a function."

    metadata = _vector_to_metadata(vector)

    if out_path is None:
        out_path = utils_path._get_temp_filepath(vector, prefix=prefix, suffix=suffix)

    assert utils_path._check_is_valid_output_filepath(out_path, overwrite=overwrite), f"out_path is not a valid output path. {out_path}"

    projection = metadata["projection_osr"]

    driver = utils_gdal._get_vector_driver_name_from_path(out_path)

    datasource_destination = driver.CreateDataSource(out_path)
    datasource_original = vector_open(vector)

    for i, _layer in enumerate(metadata["layers"]):
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
    out_path: Optional[str] = None,
    process_layer: int = -1,
    allow_lists: bool = True,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    overwrite: bool = True,
) -> Union[str, List[str]]:
    """
    Filters a vector using its attribute table and a function.

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

    allow_lists : bool, optional
        If True, vector can be a list of vector layers or paths. If False, vector must be a single vector layer or path. default: True

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
    utils_base._type_check(allow_lists, [bool], "allow_lists")

    if isinstance(vector, list) and not allow_lists:
        raise ValueError("The vector parameter cannot be a list when allow_lists is False.")

    vector_list = utils_base._get_variable_as_list(vector)

    if not utils_gdal._check_is_vector_list(vector_list):
        raise ValueError("The vector parameter must be a list of vector layers.")

    path_list = utils_io._get_output_paths(
        vector_list,
        out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        overwrite=overwrite,
    )

    output = []

    for index, in_vector in vector_list:
        output.append(_vector_filter(
            in_vector,
            filter_function,
            out_path=path_list[index],
            process_layer=process_layer,
            prefix=prefix,
            suffix=suffix,
            overwrite=overwrite,
        ))

    if isinstance(vector, list):
        return output

    return output[0]


def vector_add_index(
    vector: Union[ogr.DataSource, str, List[Union[ogr.DataSource, str]]],
    allow_lists: bool = True,
) -> Union[str, List[str]]:
    """
    Adds a spatial index to the vector in place, if it doesn't have one.

    Parameters
    ----------
    vector : Union[ogr.DataSource, str, List[Union[ogr.DataSource, str]]]
        A vector layer(s) or path(s) to a vector file.

    allow_lists : bool, optional
        If True, vector can be a list of vector layers or paths. If False, vector must be a single vector layer or path. default: True
    
    Returns
    -------
    Union[str, List[str]]
        The path(s) to the input vector file(s).
    """
    utils_base._type_check(vector, [str, ogr.DataSource, [str, ogr.DataSource]], "vector")

    if isinstance(vector, list) and not allow_lists:
        raise ValueError("The vector parameter cannot be a list when allow_lists is False.")

    vector_list = utils_base._get_variable_as_list(vector)

    if not utils_gdal._check_is_vector_list(vector_list):
        raise ValueError("The vector parameter must be a list of vector layers.")

    output = utils_gdal._get_path_from_dataset_list(vector_list)

    try:
        for in_vector in vector_list:
            metadata = _vector_to_metadata(in_vector)
            ref = _open_vector(in_vector)

            for layer in metadata["layers"]:
                name = layer["layer_name"]
                geom = layer["column_geom"]

                sql = f"SELECT CreateSpatialIndex('{name}', '{geom}') WHERE NOT EXISTS (SELECT HasSpatialIndex('{name}', '{geom}'));"
                ref.ExecuteSQL(sql, dialect="SQLITE")
    except:
        raise RuntimeError(f"Error while creating indices for {vector}") from None

    if isinstance(vector, list):
        return output

    return output[0]


def _vector_add_shapes_in_place(
    vector: Union[ogr.DataSource, str],
    shapes: Optional[List[str]] = None,
    prefix: str = "",
    verbose: bool = False,
) -> str:
    """ Internal. """
    assert isinstance(vector, (ogr.DataSource, str)), "vector must be a vector layer or path to one."
    assert isinstance(shapes, (list, tuple)) or shapes is None, "shapes must be a list of shapes."
    assert isinstance(prefix, str), "prefix must be a string."

    all_shapes = ["area", "perimeter", "ipq", "hull", "compactness", "centroid"]

    if shapes is None:
        shapes = all_shapes
    else:
        for shape in shapes:
            if shape not in all_shapes:
                raise ValueError(f"{shape} is not a valid shape.")

    datasource = _open_vector(vector)
    out_path = utils_gdal._get_path_from_dataset(datasource, dataset_type="vector")
    metadata = _vector_to_metadata(datasource)

    for index in range(metadata["layer_count"]):
        vector_current_fields = metadata["layers"][index]["field_names"]
        vector_layer = datasource.GetLayer(index)

        vector_layer.StartTransaction()

        # Add missing fields
        for attribute in shapes:
            if attribute == "centroid":
                if "centroid_x" not in vector_current_fields:
                    field_defn = ogr.FieldDefn(f"{prefix}centroid_x", ogr.OFTReal)
                    vector_layer.CreateField(field_defn)

                if "centroid_y" not in vector_current_fields:
                    field_defn = ogr.FieldDefn(f"{prefix}centroid_y", ogr.OFTReal)
                    vector_layer.CreateField(field_defn)

            elif attribute not in vector_current_fields:
                field_defn = ogr.FieldDefn(f"{prefix}{attribute}", ogr.OFTReal)
                vector_layer.CreateField(field_defn)

        vector_feature_count = vector_layer.GetFeatureCount()

        if verbose:
            utils_base.progress(0, vector_feature_count, name="shape")

        for i in range(vector_feature_count):
            vector_feature = vector_layer.GetNextFeature()

            try:
                vector_geom = vector_feature.GetGeometryRef()
            except RuntimeWarning:
                # vector_geom.Buffer(0)
                raise RuntimeWarning("Invalid geometry at : ", i) from None

            if vector_geom is None:
                raise RuntimeError("Invalid geometry. Could not fix.")

            centroid = vector_geom.Centroid()
            vector_area = vector_geom.GetArea()
            vector_perimeter = vector_geom.Boundary().Length()

            if "ipq" or "compact" in shapes:
                vector_ipq = 0
                if vector_perimeter != 0:
                    vector_ipq = (4 * np.pi * vector_area) / vector_perimeter ** 2

            if "centroid" in shapes:
                vector_feature.SetField(f"{prefix}centroid_x", centroid.GetX())
                vector_feature.SetField(f"{prefix}centroid_y", centroid.GetY())

            if "hull" in shapes or "compact" in shapes:
                vector_hull = vector_geom.ConvexHull()
                hull_area = vector_hull.GetArea()
                hull_peri = vector_hull.Boundary().Length()
                hull_ratio = float(vector_area) / float(hull_area)
                compactness = np.sqrt(float(hull_ratio) * float(vector_ipq))

            if "area" in shapes:
                vector_feature.SetField(f"{prefix}area", vector_area)
            if "perimeter" in shapes:
                vector_feature.SetField(f"{prefix}perimeter", vector_perimeter)
            if "ipq" in shapes:
                vector_feature.SetField(f"{prefix}ipq", vector_ipq)
            if "hull" in shapes:
                vector_feature.SetField(f"{prefix}hull_area", hull_area)
                vector_feature.SetField(f"{prefix}hull_peri", hull_peri)
                vector_feature.SetField(f"{prefix}hull_ratio", hull_ratio)
            if "compact" in shapes:
                vector_feature.SetField(f"{prefix}compact", compactness)

            vector_layer.SetFeature(vector_feature)

            if verbose:
                utils_base.progress(i, vector_feature_count, name="shape")

        vector_layer.CommitTransaction()

    return out_path


def vector_add_shapes_in_place(
    vector: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
    shapes: Optional[List[str]] = None,
    prefix: str = "",
    allow_lists: bool = True,
    verbose: bool = False,
) -> Union[str, List[str]]:
    """
    Adds shape calculations to a vector such as area and perimeter.
    Can also add compactness measurements.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource, List[str, ogr.DataSource]]
        Vector layer(s) or path(s) to vector layer(s).
    
    shapes : Optional[List[str]], optional
        The shapes to calculate. The following a possible:
            * Area          (In same unit as projection)
            * Perimeter     (In same unit as projection)
            * IPQ           (0-1) given as (4*Pi*Area)/(Perimeter ** 2)
            * Hull Area     (The area of the convex hull. Same unit as projection)
            * Compactness   (0-1) given as sqrt((area / hull_area) * ipq)
            * Centroid      (Coordinate of X and Y)
        Default: all shapes.
    
    prefix : str, optional
        Prefix to add to the field names. Default: "".

    allow_lists : bool, optional
        If True, will accept a list of vectors. If False, will raise an error if a list is passed. Default: True.

    verbose : bool, optional
        If True, will print progress. Default: False.

    Returns
    -------
    out_path : str
        Path to the output vector.
    """
    utils_base._type_check(vector, [str, ogr.DataSource, [str, ogr.DataSource]], "vector")
    utils_base._type_check(shapes, [[str], None], "shapes")

    if not allow_lists and isinstance(vector, list):
        raise ValueError("Lists of vectors are not supported when allow_list is False.")

    vector_list = utils_base._get_variable_as_list(vector)
    output = utils_gdal._get_path_from_dataset_list(vector_list)

    for in_vector in vector_list:
        output.append(_vector_add_shapes_in_place(
            in_vector,
            shapes=shapes,
            prefix=prefix,
            verbose=verbose,
        ))

    if isinstance(vector, list):
        return output

    return output[0]


def vector_get_attribute_table(
    vector: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
    process_layer: int = -1,
    include_fids: bool = False,
    include_geometry: bool = False,
    include_attributes: bool = True,
    allow_lists: bool = True,
) -> Dict[str, Any]:
    """
    Get the attribute table(s) of a vector.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource, List[str, ogr.DataSource]]
        Vector layer(s) or path(s) to vector layer(s).

    process_layer : int, optional
        The layer to process. Default: -1 (all layers).

    include_fids : bool, optional
        If True, will include the FID column. Default: False.

    include_geometry : bool, optional
        If True, will include the geometry column. Default: False.

    include_attributes : bool, optional
        If True, will include the attribute columns. Default: True.

    allow_lists : bool, optional
        If True, will accept a list of vectors. If False, will raise an error if a list is passed. Default: True.
    
    Returns
    -------
    attribute_table : Dict[str, Any]
        The attribute table(s) of the vector(s).
    """
    utils_base._type_check(vector, [str, ogr.DataSource, [str, ogr.DataSource]], "vector")
    utils_base._type_check(process_layer, [int], "process_layer")
    utils_base._type_check(include_fids, [bool], "include_fids")
    utils_base._type_check(include_geometry, [bool], "include_geometry")
    utils_base._type_check(include_attributes, [bool], "include_attributes")
    utils_base._type_check(allow_lists, [bool], "allow_lists")

    ref = vector_open(vector)
    metadata = _vector_to_metadata(ref)

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

        if include_geometry:
            geom_defn = feature.GetGeometryRef()
            attributes.append(geom_defn.ExportToIsoWkt())

        attribute_table.append(attributes)

    attribute_table_header.insert(0, "fid")

    if include_geometry:
        attribute_table_header.append("geom")

    return attribute_table


def vector_filter_layer(
    vector: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
    layer_name_or_idx: Union[str, int],
    out_path: Optional[str] = None,
    prefix: str = "",
    suffix: str = "_layer",
    add_uuid: bool = False,
    overwrite: bool = True,
):
    """
    Filters a multi-layer vector source to a single layer.
    
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
    ref = vector_open(vector)
    meta = vector_to_metadata(ref, allow_lists=False)

    out_path = utils_io._get_output_paths(
        meta["path"],
        out_path,
        overwrite=overwrite,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
    )

    if isinstance(layer_name_or_idx, int):
        layer = ref.GetLayerByIndex(layer_name_or_idx)
    elif isinstance(layer_name_or_idx, str):
        layer = ref.GetLayer(layer_name_or_idx)
    else:
        raise RuntimeError("Wrong datatype for layer selection")

    driver = utils_gdal._get_vector_driver_name_from_path(out_path)

    destination = driver.CreateDataSource(out_path)
    destination.CopyLayer(layer, layer.GetName(), ["OVERWRITE=YES"])
    destination.FlushCache()

    return out_path
