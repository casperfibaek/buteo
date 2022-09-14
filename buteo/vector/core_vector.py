"""
### Basic IO functions for working with Vectprs ###

The basic module for interacting with vector data

TODO:
    * more attribute functions.
    * repair vector
    * sanity checks: vectors_intersect, is_not_empty, does_vectors_match, match_vectors
    * rasterize - antialiasing and weights
    * join by attribute, location, summary
    * buffer, union, erase, etc..
"""

# Standard library
import sys; sys.path.append("../../")
import os

# External
import numpy as np
from osgeo import ogr, gdal, osr

# Internal
from buteo.utils import bbox_utils, core_utils, gdal_utils


def _open_vector(vector, *, writeable=True, allow_raster=True):
    """ Internal. """
    assert isinstance(vector, (str, ogr.DataSource)), "vector must be a path or a DataSource"

    opened = None
    if gdal_utils.is_vector(vector):
        if isinstance(vector, str):
            gdal.PushErrorHandler("CPLQuietErrorHandler")
            opened = ogr.Open(vector, gdal.GF_Write) if writeable else ogr.Open(vector, gdal.GF_Read)
            gdal.PopErrorHandler()
        elif isinstance(vector, ogr.DataSource):
            opened = vector
        else:
            raise Exception(f"Could not read input vector: {vector}")

    elif allow_raster and gdal_utils.is_raster(vector):
        if isinstance(vector, str):
            gdal.PushErrorHandler("CPLQuietErrorHandler")
            opened = gdal.Open(vector, gdal.GF_Write) if writeable else ogr.Open(vector, gdal.GF_Read)
            gdal.PopErrorHandler()
        elif isinstance(vector, gdal.Dataset):
            opened = vector
        else:
            raise Exception(f"Could not read input vector: {vector}")

        bbox = bbox_utils.get_bbox_from_geotransform(
            opened.GetGeoTransform(),
            opened.RasterXSize,
            opened.RasterYSize,
        )

        projection_wkt = opened.GetProjection()
        projection_osr = osr.SpatialReference()
        projection_osr.ImportFromWkt(projection_wkt)

        vector_bbox = bbox_utils.convert_bbox_to_vector(bbox, projection_osr)
        opened = ogr.Open(vector_bbox, gdal.GF_Write) if writeable else ogr.Open(vector_bbox, gdal.GF_Read)

    else:
        raise Exception(f"Could not read input vector: {vector}")

    if opened is None:
        raise Exception(f"Could not read input vector: {vector}")

    return opened


def open_vector(
    vector,
    *,
    writeable=True,
    allow_raster=True,
    allow_lists=True,
):
    """
    Opens a vector to an ogr.Datasource class.

    ## Args:
    `vector` (_str_/_ogr.DataSource_/_gdal.Dataset_): The vector to open. If a
    raster is supplied the bounding box is opened as a vector. </br>

    ## Kwargs:
    `writeable` (_bool_): If True, the vector is opened in write mode. (Default: **True**) </br>
    `allow_raster` (_bool_): If True, a raster will be opened as a vector bounding box. (Default: **True**) </br>
    `allow_lists` (_bool_): If True, the input can be a list of vectors. (Default: **True**) </br>

    ## Returns:
    (_ogr.DataSource_/_list_): The opened vector(s).
    """
    core_utils.type_check(vector, [str, ogr.DataSource, gdal.Dataset, [str, ogr.DataSource, gdal.Dataset]], "vector")
    core_utils.type_check(writeable, [bool], "writeable")

    if isinstance(vector, list) and not allow_lists:
        raise ValueError("Cannot open a list of vectors when allow_list is False.")

    vectors = core_utils.ensure_list(vector)

    output = []
    for element in vectors:
        output.append(_open_vector(element, writeable=writeable, allow_raster=allow_raster))

    if isinstance(vector, list):
        return output

    return output[0]


def _vector_to_metadata(vector):
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

    in_memory = gdal_utils.is_in_memory(datasource)

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
        }

        layer_bboxes = bbox_utils.additional_bboxes(layer_bbox, projection_osr)

        for key, value in layer_bboxes.items():
            layer_dict[key] = value


        ## MOVE TO A SINGLE FUNCTION
        def get_bbox_as_vector_layer():
            return bbox_utils.convert_bbox_to_vector(layer_bbox, projection_osr) # pylint: disable=cell-var-from-loop


        def get_bbox_as_vector_latlng_layer():
            projection_osr_latlng = osr.SpatialReference()
            projection_osr_latlng.ImportFromEPSG(4326)

            return bbox_utils.convert_bbox_to_vector(layer_dict["bbox_latlng"], projection_osr_latlng)  # pylint: disable=cell-var-from-loop


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
    }

    vector_bboxes = bbox_utils.additional_bboxes(vector_bbox, projection_osr)

    for key, value in vector_bboxes.items():
        metadata[key] = value


    def get_bbox_as_vector():
        return bbox_utils.convert_bbox_to_vector(vector_bbox, projection_osr) # pylint: disable=cell-var-from-loop


    def get_bbox_as_vector_latlng():
        projection_osr_latlng = osr.SpatialReference()
        projection_osr_latlng.ImportFromEPSG(4326)

        return bbox_utils.convert_bbox_to_vector(metadata["bbox_latlng"], projection_osr_latlng)  # pylint: disable=cell-var-from-loop


    metadata["get_bbox_vector"] = get_bbox_as_vector
    metadata["get_bbox_vector_latlng"] = get_bbox_as_vector_latlng

    return metadata


def vector_to_metadata(vector, *, allow_lists=True):
    """
    Creates a dictionary with metadata about the vector layer.

    ## Args:
    `vector` (_ogr.DataSource_/_str_/_list_): A vector layer(s) or path(s) to a vector file.

    ## Kwargs:
    `allow_lists` (_bool_): If **True**, vector can be a list of vector layers or paths. If `False`, `vector` must be a single vector layer or path. (default: **True**)

    ## Returns:
    (_dict_/_list_) A dictionary with metadata about the vector layer(s) or a list of dictionaries with metadata about the vector layer(s).
    """
    core_utils.type_check(vector, [str, ogr.DataSource, [str, ogr.DataSource]], "vector")
    core_utils.type_check(allow_lists, [bool], "allow_lists")

    if isinstance(vector, list) and not allow_lists:
        raise ValueError("The vector parameter cannot be a list when allow_lists is False.")

    vector_list = core_utils.ensure_list(vector)

    if not gdal_utils.is_vector_list(vector_list):
        raise ValueError("The vector parameter must be a list of vector layers.")

    output = []

    for in_vector in vector_list:
        output.append(_vector_to_metadata(in_vector))

    if isinstance(vector, list):
        return output

    return output[0]


def _filter_vector(
    vector,
    filter_function,
    out_path=None,
    *,
    process_layer=-1,
    prefix="",
    suffix="",
    overwrite=True,
):
    """ Internal. """
    assert isinstance(vector, (str, ogr.DataSource)), "vector must be a string or an ogr.DataSource object."
    assert isinstance(filter_function, (type(lambda: True))), "filter_function must be a function."

    metadata = _vector_to_metadata(vector)

    if out_path is None:
        out_path = gdal_utils.create_memory_path(
            gdal_utils.get_path_from_dataset(vector),
            prefix=prefix,
            suffix=suffix,
            add_uuid=True,
        )

    assert core_utils.is_valid_output_path(out_path, overwrite=overwrite), f"out_path is not a valid output path. {out_path}"

    projection = metadata["projection_osr"]

    driver = gdal_utils.path_to_driver_vector(out_path)

    datasource_destination = driver.CreateDataSource(out_path)
    datasource_original = open_vector(vector)

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


def filter_vector(
    vector,
    filter_function,
    *,
    out_path=None,
    process_layer=-1,
    allow_lists=True,
    prefix="",
    suffix="",
    add_uuid=False,
    overwrite=True,
):
    """
    Filters a vector using its attribute table and a function.

    ## Args:
    `vector` (_ogr.DataSource_/_str_/_list_): A vector layer(s) or path(s) to a vector file.
    `filter_function` (_function_): A function that takes a dictionary of attributes and returns a boolean.

    ## Kwargs:
    `out_path` (_str_): Path to the output vector file. If `None`, a memory vector will be created. (Default: **None**) </br>
    `process_layer` (_int_): The index of the layer to process. If `-1`, all layers will be processed. (Default: **-1**) </br>
    `allow_lists` (_bool_): If **True**, vector can be a list of vector layers or paths. If `False`, `vector` must be a single vector layer or path. (default: **True**) </br>
    `prefix` (_str_): A prefix to add to the output vector file. (Default: **""**) </br>
    `suffix` (_str_): A suffix to add to the output vector file. (Default: **""**) </br>
    `add_uuid` (_bool_): If **True**, a UUID will be added to the output vector file. (Default: **False**) </br>
    `overwrite` (_bool_): If **True**, the output vector file will be overwritten if it already exists. (Default: **True**) </br>

    ## Returns:
    (_str_/_list_): Path(s) to the output vector file(s).
    """
    core_utils.type_check(vector, [str, ogr.DataSource, [str, ogr.DataSource]], "vector")
    core_utils.type_check(filter_function, [type(lambda: True)], "filter_function")
    core_utils.type_check(process_layer, [int], "process_layer")
    core_utils.type_check(allow_lists, [bool], "allow_lists")

    if isinstance(vector, list) and not allow_lists:
        raise ValueError("The vector parameter cannot be a list when allow_lists is False.")

    vector_list = core_utils.ensure_list(vector)

    if not gdal_utils.is_vector_list(vector_list):
        raise ValueError("The vector parameter must be a list of vector layers.")

    path_list = gdal_utils.create_output_path_list(
        vector_list,
        out_path=out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        overwrite=overwrite,
    )

    output = []

    for index, in_vector in vector_list:
        output.append(_filter_vector(
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


def vector_add_index(vector, allow_lists=True):
    """
    Adds a spatial index to the vector in place, if it doesn't have one.

    ## Args:
    (_ogr.DataSource_/_str_/_list_): A vector layer(s) or path(s) to add indices to.

    ## Kwargs:
    `allow_lists` (_bool_): If **True**, vector can be a list of vector layers or paths. If `False`, `vector` must be a single vector layer or path. (default: **True**)

    ## Returns:
    (_str_/_list_): Path(s) to the input rasters.
    """
    core_utils.type_check(vector, [str, ogr.DataSource, [str, ogr.DataSource]], "vector")

    if isinstance(vector, list) and not allow_lists:
        raise ValueError("The vector parameter cannot be a list when allow_lists is False.")

    vector_list = core_utils.ensure_list(vector)

    if not gdal_utils.is_vector_list(vector_list):
        raise ValueError("The vector parameter must be a list of vector layers.")

    output = gdal_utils.get_path_from_dataset_list(vector_list)

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
        raise Exception(f"Error while creating indices for {vector}") from None

    if isinstance(vector, list):
        return output

    return output[0]


def _vector_add_shapes_in_place(vector, *, shapes=None, prefix="", verbose=False):
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
    out_path = gdal_utils.get_path_from_dataset(datasource, dataset_type="vector")
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
            core_utils.progress(0, vector_feature_count, name="shape")

        for i in range(vector_feature_count):
            vector_feature = vector_layer.GetNextFeature()

            try:
                vector_geom = vector_feature.GetGeometryRef()
            except Exception:
                vector_geom.Buffer(0)
                Warning("Invalid geometry at : ", i)

            if vector_geom is None:
                raise Exception("Invalid geometry. Could not fix.")

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
                core_utils.progress(i, vector_feature_count, name="shape")

        vector_layer.CommitTransaction()

    return out_path


def vector_add_shapes_in_place(vector, *, shapes=None, prefix="", allow_lists=True, verbose=False):
    """
    Adds shape calculations to a vector such as area and perimeter.
    Can also add compactness measurements.

    ## Args:
    `vector` (_str_/_ogr.DataSource_/_list_): Vector layer(s) or path(s) to vector layer(s).

    ## Kwargs:
    `shapes` (_list_/_None_): The shapes to calculate. The following a possible: </br>
        * Area          (In same unit as projection)
        * Perimeter     (In same unit as projection)
        * IPQ           (0-1) given as (4*Pi*Area)/(Perimeter ** 2)
        * Hull Area     (The area of the convex hull. Same unit as projection)
        * Compactness   (0-1) given as sqrt((area / hull_area) * ipq)
        * Centroid      (Coordinate of X and Y)
    The default is all shapes. </br>
    `prefix` (_str_): Prefix to add to the field names.
    `allow_lists` (_bool_): If True, will accept a list of vectors. If False, will raise an error if a list is passed.
    `verbose` (_bool_): If True, will print progress.

    ## Returns:
    (_str_/_list_): Path(s) to the original rasters that have been augmented in place.

    Returns:
        Either the path to the updated vector or a list of the input vectors.
    """
    core_utils.type_check(vector, [str, ogr.DataSource, [str, ogr.DataSource]], "vector")
    core_utils.type_check(shapes, [[str], None], "shapes")

    if not allow_lists and isinstance(vector, list):
        raise ValueError("Lists of vectors are not supported when allow_list is False.")

    vector_list = core_utils.ensure_list(vector)
    output = gdal_utils.get_path_from_dataset_list(vector_list)

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
    vector,
    process_layer=-1,
    include_fids=False,
    include_geometry=False,
    include_attributes=True,
    allow_lists=True,
):
    """
    Get the attribute table(s) of a vector.
    """
    core_utils.type_check(vector, [str, ogr.DataSource, [str, ogr.DataSource]], "vector")
    core_utils.type_check(process_layer, [int], "process_layer")
    core_utils.type_check(include_fids, [bool], "include_fids")
    core_utils.type_check(include_geometry, [bool], "include_geometry")
    core_utils.type_check(include_attributes, [bool], "include_attributes")
    core_utils.type_check(allow_lists, [bool], "allow_lists")

    ref = open_vector(vector)
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
