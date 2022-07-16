"""
The basic module for interacting with vector data

TODO:
    - Improve documentation
    - repair vector
    - sanity checks: vectors_intersect, is_not_empty, does_vectors_match, match_vectors
    - rasterize - with antialiasing/weights
    - join by attribute + summary
    - join by location + summary
    - buffer, union, erase
    - Multithreaded processing
    - Rename layers function
"""

import sys; sys.path.append("../../") # Path: buteo/vector/io.py
import os
from uuid import uuid4

from osgeo import ogr, osr, gdal
import numpy as np

from buteo.raster.io import _raster_to_metadata
from buteo.utils.bbox import additional_bboxes
from buteo.utils.gdal_utils import (
    is_vector,
    is_raster,
    path_to_driver_vector,
)
from buteo.utils.core import (
    progress,
    remove_if_overwrite,
    overwrite_required,
    type_check,
    folder_exists,
)


def open_vector(
    vector,
    *,
    convert_mem_driver=True,
    writeable=True,
    layer=-1,
):
    """Opens a vector to an ogr.Datasource class.

    Args:
        vector (path | datasource): A path to a vector or a ogr datasource.

        convert_mem_driver (bool): Converts MEM driver vectors to /vsimem/ geopackage.

        writable (bool): Should the opened raster be writeable.

    Returns:
        A gdal.Dataset
    """
    type_check(vector, [str, ogr.DataSource, gdal.Dataset], "vector")
    type_check(convert_mem_driver, [bool], "convert_mem_driver")
    type_check(writeable, [bool], "writeable")
    type_check(layer, [int], "layer")

    opened = None
    try:
        if is_vector(vector):
            gdal.PushErrorHandler("CPLQuietErrorHandler")

            if isinstance(vector, str):
                opened = ogr.Open(vector, 1) if writeable else ogr.Open(vector, 0)
            elif isinstance(vector, ogr.DataSource):
                opened = vector
            else:
                raise Exception(f"Could not read input vector: {vector}")

            gdal.PopErrorHandler()
        elif is_raster(vector):
            temp_opened = None
            if isinstance(vector, str):
                gdal.PushErrorHandler("CPLQuietErrorHandler")

                temp_opened = (
                    gdal.Open(vector, 1) if writeable else gdal.Open(vector, 0)
                )

                gdal.PopErrorHandler()
            elif isinstance(vector, gdal.Dataset):
                temp_opened = vector
            else:
                raise Exception(f"Could not read input vector: {vector}")

            raster_metadata = _raster_to_metadata(temp_opened)
            opened = raster_metadata["get_extent_datasource"]()

        else:
            raise Exception(f"Could not read input vector: {vector}")
    except:
        raise Exception(f"Could not read input vector: {vector}")

    if opened is None:
        raise Exception(f"Could not read input vector: {vector}")

    driver = opened.GetDriver()
    driver_name = driver.GetName()

    if driver is None:
        raise Exception("Unable to parse the driver of vector.")

    if layer != -1:
        layer_count = opened.GetLayerCount()

        if layer > layer_count - 1:
            raise Exception(f"Requested a non-existing layer: {layer}")

        if layer_count > 1:
            driver_name = "Memory"

    if convert_mem_driver and driver_name == "Memory":
        path = opened.GetDescription()
        basename = os.path.basename(path)
        name = os.path.splitext(basename)[0]
        raster_name = f"/vsimem/{name}_{uuid4().int}.gpkg"
        driver = gdal.GetDriverByName("GPKG")

        if layer != -1:
            opened = driver.CreateDataSource(raster_name)
            orignal_layer = opened.GetLayerByIndex(layer)
            opened.CopyLayer(
                orignal_layer, orignal_layer.GetDescription(), ["OVERWRITE=YES"]
            )
        else:
            opened = driver.CreateCopy(raster_name, opened)

    return opened


def get_vector_path(vector):
    """Takes a string or a ogr.Datasource and returns its path.

    Args:
        vector (path | Dataset): A path to a vector or an ogr.Datasource.

    Returns:
        A string representing the path to the vector.
    """
    if isinstance(vector, str) and (len(vector) >= 8 and vector[0:8]) == "/vsimem/":
        return vector

    type_check(vector, [str, ogr.DataSource], "vector")

    opened = open_vector(vector, convert_mem_driver=True, writeable=False)
    try:
        path = str(opened.GetDescription())

        if len(path) >= 8 and path[0:8] == "/vsimem/":
            return path
        elif os.path.exists(path):
            return path
        else:
            raise Exception(f"Error while getting path from raster: {vector}")

    except:
        raise Exception(f"Error while getting path from raster: {vector}")


def to_vector_list(variable):
    """Reads a list of vectors and returns a list of paths to the vectors.

    Args:
        variable (list, path | Dataset): The vectors to generate paths for.

    Returns:
        A list of paths to the vectors.
    """
    return_list = []
    if isinstance(variable, list):
        return_list = variable
    else:
        if not isinstance(variable, (str, ogr.DataSource)):
            raise ValueError(f"Error in vector list: {variable}")

        return_list.append(get_vector_path(variable))

    if len(return_list) == 0:
        raise ValueError("Empty vector list.")

    return return_list


def _vector_to_metadata(
    vector,
    *,
    process_layer=-1,
):
    """OBS: Internal. Single output.

    Creates a dictionary with metadata about the vector layer.
    """
    type_check(vector, [str, ogr.DataSource], "vector")
    type_check(process_layer, [int], "process_layer")

    datasource = open_vector(vector, convert_mem_driver=False)

    vector_driver = datasource.GetDriver()

    path = datasource.GetDescription()
    basename = os.path.basename(path)
    name = os.path.splitext(basename)[0]
    ext = os.path.splitext(basename)[1]
    driver_name = vector_driver.GetName()

    in_memory = False
    if driver_name == "MEM":
        in_memory = True
    elif len(path) >= 8 and path[0:8] == "/vsimem/":
        in_memory = True

    layer_count = datasource.GetLayerCount()
    layers = []

    processed = False

    for layer_index in range(layer_count):

        if process_layer != -1 and layer_index != process_layer:
            continue

        layer = datasource.GetLayerByIndex(layer_index)

        x_min, x_max, y_min, y_max = layer.GetExtent()
        layer_name = layer.GetName()
        extent = [x_min, y_max, x_max, y_min]
        extent_ogr = [x_min, x_max, y_min, y_max]
        extent_dict = {
            "left": x_min,
            "top": y_max,
            "right": x_max,
            "bottom": y_min,
        }

        column_fid = layer.GetFIDColumn()
        column_geom = layer.GetGeometryColumn()

        if column_geom == "":
            column_geom = "geom"

        feature_count = layer.GetFeatureCount()

        projection_osr = layer.GetSpatialRef()
        projection = layer.GetSpatialRef().ExportToWkt()

        if processed is False:
            ds_projection = projection
            ds_projection_osr = projection_osr
            ds_x_min = x_min
            ds_x_max = x_max
            ds_y_min = y_min
            ds_y_max = y_max

            processed = True
        else:
            if x_min < ds_x_min:
                ds_x_min = x_min
            if x_max > ds_x_max:
                ds_x_max = x_max
            if y_min < ds_y_min:
                ds_y_min = y_min
            if y_max > ds_y_max:
                ds_y_max = y_max

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
            "projection": projection,
            "projection_osr": projection_osr,
            "geom_type": geom_type,
            "geom_type_ogr": geom_type_ogr,
            "field_count": field_count,
            "field_names": field_names,
            "field_types": field_types,
            "field_types_ogr": field_types_ogr,
            "extent": extent,
            "extent_ogr": extent_ogr,
            "extent_dict": extent_dict,
            "extent_wkt": None,
            "extent_latlng": None,
            "extent_gdal_warp_latlng": None,
            "extent_ogr_latlng": None,
            "extent_dict_latlng": None,
            "extent_wkt_latlng": None,
            "extent_geojson": None,
            "extent_geojson_dict": None,
            "extent_datasource": None,
            "extent_datasource_latlng": None,
        }

        layers.append(layer_dict)

    ds_extent = [ds_x_min, ds_y_max, ds_x_max, ds_y_min]
    ds_extent_ogr = [ds_x_min, ds_x_max, ds_y_min, ds_y_max]
    ds_extent_gdal_warp = [ds_x_min, ds_y_min, ds_x_max, ds_y_max]
    ds_extent_dict = {
        "left": ds_x_min,
        "top": ds_y_max,
        "right": ds_x_max,
        "bottom": ds_y_min,
    }

    ds_expanded_extents = additional_bboxes(ds_extent_ogr, projection_osr)

    metadata = {
        "path": path,
        "basename": basename,
        "name": name,
        "ext": ext,
        "in_memory": in_memory,
        "projection": ds_projection,
        "projection_osr": ds_projection_osr,
        "driver": driver_name,
        "x_min": ds_x_min,
        "y_max": ds_y_max,
        "x_max": ds_x_max,
        "y_min": ds_y_min,
        "is_vector": True,
        "is_raster": False,
        "layer_count": layer_count,
        "layers": layers,
        "extent": ds_extent,
        "extent_dict": ds_extent_dict,
        "extent_ogr": ds_extent_ogr,
        "extent_gdal_warp": ds_extent_gdal_warp,
        "extent_gdal_warp_latlng": ds_expanded_extents["extent_gdal_warp_latlng"],
        "extent_wkt": ds_expanded_extents["extent_wkt"],
        "extent_latlng": ds_expanded_extents["extent_latlng"],
        "extent_ogr_latlng": ds_expanded_extents["extent_ogr_latlng"],
        "extent_dict_latlng": ds_expanded_extents["extent_dict_latlng"],
        "extent_wkt_latlng": ds_expanded_extents["extent_wkt_latlng"],
        "extent_geojson": ds_expanded_extents["extent_geojson"],
        "extent_geojson_dict": ds_expanded_extents["extent_geojson_dict"],
        "extent_datasource": None,
        "extent_datasource_latlng": None,
    }

    def ds_get_extent_datasource():
        extent_name = f"/vsimem/{name}_{uuid4().int}_extent.gpkg"

        driver = ogr.GetDriverByName("GPKG")
        extent_ds = driver.CreateDataSource(extent_name)
        layer = extent_ds.CreateLayer(
            "extent_ogr", ds_projection_osr, ogr.wkbPolygon
        )

        feature = ogr.Feature(layer.GetLayerDefn())
        extent_geom = ogr.CreateGeometryFromWkt(metadata["extent_wkt"], ds_projection_osr)
        feature.SetGeometry(extent_geom)
        layer.CreateFeature(feature)

        feature = None
        layer.SyncToDisk()

        return extent_name


    def ds_get_extent_datasource_latlng():
        extent_latlng_name = f"/vsimem/{name}_{uuid4().int}_extent_latlng.gpkg"

        target_projection = osr.SpatialReference()
        target_projection.ImportFromEPSG(4326)

        driver = ogr.GetDriverByName("GPKG")
        extent_ds_latlng = driver.CreateDataSource(extent_latlng_name)
        layer = extent_ds_latlng.CreateLayer(
            "extent_latlng", target_projection, ogr.wkbPolygon
        )

        feature = ogr.Feature(layer.GetLayerDefn())
        extent_geom_latlng = ogr.CreateGeometryFromWkt(metadata["extent_wkt_latlng"], target_projection)
        feature.SetGeometry(extent_geom_latlng)
        layer.CreateFeature(feature)

        feature = None
        layer.SyncToDisk()

        return extent_latlng_name

    def ds_get_extent_geom():
        return ogr.CreateGeometryFromWkt(metadata["extent_wkt"], ds_projection_osr)
    

    def ds_get_extent_geom_latlng():
        target_projection = osr.SpatialReference()
        target_projection.ImportFromEPSG(4326)
        return ogr.CreateGeometryFromWkt(metadata["extent_wkt_latlng"], target_projection)


    metadata["extent_datasource"] = ds_get_extent_datasource
    metadata["extent_datasource_latlng"] = ds_get_extent_datasource_latlng
    metadata["extent_geom"] = ds_get_extent_geom
    metadata["extent_geom_latlng"] = ds_get_extent_geom_latlng

    # Individual layer extents
    for layer_index in range(layer_count):
        layer_dict = layers[layer_index]

        layer_expanded_extents = expand_extent(layer_dict["extent_ogr"], projection_osr)


        for key, value in layer_expanded_extents.items():
            layers[layer_index][key] = value

            def layer_get_extent_datasource():
                extent_name = f"/vsimem/{name}_l{layer_index}_{uuid4().int}_extent.fgb"

                driver = ogr.GetDriverByName("FlatGeobuf")
                extent_ds = driver.CreateDataSource(extent_name)
                layer = extent_ds.CreateLayer(
                    "extent_ogr", ds_projection_osr, ogr.wkbPolygon
                )

                feature = ogr.Feature(layer.GetLayerDefn())
                extent_geom = ogr.CreateGeometryFromWkt(layers[layer_index]["extent_wkt"], ds_projection_osr)
                feature.SetGeometry(extent_geom)
                layer.CreateFeature(feature)

                feature = None
                layer.SyncToDisk()

                return extent_name


            def layer_get_extent_datasource_latlng():
                extent_latlng_name = f"/vsimem/{name}_l{layer_index}_{uuid4().int}_extent_latlng.fgb"

                target_projection = osr.SpatialReference()
                target_projection.ImportFromEPSG(4326)

                driver = ogr.GetDriverByName("FlatGeobuf")
                extent_ds_latlng = driver.CreateDataSource(extent_latlng_name)
                layer = extent_ds_latlng.CreateLayer(
                    "extent_latlng", target_projection, ogr.wkbPolygon
                )

                feature = ogr.Feature(layer.GetLayerDefn())
                extent_geom_latlng = ogr.CreateGeometryFromWkt(layers[layer_index]["extent_wkt_latlng"], target_projection)
                feature.SetGeometry(extent_geom_latlng)
                layer.CreateFeature(feature)

                feature = None
                layer.SyncToDisk()

                return extent_latlng_name


            def layer_get_extent_geom():
                return ogr.CreateGeometryFromWkt(layers[layer_index]["extent_wkt"], ds_projection_osr)
            

            def layer_get_extent_geom_latlng():
                target_projection = osr.SpatialReference()
                target_projection.ImportFromEPSG(4326)
                return ogr.CreateGeometryFromWkt(layers[layer_index]["extent_wkt_latlng"], target_projection)


            layers[layer_index]["extent_datasource"] = layer_get_extent_datasource
            layers[layer_index]["extent_datasource_latlng"] = layer_get_extent_datasource_latlng
            layers[layer_index]["extent_geom"] = layer_get_extent_geom
            layers[layer_index]["extent_geom_latlng"] = layer_get_extent_geom_latlng

    return metadata


def vector_to_metadata(
    vector,
    process_layer=-1,
):
    """Creates a dictionary with metadata about the vector layer.

    Args:
        vector (path | DataSource): The vector to analyse.

    Returns:
        A dictionary containing the metadata.
    """
    type_check(vector, [list, str, ogr.DataSource], "vector")
    type_check(process_layer, [int], "process_layer")

    vector_list = to_vector_list(vector)

    output = []

    for in_vector in vector_list:
        output.append(
            _vector_to_metadata(in_vector, process_layer=process_layer)
        )

    if isinstance(vector, list):
        return output

    return output[0]


def ready_io_vector(
    vector,
    out_path,
    *,
    overwrite=True,
    add_uuid=False,
    prefix="",
    postfix="",
):
    type_check(vector, [list, str, ogr.DataSource], "vector")
    type_check(out_path, [list, str], "out_path", allow_none=True)
    type_check(overwrite, [bool], "overwrite")
    type_check(prefix, [str], "prefix")
    type_check(postfix, [str], "postfix")

    vector_list = to_vector_list(vector)

    if isinstance(out_path, list):
        if len(vector_list) != len(out_path):
            raise ValueError(
                "The length of vector_list must equal the length of the out_path"
            )

    # Check if folder exists and is required.
    if len(vector_list) > 1 and isinstance(out_path, str):
        if not os.path.dirname(os.path.abspath(out_path)):
            raise ValueError(
                f"Output folder does not exist. Please create first. {out_path}"
            )

    # Generate output names
    path_list = []
    for index, in_vector in enumerate(vector_list):
        metadata = _vector_to_metadata(in_vector)

        name = metadata["name"]

        if add_uuid:
            uuid = uuid4().int
        else:
            uuid = ""

        if out_path is None:
            path = f"/vsimem/{prefix}{name}{uuid}{postfix}.gpkg"
        elif isinstance(out_path, str):
            if folder_exists(out_path):
                path = os.path.join(out_path, f"{prefix}{name}{uuid}{postfix}.tif")
            else:
                path = out_path
        elif isinstance(out_path, list):
            if out_path[index] is None:
                path = f"/vsimem/{prefix}{name}{uuid}{postfix}.tif"
            elif isinstance(out_path[index], str):
                path = out_path[index]
            else:
                raise ValueError(f"Unable to parse out_path: {out_path}")
        else:
            raise ValueError(f"Unable to parse out_path: {out_path}")

        overwrite_required(path, overwrite)
        path_list.append(path)

    return (vector_list, path_list)


def _vector_to_memory(
    vector,
    memory_path=None,
    copy_if_already_in_memory=True,
    *,
    layer_to_extract=-1,
):
    """OBS: Internal. Single output.

    Copies a vector source to memory.
    """
    type_check(vector, [str, ogr.DataSource], "vector")
    type_check(memory_path, [str], "memory_path", allow_none=True)
    type_check(layer_to_extract, [int], "layer_to_extract")

    ref = open_vector(vector)
    path = get_vector_path(ref)
    metadata = _vector_to_metadata(ref)
    name = metadata["name"]

    if not copy_if_already_in_memory and metadata["in_memory"]:
        if layer_to_extract == -1:
            return path

    if memory_path is not None:
        if memory_path[0:8] == "/vsimem/":
            vector_name = memory_path
        else:
            vector_name = f"/vsimem/{memory_path}"
        driver = ogr.GetDriverByName(path_to_driver_vector(memory_path))
    else:
        vector_name = f"/vsimem/{name}_{uuid4().int}.gpkg"
        driver = ogr.GetDriverByName("GPKG")

    if driver is None:
        raise Exception(f"Error while parsing driver for: {vector}")

    copy = driver.CreateDataSource(vector_name)

    for layer_idx in range(metadata["layer_count"]):
        if layer_to_extract is not None and layer_idx != layer_to_extract:
            continue

        layername = metadata["layers"][layer_idx]["layer_name"]
        copy.CopyLayer(ref.GetLayer(layer_idx), layername, ["OVERWRITE=YES"])

    return vector_name


def vector_to_memory(
    vector,
    memory_path=None,
    copy_if_already_in_memory=False,
    *,
    layer_to_extract=-1,
):
    """Copies a vector source to memory.

    Args:
        vector (list | path | DataSource): The vector to copy to memory

    **kwargs:
        memory_path (str | None): If a path is provided, uses the
        appropriate driver and uses the VSIMEM gdal system.
        Example: vector_to_memory(clip_ref.tif, "clip_geom.gpkg")
        /vsimem/ is autumatically added.

        layer_to_extract (int | None): The layer in the vector to copy.
        if None is specified, all layers are copied.

        opened (bool): If a memory path is specified, the default is
        to return a path. If open is supplied. The vector is opened
        before returning.

    Returns:
        An in-memory ogr.DataSource. If a memory path was provided a
        string for the in-memory location is returned.
    """
    type_check(vector, [list, str, ogr.DataSource], "vector")
    type_check(memory_path, [list, str], "memory_pathout_path", allow_none=True)
    type_check(layer_to_extract, [int], "layer_to_extract")

    vector_list, path_list = ready_io_vector(vector, memory_path)

    output = []
    for index, in_vector in enumerate(vector_list):
        path = path_list[index]

        output.append(
            _vector_to_memory(
                in_vector,
                memory_path=path,
                layer_to_extract=layer_to_extract,
                copy_if_already_in_memory=copy_if_already_in_memory,
            )
        )

    if isinstance(vector, list):
        return output

    return output[0]


def _vector_to_disk(
    vector,
    out_path,
    *,
    overwrite=True,
):
    """OBS: Internal. Single output.

    Copies a vector source to disk.
    """
    type_check(vector, [str, ogr.DataSource], "vector")
    type_check(out_path, [str], "out_path")
    type_check(overwrite, [bool], "overwrite")

    overwrite_required(out_path, overwrite)

    datasource = open_vector(vector)
    metadata = _vector_to_metadata(vector)

    if not os.path.dirname(os.path.abspath(out_path)):
        raise ValueError(
            f"Output folder does not exist. Please create first. {out_path}"
        )

    driver = ogr.GetDriverByName(path_to_driver_vector(out_path))

    if driver is None:
        raise Exception(f"Error while parsing driver for: {vector}")

    remove_if_overwrite(out_path, overwrite)

    copy = driver.CreateDataSource(out_path)

    for layer_idx in range(metadata["layer_count"]):
        layer_name = metadata["layers"][layer_idx]["layer_name"]
        copy.CopyLayer(
            datasource.GetLayer(layer_idx), str(layer_name), ["OVERWRITE=YES"]
        )

    # Flush to disk
    copy = None

    return out_path


def vector_to_disk(
    vector,
    out_path,
    *,
    overwrite=True,
):
    """Copies a vector source to disk.

    Args:
        vector (path | DataSource): The vector to copy to disk

        out_path (path): The destination to save to.

    **kwargs:
        overwite (bool): Is it possible to overwrite the out_path if it exists.

    Returns:
        An path to the created vector.
    """
    type_check(vector, [str, ogr.DataSource], "vector")
    type_check(out_path, [str], "out_path")
    type_check(overwrite, [bool], "overwrite")

    vector_list, path_list = ready_io_vector(vector, out_path, overwrite=overwrite)

    output = []
    for index, in_vector in enumerate(vector_list):
        path = path_list[index]
        output.append(_vector_to_disk(in_vector, path, overwrite=overwrite))

    if isinstance(vector, list):
        return output

    return output[0]


def filter_vector(vector, filter_where, process_layer=0):
    metadata = _vector_to_metadata(vector)

    name = f"/vsimem/{uuid4().int}_filtered.shp"

    projection = metadata["projection_osr"]

    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.CreateDataSource(name)
    og_ds = open_vector(vector)

    added = 0

    for index, _layer in enumerate(metadata["layers"]):
        if index != process_layer:
            continue

        features = metadata["layers"][index]["feature_count"]
        field_names = metadata["layers"][index]["field_names"]

        if filter_where[0] not in field_names:
            continue

        geom_type = metadata["layers"][index]["geom_type_ogr"]

        og_layer = og_ds.GetLayer(index)
        ds_layer = ds.CreateLayer(f"filtered_{uuid4().int}", projection, geom_type)

        found_match = False
        for _ in range(features):
            feature = og_layer.GetNextFeature()

            field_val = feature.GetField(filter_where[0])

            if field_val == filter_where[1]:
                ds_layer.CreateFeature(feature.Clone())

                found_match = True

        ds_layer.SyncToDisk()

        if found_match:
            added += 1

    if added == 0:
        raise ValueError("No matches found.")

    return name


def vector_add_index(vector):
    """Adds a spatial index to the vector if it doesn't have one.

    Args:
        vector (list, path | vector): The vector to add the index to.

    Returns:
        A path to the original vector.
    """
    type_check(vector, [list, str, ogr.DataSource], "vector")

    vector_list = to_vector_list(vector)

    try:
        for in_vector in vector_list:
            metadata = _vector_to_metadata(in_vector)
            ref = open_vector(in_vector)

            for layer in metadata["layers"]:
                name = layer["layer_name"]
                geom = layer["column_geom"]

                sql = f"SELECT CreateSpatialIndex('{name}', '{geom}') WHERE NOT EXISTS (SELECT HasSpatialIndex('{name}', '{geom}'));"
                ref.ExecuteSQL(sql, dialect="SQLITE")
    except:
        raise Exception(f"Error while creating indices for {vector}")

    return vector_list


def vector_feature_to_layer(vector, layer=1):
    """Adds a spatial index to the vector if it doesn't have one.

    Args:
        vector (list, path | vector): The vector to add the index to.

    Returns:
        A path to the original vector.
    """
    type_check(vector, [list, str, ogr.DataSource], "vector")

    vector_list = to_vector_list(vector)

    try:
        for in_vector in vector_list:
            metadata = _vector_to_metadata(in_vector)
            ref = open_vector(in_vector)

            for layer in metadata["layers"]:
                name = layer["layer_name"]
                geom = layer["column_geom"]

                sql = f"SELECT CreateSpatialIndex('{name}', '{geom}') WHERE NOT EXISTS (SELECT HasSpatialIndex('{name}', '{geom}'));"
                ref.ExecuteSQL(sql, dialect="SQLITE")
    except:
        raise Exception(f"Error while creating indices for {vector}")

    return vector_list


def _vector_add_shapes(
    vector,
    shapes=["area", "perimeter", "ipq", "hull", "compactness", "centroid"],
):
    """OBS: Internal. Single output.

    Adds shape calculations to a vector such as area and perimeter.
    Can also add compactness measurements.
    """
    type_check(vector, [str, ogr.DataSource], "vector")
    type_check(shapes, [list], "shapes")

    datasource = open_vector(vector)
    out_path = get_vector_path(datasource)
    metadata = _vector_to_metadata(datasource)

    for index in range(metadata["layer_count"]):
        vector_current_fields = metadata["layers"][index]["field_names"]
        vector_layer = datasource.GetLayer(index)

        vector_layer.StartTransaction()

        # Add missing fields
        for attribute in shapes:
            if attribute == "centroid":
                if "centroid_x" not in vector_current_fields:
                    field_defn = ogr.FieldDefn("centroid_x", ogr.OFTReal)
                    vector_layer.CreateField(field_defn)

                if "centroid_y" not in vector_current_fields:
                    field_defn = ogr.FieldDefn("centroid_y", ogr.OFTReal)
                    vector_layer.CreateField(field_defn)

            elif attribute not in vector_current_fields:
                field_defn = ogr.FieldDefn(attribute, ogr.OFTReal)
                vector_layer.CreateField(field_defn)

        vector_feature_count = vector_layer.GetFeatureCount()
        for i in range(vector_feature_count):
            vector_feature = vector_layer.GetNextFeature()

            try:
                vector_geom = vector_feature.GetGeometryRef()
            except:
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
                vector_feature.SetField("centroid_x", centroid.GetX())
                vector_feature.SetField("centroid_y", centroid.GetY())

            if "hull" in shapes or "compact" in shapes:
                vector_hull = vector_geom.ConvexHull()
                hull_area = vector_hull.GetArea()
                hull_peri = vector_hull.Boundary().Length()
                hull_ratio = float(vector_area) / float(hull_area)
                compactness = np.sqrt(float(hull_ratio) * float(vector_ipq))

            if "area" in shapes:
                vector_feature.SetField("area", vector_area)
            if "perimeter" in shapes:
                vector_feature.SetField("perimeter", vector_perimeter)
            if "ipq" in shapes:
                vector_feature.SetField("ipq", vector_ipq)
            if "hull" in shapes:
                vector_feature.SetField("hull_area", hull_area)
                vector_feature.SetField("hull_peri", hull_peri)
                vector_feature.SetField("hull_ratio", hull_ratio)
            if "compact" in shapes:
                vector_feature.SetField("compact", compactness)

            vector_layer.SetFeature(vector_feature)

            progress(i, vector_feature_count, name="shape")

        vector_layer.CommitTransaction()

    return out_path


def vector_add_shapes(
    vector,
    shapes=["area", "perimeter", "ipq", "hull", "compactness", "centroid"],
):
    """Adds shape calculations to a vector such as area and perimeter.
        Can also add compactness measurements.

    Args:
        vector (path | vector): The vector to add shapes to.

    **kwargs:
        shapes (list): The shapes to calculate. The following a possible:
            * Area          (In same unit as projection)
            * Perimeter     (In same unit as projection)
            * IPQ           (0-1) given as (4*Pi*Area)/(Perimeter ** 2)
            * Hull Area     (The area of the convex hull. Same unit as projection)
            * Compactness   (0-1) given as sqrt((area / hull_area) * ipq)
            * Centroid      (Coordinate of X and Y)

    Returns:
        Either the path to the updated vector or a list of the input vectors.
    """
    type_check(vector, [list, str, ogr.DataSource], "vector")
    type_check(shapes, [list], "shapes")

    vector_list = to_vector_list(vector)

    output = []
    for in_vector in vector_list:
        output.append(_vector_add_shapes(in_vector, shapes=shapes))

    if isinstance(vector, list):
        return output

    return output[0]
