import sys

sys.path.append("../../")
import os
import numpy as np
from uuid import uuid4
from typing import Union, List, Dict, Optional, Any
from osgeo import ogr, osr

from buteo.project_types import Metadata_vector_layer, Number, Metadata_vector
from buteo.gdal_utils import (
    to_vector_list,
    vector_to_reference,
    is_vector,
    path_to_driver,
    advanced_extents,
)
from buteo.utils import progress, remove_if_overwrite, type_check


# TODO:
#   - repair vector
#   - sanity checks: vectors_intersect, is_not_empty, does_vectors_match, match_vectors
#   - rasterize - with antialiasing/weights
#   - join by attribute + summary
#   - join by location + summary
#   - buffer, union, erase
#   - multithreaded processing
#   - Rename layers function


def vector_to_metadata(
    vector: Union[List[Union[ogr.DataSource, str]], ogr.DataSource, str],
    process_layer: int = -1,
    simple: bool = True,
) -> Union[Metadata_vector, List[Metadata_vector]]:
    """ Creates a dictionary with metadata about the vector layer.

    Args:
        vector (list | path | DataSource): The vector to analyse.

    **kwargs:
        simple (bool): Should the metadata include a
            footprint of the raster in wgs84. Requires a reprojection
            check do not use it if not required and performance is important.

    Returns:
        A dictionary containing the metadata.
    """
    type_check(vector, [list, str, ogr.DataSource], "vector")
    type_check(simple, [bool], "simple")

    vectors = to_vector_list(vector)

    metadatas: List[Metadata_vector] = []

    for in_vector in vectors:
        datasource: ogr.DataSource = vector_to_reference(in_vector)

        vector_driver: ogr.Driver = datasource.GetDriver()

        path: str = datasource.GetDescription()
        basename: str = os.path.basename(path)
        name: str = os.path.splitext(basename)[0]
        ext: str = os.path.splitext(basename)[1]
        driver_name: str = vector_driver.GetName()

        layer_count: int = datasource.GetLayerCount()
        layers: List[Metadata_vector_layer] = []

        processed: bool = False

        for layer_index in range(layer_count):

            if process_layer != -1 and layer_index != process_layer:
                continue

            layer: ogr.Layer = datasource.GetLayerByIndex(layer_index)

            x_min, x_max, y_min, y_max = layer.GetExtent()
            layer_name: str = layer.GetName()
            extent: List[Number] = [x_min, y_max, x_max, y_min]
            extent_ogr: List[Number] = [x_min, x_max, y_min, y_max]
            extent_dict: Dict[str, Number] = {
                "left": x_min,
                "top": y_max,
                "right": x_max,
                "bottom": y_min,
            }

            column_fid: str = layer.GetFIDColumn()
            column_geom: str = layer.GetGeometryColumn()

            if column_geom == "":
                column_geom = "geom"

            feature_count: int = layer.GetFeatureCount()

            projection_osr = layer.GetSpatialRef()
            projection = layer.GetSpatialRef().ExportToWkt()

            if processed is False:
                ds_projection = projection
                ds_projection_osr = projection_osr
                ds_x_min: Number = x_min
                ds_x_max: Number = x_max
                ds_y_min: Number = y_min
                ds_y_max: Number = y_max

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

            layer_defn: ogr.FeatureDefn = layer.GetLayerDefn()

            geom_type_ogr: int = layer_defn.GetGeomType()
            geom_type: str = ogr.GeometryTypeToName(layer_defn.GetGeomType())

            field_count: int = layer_defn.GetFieldCount()
            field_names: List[str] = []
            field_types: List[str] = []
            field_types_ogr: List[int] = []

            for field_index in range(field_count):
                field_defn: ogr.FieldDefn = layer_defn.GetFieldDefn(field_index)
                field_names.append(field_defn.GetName())
                field_type = field_defn.GetType()
                field_types_ogr.append(field_type)
                field_types.append(field_defn.GetFieldTypeName(field_type))

            layer_dict: Metadata_vector_layer = {
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
                "extent_datasource": None,
                "extent_geom": None,
                "extent_latlng": None,
                "extent_gdal_warp_latlng": None,
                "extent_ogr_latlng": None,
                "extent_dict_latlng": None,
                "extent_wkt_latlng": None,
                "extent_datasource_latlng": None,
                "extent_geom_latlng": None,
                "extent_geojson": None,
                "extent_geojson_dict": None,
            }

            layers.append(layer_dict)

        ds_extent: List[Number] = [ds_x_min, ds_y_max, ds_x_max, ds_y_min]
        ds_extent_ogr: List[Number] = [ds_x_min, ds_x_max, ds_y_min, ds_y_max]
        ds_extent_gdal_warp: List[Number] = [ds_x_min, ds_y_min, ds_x_max, ds_y_max]
        ds_extent_dict: Dict[str, Number] = {
            "left": ds_x_min,
            "top": ds_y_max,
            "right": ds_x_max,
            "bottom": ds_y_min,
        }

        metadata: Metadata_vector = {
            "path": path,
            "basename": basename,
            "name": name,
            "ext": ext,
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
            "extent_ogr": ds_extent_ogr,
            "extent_gdal_warp": ds_extent_gdal_warp,
            "extent_dict": ds_extent_dict,
            "extent_wkt": None,
            "extent_datasource": None,
            "extent_geom": None,
            "extent_latlng": None,
            "extent_gdal_warp_latlng": None,
            "extent_ogr_latlng": None,
            "extent_dict_latlng": None,
            "extent_wkt_latlng": None,
            "extent_datasource_latlng": None,
            "extent_geom_latlng": None,
            "extent_geojson": None,
            "extent_geojson_dict": None,
        }

        if not simple:
            # Combined extents
            extended_extents = advanced_extents(ds_extent_ogr, ds_projection_osr)

            for key, value in extended_extents.items():
                metadata[key] = value  # type: ignore

            # Individual layer extents
            for layer_index in range(layer_count):
                layer_dict = layers[layer_index]
                extended_extents_layer = advanced_extents(
                    layer_dict["extent_ogr"], layer_dict["projection_osr"]
                )

                for key, value in extended_extents_layer.items():
                    metadata[key] = value  # type: ignore

        metadatas.append(metadata)

    if isinstance(vector, list):
        return metadatas

    return metadatas[0]


def vector_to_memory(
    vector: Union[List[Union[ogr.DataSource, str]], ogr.DataSource, str],
    memory_path: Union[List[str], str, None] = None,
    layer_to_extract: int = -1,
    opened: bool = False,
) -> Union[str, ogr.DataSource]:
    """ Copies a vector source to memory.

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
    ref = vector_to_reference(vector)
    metadata = vector_to_metadata(ref)

    basename = metadata["name"] if metadata["name"] is not None else "name"

    driver = None
    vector_name = None
    if memory_path is not None:
        if memory_path[0:8] == "/vsimem/":
            vector_name = memory_path
        else:
            vector_name = f"/vsimem/{memory_path}"
        driver = ogr.GetDriverByName(path_to_driver(memory_path))
    else:
        vector_name = f"/vsimem/memvector_{basename}_{uuid4().int}.gpkg"
        driver = ogr.GetDriverByName("GPKG")

    copy = driver.CreateDataSource(vector_name)

    for layer_idx in range(metadata["layer_count"]):
        if layer_to_extract is not None and layer_idx != layer_to_extract:
            continue
        layername = metadata["layers"][layer_idx]["layer_name"]
        copy.CopyLayer(ref.GetLayer(layer_idx), layername, ["OVERWRITE=YES"])

    if opened:
        return copy

    return vector_name


def vector_to_disk(
    vector: Union[str, ogr.DataSource],
    out_path: str,
    overwrite: bool = True,
    opened: bool = False,
) -> str:
    """ Copies a vector source to disk.

    Args:
        vector (path | DataSource): The vector to copy to disk

        out_path (path): The destination to save to.

    **kwargs:
        overwite (bool): Is it possible to overwrite the out_path if it exists.

    Returns:
        An path to the created vector.
    """
    if not is_vector(vector):
        raise TypeError("Input not a vector.")

    driver = ogr.GetDriverByName(path_to_driver(out_path))
    assert driver != None, "Unable to parse driver."

    metadata = vector_to_metadata(vector)

    remove_if_overwrite(out_path, overwrite)

    copy = driver.CreateDataSource(out_path)

    ref = vector_to_reference(vector)

    for layer_idx in range(metadata["layer_count"]):
        layer_name = metadata["layers"][layer_idx]["layer_name"]
        copy.CopyLayer(ref.GetLayer(layer_idx), str(layer_name), ["OVERWRITE=YES"])

    # Flush to disk
    copy = None

    if opened:
        return vector_to_reference(out_path)

    return out_path


def vector_add_index(
    vector: Union[List[Union[str, ogr.DataSource]], str, ogr.DataSource]
) -> None:
    """ Adds a spatial index to the vector if it doesn't have one.

    Args:
        vector (list, path | vector): The vector to add the index to.

    Returns:
        None
    """
    type_check(vector, [list, str, ogr.DataSource], "vector")

    in_vectors = to_vector_list(vector)

    metadatas = vector_to_metadata(in_vectors)

    if not isinstance(metadatas, list):
        raise Exception("Error while parsing metadata.")

    for index, in_vector in enumerate(in_vectors):
        metadata = metadatas[index]
        ref = vector_to_reference(in_vector)

        for layer in metadata["layers"]:
            name = layer["layer_name"]
            geom = layer["column_geom"]

            sql = f"SELECT CreateSpatialIndex('{name}', '{geom}') WHERE NOT EXISTS (SELECT HasSpatialIndex('{name}', '{geom}'));"
            ref.ExecuteSQL(sql, dialect="SQLITE")

    return None


# TODO: Update this to work on every geom layer.
# TODO: Add output array.
# TODO: Add copy
# TODO: Update to do circular hull
def vector_add_shapes(
    vector: Union[str, ogr.DataSource],
    shapes: list = ["area", "perimeter", "ipq", "hull", "compactness", "centroid"],
    output_array: bool = False,
    output_copy: bool = False,
) -> Union[str, np.ndarray]:
    """ Adds shape calculations to a vector such as area and perimeter.
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

        output_array (bool): If true a numpy array matching the FID's of the
        vector and the calculated shapes (same order as 'shapes').

        output_copy (bool): If true a copy is return with the added shapes. if
        false the input vector is updated in place.

    Returns:
        Either the path to the updated vector
    """

    internal_vector = ogr.Open(vector, 1)
    vector_layer = internal_vector.GetLayer(0)
    vector_layer_defn = vector_layer.GetLayerDefn()
    vector_field_counts = vector_layer_defn.GetFieldCount()
    vector_current_fields = []

    # Get current fields
    for i in range(vector_field_counts):
        vector_current_fields.append(vector_layer_defn.GetFieldDefn(i).GetName())

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

    return vector


def vector_in_memory(vector):
    metadata = vector_to_metadata(vector)

    if metadata["driver"] == "Memory":
        return True

    if "/vsimem/" in metadata["path"]:
        return True

    return False


def vector_to_path(vector):
    metadata = vector_to_metadata(vector)

    if metadata["driver"] == "Memory":
        out_format = ".gpkg"
        out_target = f"/vsimem/memvect_{uuid4().int}{out_format}"
        return vector_to_memory(vector, memory_path=out_target)

    if str(metadata["path"])[0:8] == "/vsimem/":
        return metadata["path"]

    if os.path.exists(metadata["abs_path"]):
        return metadata["abs_path"]

    raise Exception("Unable to find path from vector source.")


def vector_to_extent(vector):
    return vector_to_metadata(vector)["extent_ogr"]
