import sys; sys.path.append('../../')
import os
import numpy as np
from typing import Union
import osgeo; from osgeo import gdal, ogr, osr

from buteo.gdal_utils import (
    parse_projection,
    vector_to_reference,
    path_to_driver,
    geoms_from_extent,
)
from buteo.utils import progress, remove_if_overwrite


# TODO:
#   - repair vector
#   - sanity checks: vectors_intersect, is_not_empty, does_vectors_match, match_vectors
#   - rasterize - with antialiasing/weights
#   - join by attribute + summary
#   - join by location + summary
#   - intersection, buffer, union, clip, erase
#   - multithreaded processing


def vector_to_memory(
    vector: Union[str, ogr.DataSource],
    memory_path: Union[str, None]=None,
    layer_to_extract: Union[int, None]=None,
) -> ogr.DataSource:
    """ Copies a vector source to memory.

    Args:
        vector (path | DataSource): The vector to copy to memory

    **kwargs:
        memory_path (str | None): If a path is provided, uses the
        appropriate driver and uses the VSIMEM gdal system.
        Example: vector_to_memory(clip_ref, "clip_geom.gpkg")
        /vsimem/ is autumatically added.

    Returns:
        An in-memory ogr.DataSource. If a memory path was provided a
        string for the in-memory location is returned.
    """
    ref = vector_to_reference(vector)
    metadata = vector_to_metadata(ref)
    basename = metadata["basename"] if metadata["basename"] is not None else "mem_vector"

    driver = None
    vector_name = None
    if memory_path is not None:
        vector_name = f"/vsimem/{memory_path}"
        driver = ogr.GetDriverByName(path_to_driver(memory_path))
    else:
        vector_name = basename
        driver = ogr.GetDriverByName("Memory")

    copy = driver.CreateDataSource(vector_name)

    for layer_idx in range(metadata["layer_count"]):
        if layer_to_extract is not None and layer_idx != layer_to_extract:
            continue
        layername = metadata["layers"][layer_idx]["layer_name"]
        copy.CopyLayer(ref.GetLayer(layer_idx), layername, ["OVERWRITE=YES"])

    if memory_path is None:
        return copy

    return vector_name


def vector_to_disk(
    vector: Union[str, ogr.DataSource],
    out_path: str,
    overwrite: bool=True,
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
    assert isinstance(vector, ogr.DataSource), "Input not a vector datasource."

    driver = ogr.GetDriverByName(path_to_driver(out_path))
    assert driver != None, "Unable to parse driver."

    metadata = vector_to_metadata(vector)

    remove_if_overwrite(out_path, overwrite)

    copy = driver.CreateDataSource(out_path)

    for layer_idx in range(metadata["layer_count"]):
        layer_name = metadata["layers"][layer_idx]["layer_name"]
        copy.CopyLayer(vector.GetLayer(layer_idx), str(layer_name), ["OVERWRITE=YES"])

    copy = None
    return out_path


def vector_to_metadata(
    vector: Union[str, ogr.DataSource],
    latlng_and_footprint: bool=True,
    process_layer: Union[str, int]="all",
) -> dict:
    """ Creates a dictionary with metadata about the vector layer.

    Args:
        vector (path | DataSource): The vector to analyse.

    **kwargs:
        latlng_and_footprint (bool): Should the metadata include a
            footprint of the raster in wgs84. Requires a reprojection
            check do not use it if not required and performance is important.
        
        process_layer (str, int): The layer to process. Default is "all". 
        Must be either an int or "all".

    Returns:
        A dictionary containing the metadata.
    """
    try:
        vector = vector if isinstance(vector, ogr.DataSource) else ogr.Open(vector)
    except:
        raise Exception("Could not read input vector")
    
    if vector is None:
        raise Exception("Could not read input vector")

    abs_path = os.path.abspath(vector.GetName())

    metadata = {
        "path": abs_path,
        "basename": os.path.basename(abs_path),
        "filetype": os.path.splitext(os.path.basename(abs_path))[1],
        "name": os.path.splitext(os.path.basename(abs_path))[0],
        "layer_count": vector.GetLayerCount(),
        "layers": [],
        "extent": None,
        "extent_dict": None,
        "x_min": None,
        "x_max": None,
        "y_min": None,
        "y_max": None,
        "extent_wgs84": None,
        "extent_dict_wgs84": None,
        "extent_wkt": None,
        "extent_ogr": None,
        "extent_ogr_geom": None,
        "extent_geojson": None,
        "extent_geojson_dict": None,
    }

    if isinstance(process_layer, int) and process_layer > (metadata["layer_count"] - 1):
        raise ValueError("Requested a non-present layer.")
    
    processed = False

    for layer_index in range(metadata["layer_count"]):

        if isinstance(process_layer, int) and layer_index != process_layer:
            continue
        
        layer = vector.GetLayerByIndex(layer_index)

        x_min, x_max, y_min, y_max = layer.GetExtent()

        layer_dict = {
            "layer_name": layer.GetName(),
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "extent": [x_min, y_max, x_max, y_min],
            "extent_dict": {
                "left": x_min,
                "top": y_max,
                "right": x_max,
                "bottom": y_min,
            },
            "fid_column": layer.GetFIDColumn(),
            "feature_count": layer.GetFeatureCount(),
            "field_count": None,
            "geom_type": None,
            "geom_type_ogr": None,
            "field_names": None,
            "field_types": None,
            "extent_wgs84": None,
            "extent_dict_wgs84": None,
            "extent_wkt": None,
            "extent_ogr": None,
            "extent_ogr_geom": None,
            "extent_geojson": None,
            "extent_geojson_dict": None,
            "projection": None,
            "projection_osr": None,
        }

        projection_wkt = layer.GetSpatialRef().ExportToWkt()
        original_projection = osr.SpatialReference()
        original_projection.ImportFromWkt(projection_wkt)

        if processed is False:
            metadata["projection"] = projection_wkt
            metadata["projection_osr"] = original_projection
            metadata["x_min"] = x_min
            metadata["x_max"] = x_max
            metadata["y_min"] = y_min
            metadata["y_max"] = y_max

            processed = True
        else:
            if x_min < metadata["x_min"]: metadata["x_min"] = x_min
            if x_max > metadata["x_max"]: metadata["x_max"] = x_max
            if y_min < metadata["y_min"]: metadata["y_min"] = y_min
            if y_max > metadata["y_max"]: metadata["y_max"] = y_max

        layer_dict["projection"] = projection_wkt
        layer_dict["projection_osr"] = original_projection

        layer_defn = layer.GetLayerDefn()

        layer_dict["field_count"] = layer_defn.GetFieldCount()
        layer_dict["geom_type"] = ogr.GeometryTypeToName(layer_defn.GetGeomType())
        layer_dict["geom_type_ogr"] = layer_defn.GetGeomType()

        layer_dict["field_names"] = []
        layer_dict["field_types"] = []

        for field_index in range(layer_dict["field_count"]):
            field_defn = layer_defn.GetFieldDefn(field_index)
            layer_dict["field_names"].append(field_defn.GetName())
            layer_dict["field_types"].append(field_defn.GetFieldTypeName(field_defn.GetType()))

        metadata["layers"].append(layer_dict)
    
    metadata["extent"] = [
        metadata["x_min"],
        metadata["y_max"],
        metadata["x_max"],
        metadata["y_min"],
    ]

    metadata["extent_dict"] = {
        "left": metadata["x_min"],
        "top": metadata["y_max"],
        "right": metadata["x_max"],
        "bottom":metadata["y_min"],
    }

    if latlng_and_footprint:
        wgs84 = osr.SpatialReference()
        wgs84.ImportFromEPSG(4326) # WGS84, latlng

        for layer_index in range(metadata["layer_count"]):
            layer_dict = metadata["layers"][layer_index]
            
            extents = geoms_from_extent(layer_dict["extent"], original_projection, wgs84, layer_dict["layer_name"])

            layer_dict["extent_wgs84"] = extents["ta_proj"]
            layer_dict["extent_wgs84_dict"] = extents["ta_proj_dict"]
            layer_dict["extent_wkt"] = extents["wkt"]
            layer_dict["extent_ogr"] = extents["ogr"]
            layer_dict["extent_ogr_geom"] = extents["ogr_geom"]
            layer_dict["extent_geojson"] = extents["geojson"]
            layer_dict["extent_geojson_dict"] = extents["geojson_dict"]

        if metadata["layer_count"] == 1:
            metadata["extent_wgs84"] = layer_dict["extent_wgs84"]
            metadata["extent_wgs84_dict"] = layer_dict["extent_wgs84_dict"]
            metadata["extent_wkt"] = layer_dict["extent_wkt"]
            metadata["extent_ogr"] = layer_dict["extent_ogr"]
            metadata["extent_ogr_geom"] = layer_dict["extent_ogr_geom"]
            metadata["extent_geojson"] = layer_dict["extent_geojson"]
            metadata["extent_geojson_dict"] = layer_dict["extent_geojson_dict"]
        else:
            extents = geoms_from_extent(metadata["extent"], original_projection, wgs84, metadata["name"])

            metadata["extent_wgs84"] = extents["ta_proj"]
            metadata["extent_wgs84_dict"] = extents["ta_proj_dict"]
            metadata["extent_wkt"] = extents["wkt"]
            metadata["extent_ogr"] = extents["ogr"]
            metadata["extent_ogr_geom"] = extents["ogr_geom"]
            metadata["extent_geojson"] = extents["geojson"]
            metadata["extent_geojson_dict"] = extents["geojson_dict"]

    vector = None

    return metadata


def reproject_vector(
    vector: Union[str, ogr.DataSource],
    projection: Union[str, ogr.DataSource, gdal.Dataset, osr.SpatialReference],
    out_path: Union[str, None]=None,
    overwrite: bool=True,
) -> Union[str, ogr.DataSource]:
    """ Reprojects a vector given a target projection.

    Args:
        vector (path | vector): The vector to reproject.
        
        projection (str | int | vector | raster): The projection is infered from
        the input. The input can be: WKT proj, EPSG proj, Proj, or read from a 
        vector or raster datasource either from path or in-memory.

    **kwargs:
        out_path (path | None): The destination to save to. If None then
        the output is an in-memory raster.

        overwite (bool): Is it possible to overwrite the out_path if it exists.

    Returns:
        An in-memory vector. If an out_path is given, the output is a string containing
        the path to the newly created vecotr.
    """
    origin = vector_to_reference(vector)
    metadata = vector_to_metadata(origin)

    origin_projection = metadata["projection_osr"]
    target_projection = parse_projection(projection)

    if origin_projection.IsSame(target_projection):
        if out_path is None:
            return vector_to_memory(vector)
        
        return vector_to_disk(vector, out_path)

    remove_if_overwrite(out_path, overwrite)

    # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
    if int(osgeo.__version__[0]) >= 3:
        origin_projection.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        target_projection.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    coord_trans = osr.CoordinateTransformation(origin_projection, target_projection)

    driver = None
    destination = None
    if out_path is not None:
        driver = ogr.GetDriverByName(path_to_driver(out_path))
        destination = driver.CreateDataSource(out_path)
    else:
        driver = ogr.GetDriverByName('Memory')
        destination = driver.CreateDataSource(metadata["name"])

    for layer_idx in range(len(metadata["layers"])):
        origin_layer = origin.GetLayerByIndex(layer_idx)
        origin_layer_defn = origin_layer.GetLayerDefn()

        layer_dict = metadata["layers"][layer_idx]
        layer_name = layer_dict["layer_name"]
        layer_geom_type = layer_dict["geom_type_ogr"]

        destination_layer = destination.CreateLayer(layer_name, target_projection, layer_geom_type)
        destination_layer_defn = destination_layer.GetLayerDefn()

        # Copy field definitions
        origin_layer_defn = origin_layer.GetLayerDefn()
        for i in range(0, origin_layer_defn.GetFieldCount()):
            field_defn = origin_layer_defn.GetFieldDefn(i)
            destination_layer.CreateField(field_defn)

        # Loop through the input features
        for _ in range(origin_layer.GetFeatureCount()):
            feature = origin_layer.GetNextFeature()
            geom = feature.GetGeometryRef()
            geom.Transform(coord_trans)

            new_feature = ogr.Feature(destination_layer_defn)
            new_feature.SetGeometry(geom)

            # Copy field values
            for i in range(0, destination_layer_defn.GetFieldCount()):
                new_feature.SetField(destination_layer_defn.GetFieldDefn(i).GetNameRef(), feature.GetField(i))

            destination_layer.CreateFeature(new_feature)
            
        destination_layer.ResetReading()
        destination_layer = None

    if out_path is not None:
        destination = None
        return out_path
    else:
        return destination


# TODO: Update this to work on every geom layer.
# TODO: Add output array.
# TODO: Add copy
# TODO: Update to do circular hull
def vector_add_shapes(
    vector: Union[str, ogr.DataSource],
    shapes: list=["area", "perimeter", "ipq", "hull", "compactness", "centroid"],
    output_array: bool=False,
    output_copy: bool=False,
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
            Warning('Invalid geometry at : ', i)

        if vector_geom is None:
            raise Exception('Invalid geometry. Could not fix.')

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
            vector_feature.SetField('area', vector_area)
        if "perimeter" in shapes:
            vector_feature.SetField('perimeter', vector_perimeter)
        if "ipq" in shapes:
            vector_feature.SetField('ipq', vector_ipq)
        if "hull" in shapes:
            vector_feature.SetField('hull_area', hull_area)
            vector_feature.SetField('hull_peri', hull_peri)
            vector_feature.SetField('hull_ratio', hull_ratio)
        if "compact" in shapes:
            vector_feature.SetField('compact', compactness)

        vector_layer.SetFeature(vector_feature)
        
        progress(i, vector_feature_count, name='shape')
        
    vector_layer.CommitTransaction()

    return vector
