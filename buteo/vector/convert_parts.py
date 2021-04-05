import sys; sys.path.append('../../')
from uuid import uuid4
from typing import Union
from osgeo import ogr

from buteo.gdal_utils import (
    is_vector,
    path_to_driver,
    vector_to_reference,
)
from buteo.utils import overwrite_required, progress, remove_if_overwrite, type_check
from buteo.vector.io import vector_to_metadata, vector_add_index


def singlepart_to_multipart(
    vector: Union[ogr.DataSource, str, list],
    out_path: str=None,
    overwrite: bool=True,
    opened: bool=False,
    vector_idx: int=-1,
) -> str:
    type_check(vector, [ogr.DataSource, str, list], "vector")
    type_check(out_path, [str], "out_path", allow_none=True)
    type_check(overwrite, [bool], "overwrite")
    type_check(opened, [bool], "opened")
    type_check(vector_idx, [int], "vector_idx")

    out_format = 'GPKG'
    out_target = f"/vsimem/multi_to_single_{uuid4().int}.gpkg"

    if out_path is not None:
        out_target = out_path
        out_format = path_to_driver(out_path)

    if not is_vector(vector):
        raise TypeError(f"Invalid vector input: {vector}.")

    overwrite_required(out_target, overwrite)

    driver = ogr.GetDriverByName(out_format)

    ref = vector_to_reference(vector)
    metadata = vector_to_metadata(ref, latlng_and_footprint=False)

    remove_if_overwrite(out_target, overwrite)

    destination = driver.CreateDataSource(out_target)

    for index, layer_meta in enumerate(metadata["layers"]):
        if vector_idx != -1 and index != vector_idx:
            continue

        name = layer_meta["layer_name"]
        geom = layer_meta["geom_column"]

        sql = f"SELECT ST_Collect({geom}) AS geom FROM {name};"

        result = ref.ExecuteSQL(sql, dialect="SQLITE")
        destination.CopyLayer(result, name, ["OVERWRITE=YES"])

    vector_add_index(destination)

    if opened:
        return destination
    
    return out_path


def multipart_to_singlepart(
    vector: Union[ogr.DataSource, str, list],
    copy_attributes: bool=False,
    out_path: str=None,
    overwrite: bool=True,
    opened: bool=False,
    vector_idx: int=-1,
    verbose: int=1,
) -> str:
    """ Clips a vector to a geometry.
    Args:
        vector (list of vectors | path | vector): The vectors(s) to clip.

        clip_geom (list of geom | path | vector | rasters): The geometry to use
        for the clipping

    **kwargs:


    Returns:
        A clipped ogr.Datasource or the path to one.
    """
    type_check(vector, [ogr.DataSource, str, list], "vector")
    type_check(out_path, [str], "out_path", allow_none=True)
    type_check(overwrite, [bool], "overwrite")
    type_check(opened, [bool], "opened")
    type_check(vector_idx, [int], "vector_idx")

    out_format = 'GPKG'
    out_target = f"/vsimem/multi_to_single_{uuid4().int}.gpkg"

    if out_path is not None:
        out_target = out_path
        out_format = path_to_driver(out_path)

    if not is_vector(vector):
        raise TypeError(f"Invalid vector input: {vector}.")

    overwrite_required(out_target, overwrite)

    driver = ogr.GetDriverByName(out_format)

    ref = vector_to_reference(vector)
    metadata = vector_to_metadata(ref, latlng_and_footprint=False)

    remove_if_overwrite(out_target, overwrite)
    destination = driver.CreateDataSource(out_target)

    for index, layer_meta in enumerate(metadata["layers"]):
        if vector_idx != -1 and index != vector_idx:
            continue

        if verbose == 1:
            layer_name = layer_meta["layer_name"]
            print(f"Splitting layer: {layer_name}")

        target_unknown = False

        if layer_meta["geom_type_ogr"] == 4: # MultiPoint
            target_type = 1 # Point
        elif layer_meta["geom_type_ogr"] == 5: # MultiLineString
            target_type = 2 # LineString
        elif layer_meta["geom_type_ogr"] == 6: # MultiPolygon
            target_type = 3 # Polygon
        elif layer_meta["geom_type_ogr"] == 1004: # MultiPoint (z)
            target_type = 1001 # Point (z)
        elif layer_meta["geom_type_ogr"] == 1005: # MultiLineString (z)
            target_type = 1002 # LineString (z)
        elif layer_meta["geom_type_ogr"] == 1006: # MultiPolygon (z)
            target_type = 1003 # Polygon (z)
        elif layer_meta["geom_type_ogr"] == 2004: # MultiPoint (m)
            target_type = 2001 # Point (m)
        elif layer_meta["geom_type_ogr"] == 2005: # MultiLineString (m)
            target_type = 2002 # LineString (m)
        elif layer_meta["geom_type_ogr"] == 2006: # MultiPolygon (m)
            target_type = 2003 # Polygon (m)
        elif layer_meta["geom_type_ogr"] == 3004: # MultiPoint (zm)
            target_type = 3001 # Point (m)
        elif layer_meta["geom_type_ogr"] == 3005: # MultiLineString (zm)
            target_type = 3002 # LineString (m)
        elif layer_meta["geom_type_ogr"] == 3006: # MultiPolygon (zm)
            target_type = 3003 # Polygon (m)
        else:
            target_unknown = True
            target_type = layer_meta["geom_type_ogr"]

        destination_layer = destination.CreateLayer(layer_meta["layer_name"], layer_meta["projection_osr"], target_type)
        layer_defn = destination_layer.GetLayerDefn()
        field_count = layer_meta["field_count"]

        original_target = ref.GetLayerByIndex(index)
        feature_count = original_target.GetFeatureCount()

        if copy_attributes:
            first_feature = original_target.GetNextFeature()
            original_target.ResetReading()

            if verbose == 1:
                print("Creating attribute fields")

            for field_id in range(field_count):
                field_defn = first_feature.GetFieldDefnRef(field_id)

                fname = field_defn.GetName()
                ftype = field_defn.GetTypeName()
                fwidth = field_defn.GetWidth()
                fprecision = field_defn.GetPrecision()

                if ftype == 'String' or ftype == "Date":
                    fielddefn = ogr.FieldDefn(fname, ogr.OFTString)
                    fielddefn.SetWidth(fwidth)
                elif ftype == 'Real':
                    fielddefn = ogr.FieldDefn(fname, ogr.OFTReal)
                    fielddefn.SetWidth(fwidth)
                    fielddefn.SetPrecision(fprecision)
                else:
                    fielddefn = ogr.FieldDefn(fname, ogr.OFTInteger) 

                destination_layer.CreateField(fielddefn)

        for _ in range(feature_count):
            feature = original_target.GetNextFeature()
            geom = feature.GetGeometryRef()

            if target_unknown:
                out_feat = ogr.Feature(layer_defn) 
                out_feat.SetGeometry(geom)

                if copy_attributes:
                    for field_id in range(field_count):
                        values = feature.GetField(field_id)
                        out_feat.SetField(field_id, values)

                destination_layer.CreateFeature(out_feat)           

            for geom_part in geom: 
                out_feat = ogr.Feature(layer_defn) 
                out_feat.SetGeometry(geom_part)

                if copy_attributes:
                    for field_id in range(field_count):
                        values = feature.GetField(field_id)
                        out_feat.SetField(field_id, values)

                destination_layer.CreateFeature(out_feat)
            
            if verbose == 1:
                progress(_, feature_count-1, "Splitting.")

    vector_add_index(destination)

    if opened:
        return destination
    
    return out_target


if __name__ == "__main__":
    folder = "C:/Users/caspe/Desktop/test/"

    out_dir = folder + "out/"
    vector = out_dir + "walls_dissolved_single.gpkg"

    singlepart_to_multipart(vector, out_path=out_dir + "walls_dissolved_single_remerged.gpkg")
