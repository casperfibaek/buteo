import sys
from typing import Union, Optional, List
from osgeo import ogr

sys.path.append("../../")

from buteo.gdal_utils import path_to_driver_vector
from buteo.utils import overwrite_required, progress, remove_if_overwrite, type_check
from buteo.vector.io import (
    internal_vector_to_metadata,
    ready_io_vector,
    vector_add_index,
    open_vector,
)


def internal_singlepart_to_multipart(
    vector: Union[str, ogr.DataSource],
    out_path: Optional[str] = None,
    overwrite: bool = True,
    add_index: bool = True,
    process_layer: int = -1,
) -> str:
    type_check(vector, [str, ogr.DataSource], "vector")
    type_check(out_path, [str], "out_path", allow_none=True)
    type_check(overwrite, [bool], "overwrite")
    type_check(add_index, [bool], "add_index")
    type_check(process_layer, [int], "process_layer")

    vector_list, path_list = ready_io_vector(vector, out_path, overwrite=overwrite)
    ref = open_vector(vector_list[0])
    out_name = path_list[0]

    out_format = path_to_driver_vector(out_name)
    driver = ogr.GetDriverByName(out_format)
    overwrite_required(out_name, overwrite)

    metadata = internal_vector_to_metadata(ref)

    remove_if_overwrite(out_name, overwrite)

    destination: ogr.DataSource = driver.CreateDataSource(out_name)

    for index, layer_meta in enumerate(metadata["layers"]):
        if process_layer != -1 and index != process_layer:
            continue

        name = layer_meta["layer_name"]
        geom = layer_meta["column_geom"]

        sql = f"SELECT ST_Collect({geom}) AS geom FROM {name};"

        result = ref.ExecuteSQL(sql, dialect="SQLITE")
        destination.CopyLayer(result, name, ["OVERWRITE=YES"])

    if add_index:
        vector_add_index(destination)

    destination.FlushCache()

    return out_name


def internal_multipart_to_singlepart(
    vector: Union[str, ogr.DataSource],
    out_path: Optional[str] = None,
    copy_attributes: bool = False,
    overwrite: bool = True,
    add_index: bool = True,
    process_layer: int = -1,
    verbose: int = 1,
) -> str:
    type_check(vector, [str, ogr.DataSource], "vector")
    type_check(out_path, [str], "out_path", allow_none=True)
    type_check(overwrite, [bool], "overwrite")
    type_check(add_index, [bool], "add_index")
    type_check(process_layer, [int], "process_layer")
    type_check(verbose, [int], "verbose")

    vector_list, path_list = ready_io_vector(vector, out_path, overwrite=overwrite)

    ref = open_vector(vector_list[0])
    out_name = path_list[0]

    driver = ogr.GetDriverByName(path_to_driver_vector(out_name))

    metadata = internal_vector_to_metadata(ref)

    remove_if_overwrite(out_name, overwrite)

    destination = driver.CreateDataSource(out_name)

    for index, layer_meta in enumerate(metadata["layers"]):
        if process_layer != -1 and index != process_layer:
            continue

        if verbose == 1:
            layer_name = layer_meta["layer_name"]
            print(f"Splitting layer: {layer_name}")

        target_unknown = False

        if layer_meta["geom_type_ogr"] == 4:  # MultiPoint
            target_type = 1  # Point
        elif layer_meta["geom_type_ogr"] == 5:  # MultiLineString
            target_type = 2  # LineString
        elif layer_meta["geom_type_ogr"] == 6:  # MultiPolygon
            target_type = 3  # Polygon
        elif layer_meta["geom_type_ogr"] == 1004:  # MultiPoint (z)
            target_type = 1001  # Point (z)
        elif layer_meta["geom_type_ogr"] == 1005:  # MultiLineString (z)
            target_type = 1002  # LineString (z)
        elif layer_meta["geom_type_ogr"] == 1006:  # MultiPolygon (z)
            target_type = 1003  # Polygon (z)
        elif layer_meta["geom_type_ogr"] == 2004:  # MultiPoint (m)
            target_type = 2001  # Point (m)
        elif layer_meta["geom_type_ogr"] == 2005:  # MultiLineString (m)
            target_type = 2002  # LineString (m)
        elif layer_meta["geom_type_ogr"] == 2006:  # MultiPolygon (m)
            target_type = 2003  # Polygon (m)
        elif layer_meta["geom_type_ogr"] == 3004:  # MultiPoint (zm)
            target_type = 3001  # Point (m)
        elif layer_meta["geom_type_ogr"] == 3005:  # MultiLineString (zm)
            target_type = 3002  # LineString (m)
        elif layer_meta["geom_type_ogr"] == 3006:  # MultiPolygon (zm)
            target_type = 3003  # Polygon (m)
        else:
            target_unknown = True
            target_type = layer_meta["geom_type_ogr"]

        destination_layer = destination.CreateLayer(
            layer_meta["layer_name"], layer_meta["projection_osr"], target_type
        )
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

                if ftype == "String" or ftype == "Date":
                    fielddefn = ogr.FieldDefn(fname, ogr.OFTString)
                    fielddefn.SetWidth(fwidth)
                elif ftype == "Real":
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
                progress(_, feature_count - 1, "Splitting.")

    if add_index:
        vector_add_index(destination)

    return out_name


def singlepart_to_multipart(
    vector: Union[List[Union[str, ogr.DataSource]], Union[str, ogr.DataSource]],
    out_path: str = None,
    overwrite: bool = True,
    add_index: bool = True,
    process_layer: int = -1,
) -> Union[List[str], str]:
    type_check(vector, [list, str, ogr.DataSource], "vector")
    type_check(out_path, [str], "out_path", allow_none=True)
    type_check(overwrite, [bool], "overwrite")
    type_check(add_index, [bool], "add_index")
    type_check(process_layer, [int], "process_layer")

    vector_list, path_list = ready_io_vector(vector, out_path, overwrite=overwrite)

    output = []
    for index, in_vector in enumerate(vector_list):
        output.append(
            internal_singlepart_to_multipart(
                in_vector,
                out_path=path_list[index],
                overwrite=overwrite,
                add_index=add_index,
                process_layer=process_layer,
            )
        )

    if isinstance(vector, list):
        return output[0]

    return output


def multipart_to_singlepart(
    vector: Union[List[Union[str, ogr.DataSource]], Union[str, ogr.DataSource]],
    out_path: Optional[Union[List[str], str]] = None,
    copy_attributes: bool = False,
    overwrite: bool = True,
    add_index: bool = True,
    process_layer: int = -1,
    verbose: int = 1,
) -> Union[List[str], str]:
    type_check(vector, [ogr.DataSource, str, list], "vector")
    type_check(out_path, [str], "out_path", allow_none=True)
    type_check(copy_attributes, [bool], "copy_attributes")
    type_check(overwrite, [bool], "overwrite")
    type_check(add_index, [bool], "add_index")
    type_check(process_layer, [int], "process_layer")
    type_check(verbose, [int], "verbose")

    vector_list, path_list = ready_io_vector(vector, out_path, overwrite=overwrite)

    output = []
    for index, in_vector in enumerate(vector_list):
        output.append(
            internal_multipart_to_singlepart(
                in_vector,
                out_path=path_list[index],
                copy_attributes=copy_attributes,
                overwrite=overwrite,
                add_index=add_index,
                process_layer=process_layer,
                verbose=verbose,
            )
        )

    if isinstance(vector, list):
        return output[0]

    return output
