"""
### Convert geometry composition. ###

Convert geometries from multiparts and singleparts and vice versa.
"""

# Standard library
import sys; sys.path.append("../../")

# External
from osgeo import ogr

# Internal
from buteo.utils import core_utils, gdal_utils
from buteo.vector import core_vector



def _singlepart_to_multipart(
    vector,
    out_path=None,
    *,
    overwrite=True,
    add_index=True,
    process_layer=-1,
):
    """ Internal. """
    assert isinstance(vector, (ogr.DataSource, str)), "Invalid input vector"
    assert gdal_utils.is_vector(vector), "Invalid input vector"

    if out_path is None:
        out_path = gdal_utils.create_memory_path(gdal_utils.get_path_from_dataset(vector), add_uuid=True)

    assert core_utils.is_valid_output_path(out_path, overwrite=overwrite), "Invalid output path"

    ref = core_vector._open_vector(vector)

    out_format = gdal_utils.get_path_from_dataset(out_path, dataset_type="vector")
    driver = ogr.GetDriverByName(out_format)

    core_utils.remove_if_required(out_path, overwrite)

    metadata = core_vector._vector_to_metadata(ref)
    destination = driver.CreateDataSource(out_path)

    for index, layer_meta in enumerate(metadata["layers"]):
        if process_layer != -1 and index != process_layer:
            continue

        name = layer_meta["layer_name"]
        geom = layer_meta["column_geom"]

        sql = f"SELECT ST_Collect({geom}) AS geom FROM {name};"

        result = ref.ExecuteSQL(sql, dialect="SQLITE")
        destination.CopyLayer(result, name, ["OVERWRITE=YES"])

    if add_index:
        core_vector.vector_add_index(destination)

    destination.FlushCache()

    return out_path


def _multipart_to_singlepart(
    vector,
    out_path=None,
    *,
    copy_attributes=False,
    overwrite=True,
    add_index=True,
    process_layer=-1,
    verbose=False,
):
    """ Internal. """
    assert isinstance(vector, (ogr.DataSource, str)), "Invalid input vector"
    assert gdal_utils.is_vector(vector), "Invalid input vector"

    if out_path is None:
        out_path = gdal_utils.create_memory_path(gdal_utils.get_path_from_dataset(vector), add_uuid=True)

    assert core_utils.is_valid_output_path(out_path, overwrite=overwrite), "Invalid output path"

    ref = core_vector._open_vector(vector)

    out_format = gdal_utils.get_path_from_dataset(out_path, dataset_type="vector")
    driver = ogr.GetDriverByName(out_format)

    core_utils.remove_if_required(out_path, overwrite)

    metadata = core_vector._vector_to_metadata(ref)

    core_utils.remove_if_required(out_path, overwrite)

    destination = driver.CreateDataSource(out_path)

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

            if verbose:
                core_utils.progress(_, feature_count - 1, "Splitting.")

    if add_index:
        core_vector.vector_add_index(destination)

    return out_path


def singlepart_to_multipart(
    vector,
    out_path=None,
    add_index=True,
    process_layer=-1,
    prefix="",
    suffix="",
    add_uuid=False,
    overwrite=True,
    allow_lists=True,
):
    """
    Converts a singlepart vector to multipart.

    ## Args:
    `vector` (_str_/_ogr.DataSource_/_list_): The vector(s) to convert. </br>

    ## Kvargs:
    `out_path` (_str_/_None_): The path(s) to the output vector. If None a memory output is produced. (Default: **None**) </br>
    `add_index` (_bool_): Add an geospatial index to the output vector. (Default: **True**) </br>
    `process_layer` (_int_): The layer index to process. (Default: **-1**) </br>
    `prefix` (_str_): The prefix to add to the layer name. (Default: **""**) </br>
    `suffix` (_str_): The suffix to add to the layer name. (Default: **""**) </br>
    `add_uuid` (_bool_): Add a UUID field to the output vector. (Default: **False**) </br>
    `overwrite` (_bool_): Overwrite the output vector if it already exists. (Default: **True**) </br>
    `allow_lists` (_bool_): Allow the input to be a list of vectors. (Default: **True**) </br>

    ## Returns:
    (_str_/_list_): The path(s) to the output vector.
    """
    core_utils.type_check(vector, [str, ogr.DataSource, [str, ogr.DataSource]], "vector")
    core_utils.type_check(out_path, [str, [str], None], "out_path")
    core_utils.type_check(overwrite, [bool], "overwrite")
    core_utils.type_check(add_index, [bool], "add_index")
    core_utils.type_check(process_layer, [int], "process_layer")
    core_utils.type_check(prefix, [str], "prefix")
    core_utils.type_check(suffix, [str], "suffix")
    core_utils.type_check(add_uuid, [bool], "add_uuid")
    core_utils.type_check(allow_lists, [bool], "allow_lists")

    if not allow_lists and isinstance(vector, list):
        raise ValueError("Vector cannot be a list when allow_lists is False.")

    vector_list = core_utils.ensure_list(vector)

    assert gdal_utils.is_vector_list(vector_list), f"Vector is not a list of vectors. {vector_list}"

    path_list = gdal_utils.create_output_path_list(
        vector_list,
        out_path=out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        overwrite=overwrite,
    )

    output = []
    for index, in_vector in enumerate(vector_list):
        output.append(
            _singlepart_to_multipart(
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
    vector,
    out_path=None,
    *,
    overwrite=True,
    add_index=True,
    process_layer=-1,
    verbose=False,
    prefix="",
    suffix="",
    add_uuid=False,
    allow_lists=True,
):
    """
    Converts a multipart vector to singlepart.

    ## Args:
    `vector` (_str_/_ogr.DataSource_/_list_): The vector(s) to convert. </br>

    ## Kvargs:
    `out_path` (_str_/_None_): The path(s) to the output vector. If None a memory output is produced. (Default: **None**) </br>
    `overwrite` (_bool_): Overwrite the output vector if it already exists. (Default: **True**) </br>
    `add_index` (_bool_): Add an geospatial index to the output vector. (Default: **True**) </br>
    `process_layer` (_int_): The layer index to process. (Default: **-1**) </br>
    `verbose` (_bool_): Print progress. (Default: **False**) </br>
    `prefix` (_str_): The prefix to add to the layer name. (Default: **""**) </br>
    `suffix` (_str_): The suffix to add to the layer name. (Default: **""**) </br>
    `add_uuid` (_bool_): Add a UUID field to the output vector. (Default: **False**) </br>
    `allow_lists` (_bool_): Allow the input to be a list of vectors. (Default: **True**) </br>

    ## Returns:
    (_str_/_list_): The path(s) to the output vector.
    """
    core_utils.type_check(vector, [str, ogr.DataSource, [str, ogr.DataSource]], "vector")
    core_utils.type_check(out_path, [str, [str], None], "out_path")
    core_utils.type_check(overwrite, [bool], "overwrite")
    core_utils.type_check(add_index, [bool], "add_index")
    core_utils.type_check(process_layer, [int], "process_layer")
    core_utils.type_check(prefix, [str], "prefix")
    core_utils.type_check(suffix, [str], "suffix")
    core_utils.type_check(add_uuid, [bool], "add_uuid")
    core_utils.type_check(allow_lists, [bool], "allow_lists")

    if not allow_lists and isinstance(vector, list):
        raise ValueError("Vector cannot be a list when allow_lists is False.")

    vector_list = core_utils.ensure_list(vector)

    assert gdal_utils.is_vector_list(vector_list), f"Vector is not a list of vectors. {vector_list}"

    path_list = gdal_utils.create_output_path_list(
        vector_list,
        out_path=out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        overwrite=overwrite,
    )

    output = []
    for index, in_vector in enumerate(vector_list):
        output.append(
            _multipart_to_singlepart(
                in_vector,
                out_path=path_list[index],
                overwrite=overwrite,
                add_index=add_index,
                process_layer=process_layer,
                verbose=verbose,
            )
        )

    if isinstance(vector, list):
        return output[0]

    return output
