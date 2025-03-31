"""### Dissolve vector geometries. ###

Dissolve vectors by attributes or geometry.
"""

# Standard library
import os
import tempfile
from typing import Union, Optional, List, cast

# External
from osgeo import ogr, gdal

# Internal
from buteo.utils import (
    utils_io,
    utils_base,
    utils_gdal,
    utils_path,
)
from buteo.core_vector.core_vector_read import open_vector as vector_open
from buteo.core_vector.core_vector_info import get_metadata_vector
from buteo.core_vector.core_vector_index import vector_add_index


def _vector_dissolve(
    vector: Union[str, ogr.DataSource],
    attribute: Optional[str] = None,
    out_path: Optional[str] = None,
    overwrite: bool = True,
    add_index: bool = True,
    process_layer: int = -1,
) -> str:
    """Internal dissolve implementation."""
    assert utils_gdal._check_is_vector(vector), "Invalid input vector"

    if out_path is None:
        out_path = utils_path._get_temp_filepath(vector, suffix="_dissolve", ext="gpkg")

    assert utils_path._check_is_valid_output_filepath(out_path, overwrite=overwrite), "Invalid output path"

    out_format = utils_gdal._get_vector_driver_name_from_path(out_path)

    driver = ogr.GetDriverByName(out_format)

    # Get datasource
    ds_result = vector_open(vector, writeable=False)
    
    # Ensure we have a single DataSource object
    if isinstance(ds_result, list):
        if not ds_result:
            raise ValueError("No valid vector datasource found")
        ref = ds_result[0]
    else:
        ref = ds_result
    
    # Now ref is guaranteed to be a single ogr.DataSource
    metadata = get_metadata_vector(ref)

    layers = []

    if process_layer == -1:
        for index in range(len(metadata["layers"])):
            layers.append(
                {
                    "name": metadata["layers"][index]["layer_name"],
                    "geom": metadata["layers"][index]["column_geom"],
                    "fields": metadata["layers"][index]["field_names"],
                }
            )
    else:
        if process_layer >= len(metadata["layers"]):
            raise ValueError(f"Invalid process_layer index: {process_layer}. Only {len(metadata['layers'])} layers available.")
        
        layers.append(
            {
                "name": metadata["layers"][process_layer]["layer_name"],
                "geom": metadata["layers"][process_layer]["column_geom"],
                "fields": metadata["layers"][process_layer]["field_names"],
            }
        )

    utils_io._delete_if_required(out_path, overwrite=overwrite)

    destination = driver.CreateDataSource(out_path)
    if destination is None:
        raise RuntimeError(f"Could not create output datasource: {out_path}")

    # Process each layer
    for layer in layers:
        if attribute is not None and attribute not in layer["fields"]:
            layer_fields = layer["fields"]
            raise ValueError(
                f"Invalid attribute for layer. Layer has the following fields: {layer_fields}"
            )

        geom_col = layer["geom"]
        name = layer["name"]

        try:
            sql = None
            if attribute is None:
                sql = f"SELECT ST_Union({geom_col}) AS geom FROM {name};"
            else:
                sql = f"SELECT {attribute}, ST_Union({geom_col}) AS geom FROM {name} GROUP BY {attribute};"

            result = ref.ExecuteSQL(sql, dialect="SQLITE")
        except Exception as e:
            # Try alternative column name if the first attempt fails
            try:
                if attribute is None:
                    sql = f"SELECT ST_Union(geometry) AS geom FROM {name};"
                else:
                    sql = f"SELECT {attribute}, ST_Union(geometry) AS geom FROM {name} GROUP BY {attribute};"

                result = ref.ExecuteSQL(sql, dialect="SQLITE")
            except Exception as inner_e:
                raise RuntimeError(f"Error executing SQL dissolve: {str(e)}, then: {str(inner_e)}")

        # Copy layer to destination
        if result:
            destination.CopyLayer(result, name, ["OVERWRITE=YES"])
            ref.ReleaseResultSet(result)
        else:
            raise RuntimeError("Failed to create result layer from SQL query")

    if add_index:
        vector_add_index(destination)

    destination.FlushCache()

    return out_path


def vector_dissolve(
    vector: Union[str, ogr.DataSource, gdal.Dataset, List[Union[str, ogr.DataSource, gdal.Dataset]]],
    attribute: Optional[str] = None,
    out_path: Optional[Union[str, List[str]]] = None,
    add_index: bool = True,
    process_layer: int = -1,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    overwrite: bool = True,
    allow_lists: bool = True,
) -> Union[str, List[str]]:
    """Dissolve vector geometries, optionally grouping by attribute.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource, gdal.Dataset, List[Union[str, ogr.DataSource, gdal.Dataset]]]
        The vector(s) to dissolve.

    attribute : Optional[str], optional
        The attribute to use for the dissolve, default: None

    out_path : Optional[Union[str, List[str]]], optional
        The output path, default: None

    add_index : bool, optional
        Add a spatial index to the output, default: True

    process_layer : int, optional
        The layer to process, default: -1 (all layers)

    prefix : str, optional
        The prefix to add to the output path, default: ""

    suffix : str, optional
        The suffix to add to the output path, default: ""

    add_uuid : bool, optional
        Add a uuid to the output path, default: False

    overwrite : bool, optional
        Overwrite the output, default: True

    allow_lists : bool, optional
        Allow lists as input, default: True

    Returns
    -------
    Union[str, List[str]]
        The output path(s) for the dissolved vector(s)
    """
    utils_base._type_check(vector, [ogr.DataSource, gdal.Dataset, str, [str, ogr.DataSource, gdal.Dataset]], "vector")
    utils_base._type_check(attribute, [str, None], "attribute")
    utils_base._type_check(out_path, [str, [str], None], "out_path")
    utils_base._type_check(add_index, [bool], "add_index")
    utils_base._type_check(process_layer, [int], "process_layer")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(overwrite, [bool], "overwrite")
    utils_base._type_check(allow_lists, [bool], "allow_lists")

    if not allow_lists and isinstance(vector, list):
        raise ValueError("Lists are not allowed when allow_lists is False.")

    input_is_list = isinstance(vector, list)
    vector_list = utils_base._get_variable_as_list(vector)

    assert utils_gdal._check_is_vector_list(vector_list), f"Invalid input vector: {vector_list}"

    # Handle output paths
    if out_path is None:
        # Create temp paths for each input
        path_list = []
        for path in vector_list:
            # When using prefix, save to disk instead of vsimem
            if prefix or suffix or add_uuid:
                # Use first path as base dir to create actual files
                base_dir = os.path.dirname(utils_gdal._get_path_from_dataset(vector_list[0]))
                if not base_dir or base_dir == "/vsimem":
                    # Fall back to temp dir if no valid directory
                    base_dir = tempfile.gettempdir()
                
                basename = os.path.basename(utils_gdal._get_path_from_dataset(path))
                name, ext = os.path.splitext(basename)
                if not ext:
                    ext = ".gpkg"
                
                # Create path with prefix/suffix
                new_name = f"{prefix}{name}{suffix}{ext}"
                temp_path = os.path.join(base_dir, new_name)
            else:
                # Use vsimem
                temp_path = utils_path._get_temp_filepath(
                    path,
                    prefix=prefix,
                    suffix=suffix,
                    add_uuid=add_uuid,
                )
            path_list.append(temp_path)
    elif isinstance(out_path, list):
        # Use provided output paths directly
        if len(out_path) != len(vector_list):
            raise ValueError("Number of output paths must match number of input vectors")
        path_list = out_path
    else:
        # Single output path for a single input
        if len(vector_list) > 1:
            raise ValueError("Single output path provided for multiple inputs")
        path_list = [out_path]

    assert utils_path._check_is_valid_output_path_list(path_list, overwrite=overwrite), f"Invalid output path generated. {path_list}"

    output = []
    for index, in_vector in enumerate(vector_list):
        output.append(
            _vector_dissolve(
                in_vector,
                attribute=attribute,
                out_path=path_list[index],
                overwrite=overwrite,
                add_index=add_index,
                process_layer=process_layer,
            )
        )

    if input_is_list:
        return output

    return output[0]
