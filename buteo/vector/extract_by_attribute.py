"""### Extract vector features by attribute values. ###

Filter vector features based on attribute values.
"""

# Standard library
from typing import Union, Optional, List, Any

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


def _vector_extract_by_attribute(
    vector: Union[str, ogr.DataSource],
    attribute: str,
    values: List[Any],
    out_path: Optional[str] = None,
    process_layer: int = -1,
    overwrite: bool = True,
) -> str:
    """Internal implementation to extract vector features by attribute values."""
    assert utils_gdal._check_is_vector(vector), "Invalid input vector"
    assert isinstance(attribute, str) and attribute.strip(), "Attribute must be a non-empty string"
    assert isinstance(values, list) and values, "Values must be a non-empty list"

    if out_path is None:
        out_path = utils_path._get_temp_filepath(vector, suffix="_extracted", ext="gpkg")

    assert utils_path._check_is_valid_output_filepath(out_path, overwrite=overwrite), "Invalid output path"

    out_format = utils_gdal._get_vector_driver_name_from_path(out_path)
    driver = ogr.GetDriverByName(out_format)

    # Open source vector
    ds_result = vector_open(vector, writeable=False)
    
    # Ensure we have a single DataSource object
    if isinstance(ds_result, list):
        if not ds_result:
            raise ValueError("No valid vector datasource found")
        source_ds = ds_result[0]
    else:
        source_ds = ds_result
    
    # Get metadata and validate
    metadata = get_metadata_vector(source_ds)
    
    layers_to_process = []
    
    if process_layer == -1:
        # Process all layers
        for idx, layer_meta in enumerate(metadata["layers"]):
            if attribute in layer_meta["field_names"]:
                layers_to_process.append((idx, layer_meta["layer_name"]))
    else:
        # Process just the specified layer
        if process_layer >= len(metadata["layers"]):
            raise ValueError(f"Invalid layer index: {process_layer}. Only {len(metadata['layers'])} layers available.")
        
        layer_meta = metadata["layers"][process_layer]
        if attribute not in layer_meta["field_names"]:
            raise ValueError(f"Attribute '{attribute}' not found in layer '{layer_meta['layer_name']}'")
        
        layers_to_process.append((process_layer, layer_meta["layer_name"]))
    
    if not layers_to_process:
        raise ValueError(f"Attribute '{attribute}' not found in any layer")

    # Delete existing file if needed
    utils_io._delete_if_required(out_path, overwrite=overwrite)
    
    # Create destination
    dest_ds = driver.CreateDataSource(out_path)
    if dest_ds is None:
        raise RuntimeError(f"Could not create output datasource: {out_path}")

    # Use SQL to extract features
    for layer_idx, layer_name in layers_to_process:
        # Format values for SQL
        formatted_values = []
        for val in values:
            if isinstance(val, (int, float, bool)):
                formatted_values.append(str(val))
            elif isinstance(val, str):
                # Escape single quotes
                escaped_val = val.replace("'", "''")
                formatted_values.append(f"'{escaped_val}'")
            else:
                # Convert to string and escape
                escaped_val = str(val).replace("'", "''")
                formatted_values.append(f"'{escaped_val}'")
        
        values_str = ", ".join(formatted_values)
        
        # Create SQL query
        sql = f"SELECT * FROM {layer_name} WHERE {attribute} IN ({values_str})"
        
        # Execute query
        result = source_ds.ExecuteSQL(sql, dialect="SQLITE")
        
        if result:
            # Copy to destination
            layer_srs = source_ds.GetLayerByName(layer_name).GetSpatialRef()
            dest_ds.CopyLayer(result, layer_name, ["OVERWRITE=YES"])
            source_ds.ReleaseResultSet(result)
        else:
            # Create empty layer with same schema if no features match
            source_layer = source_ds.GetLayerByName(layer_name)
            dest_layer = dest_ds.CreateLayer(layer_name, srs=layer_srs, 
                                            geom_type=source_layer.GetGeomType())
            
            # Copy field definitions
            layer_defn = source_layer.GetLayerDefn()
            for i in range(layer_defn.GetFieldCount()):
                dest_layer.CreateField(layer_defn.GetFieldDefn(i))

    dest_ds.FlushCache()
    
    return out_path


def vector_extract_by_attribute(
    vector: Union[str, ogr.DataSource, gdal.Dataset, List[Union[str, ogr.DataSource, gdal.Dataset]]],
    attribute: str,
    values: Union[Any, List[Any]],
    out_path: Optional[Union[str, List[str]]] = None,
    process_layer: int = -1,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    overwrite: bool = True,
) -> Union[str, List[str]]:
    """Extract vector features where attribute matches specified values.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource, gdal.Dataset, List[Union[str, ogr.DataSource, gdal.Dataset]]]
        The vector(s) to extract features from.

    attribute : str
        The attribute field to filter on.

    values : Union[Any, List[Any]]
        The value(s) to match. Features with attribute values in this list will be extracted.

    out_path : Optional[Union[str, List[str]]], optional
        The output path(s), default: None

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

    Returns
    -------
    Union[str, List[str]]
        The output path(s) for the extracted vector(s)
    """
    utils_base._type_check(vector, [ogr.DataSource, gdal.Dataset, str, [str, ogr.DataSource, gdal.Dataset]], "vector")
    utils_base._type_check(attribute, [str], "attribute")
    utils_base._type_check(out_path, [str, [str], None], "out_path")
    utils_base._type_check(process_layer, [int], "process_layer")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(overwrite, [bool], "overwrite")

    # Convert single value to list
    if not isinstance(values, list):
        values = [values]

    input_is_list = isinstance(vector, list)
    vector_list = utils_base._get_variable_as_list(vector)

    assert utils_gdal._check_is_vector_list(vector_list), f"Invalid input vector: {vector_list}"

    # Handle output paths
    if out_path is None:
        # Create temp paths for each input
        path_list = []
        for path in vector_list:
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
            _vector_extract_by_attribute(
                in_vector,
                attribute=attribute,
                values=values,
                out_path=path_list[index],
                process_layer=process_layer,
                overwrite=overwrite,
            )
        )

    if input_is_list:
        return output

    return output[0]
