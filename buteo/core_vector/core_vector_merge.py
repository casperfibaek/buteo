"""### Merge vectors. ###

Merges vectors into a single vector file.
"""

# Standard library
import os
from typing import Union, List, Optional, Tuple

# External
from osgeo import ogr
from osgeo_utils.ogrmerge import ogrmerge

# Internal
from buteo.utils import (
    utils_base,
    utils_gdal,
    utils_path,
    utils_projection,
)
from buteo.core_vector.core_vector_read import _open_vector


def vector_merge_layers(
    vectors: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
    out_path: Optional[str] = None,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    overwrite: bool = True,
) -> str:
    """Merge vector layers into a single file.
    
    This is a simpler version of vector_merge_features specifically for merging
    layers from the same or different vectors.

    Parameters
    ----------
    vectors : Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]]
        Vector or list of vectors to merge.
    out_path : Optional[str], optional
        Output path. If None, a temporary path is created. Default: None
    prefix : str, optional
        Prefix to add to output filename. Default: ""
    suffix : str, optional
        Suffix to add to output filename. Default: ""
    add_uuid : bool, optional
        Add a UUID to output filename. Default: False
    overwrite : bool, optional
        Whether to overwrite existing files. Default: True

    Returns
    -------
    str
        Path to the output merged vector file
    """
    # Input validation
    utils_base._type_check(vectors, [str, ogr.DataSource, [str, ogr.DataSource]], "vectors")
    utils_base._type_check(out_path, [str, None], "out_path")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(overwrite, [bool], "overwrite")

    # Convert to list if necessary
    vector_list = utils_base._get_variable_as_list(vectors)
    
    # Define output path
    if out_path is None:
        out_path = utils_path._get_temp_filepath(
            vector_list[0], 
            prefix=prefix, 
            suffix=suffix or "_merged", 
            add_uuid=add_uuid,
            ext="gpkg"
        )
    
    # Handle overwrite
    if utils_path._check_file_exists(out_path):
        if overwrite:
            utils_gdal._delete_raster_or_vector(out_path)
        else:
            raise ValueError(f"Output file already exists: {out_path}")

    # Create output datasource
    drv_name = utils_gdal._get_vector_driver_name_from_path(out_path)
    drv = ogr.GetDriverByName(drv_name)
    ds_out = drv.CreateDataSource(out_path)
    
    # Process each input vector
    for vector in vector_list:
        ds = _open_vector(vector)
        
        # Process each layer in the input vector
        for i in range(ds.GetLayerCount()):
            layer = ds.GetLayer(i)
            layer_name = layer.GetName()
            
            # Create output layer with same geometry type and spatial reference
            out_layer = ds_out.CreateLayer(
                layer_name,
                layer.GetSpatialRef(),
                layer.GetGeomType()
            )
            
            # Copy field definitions
            layer_defn = layer.GetLayerDefn()
            for i in range(layer_defn.GetFieldCount()):
                field_defn = layer_defn.GetFieldDefn(i)
                out_layer.CreateField(field_defn)
            
            # Copy features
            for feature in layer:
                out_layer.CreateFeature(feature.Clone())
    
    # Clean up
    ds_out.FlushCache()
    ds_out = None
    
    return out_path


def vector_merge_features(
    vectors: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
    out_path: Optional[str] = None,
    projection: Optional[str] = None,
    single_layer: bool = True,
    overwrite: bool = True,
    skip_failures: bool = True,
) -> str:
    """Merge vectors to a single geopackage.

    Parameters
    ----------
    vectors : Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]]
        The input vectors.
    out_path : Optional[str], optional
        The output path, default: None.
    projection : Optional[str], optional
        The projection of the output vector, default: None.
        If None, the projection of the first input vector is used.
    single_layer : bool, optional
        If True, all layers are merged into a single layer, default: True.
    overwrite : bool, optional
        If True, the output file is overwritten if it exists, default: True.
    skip_failures : bool, optional
        If True, failures are skipped, default: True.

    Returns
    -------
    str
        The output path.

    Raises
    ------
    ValueError
        If input vectors are invalid or empty.
    RuntimeError
        If merge operation fails.
    """
    # Input validation
    utils_base._type_check(vectors, [[str, ogr.DataSource]], "vector")
    utils_base._type_check(out_path, [str, [str], None], "out_path")
    utils_base._type_check(projection, [str, None], "projection")
    utils_base._type_check(single_layer, [bool], "single_layer")
    utils_base._type_check(overwrite, [bool], "overwrite")
    utils_base._type_check(skip_failures, [bool], "skip_failures")

    vector_list = utils_base._get_variable_as_list(vectors)

    if not vector_list:
        raise ValueError("No input vectors provided")

    if not utils_gdal._check_is_vector_list(vector_list):
        raise ValueError("Invalid input vector list")

    # Check if all input vectors exist and are readable
    for vec in vector_list:
        if isinstance(vec, str) and not utils_path._check_file_exists(vec):
            raise ValueError(f"Input vector does not exist: {vec}")

    # Setup output path
    if out_path is None:
        out_path = utils_path._get_temp_filepath(vector_list[0], suffix="_merged")
    elif utils_path._check_file_exists(out_path) and not overwrite:
        raise ValueError(f"Output file already exists: {out_path}")

    # Get driver
    driver_name = utils_gdal._get_driver_name_from_path(out_path)
    if not driver_name:
        raise ValueError(f"Could not determine driver for output path: {out_path}")

    # Setup projection
    try:
        if projection is None:
            target_projection = utils_projection._get_projection_from_vector(vector_list[0]).ExportToWkt()
        else:
            target_projection = utils_projection.parse_projection_wkt(projection)
    except Exception as e:
        raise ValueError(f"Failed to process projection: {str(e)}") from e

    # Ensure output directory exists
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # First, create an empty target file to avoid "No such file or directory" error
    driver = ogr.GetDriverByName(driver_name)
    if driver is None:
        raise RuntimeError(f"Could not get driver: {driver_name}")
    
    # Delete existing file if overwrite=True
    if overwrite and utils_path._check_file_exists(out_path):
        utils_gdal._delete_raster_or_vector(out_path)
    
    # Create empty dataset
    ds = driver.CreateDataSource(out_path)
    if ds is None:
        raise RuntimeError(f"Could not create dataset: {out_path}")
    ds = None  # Close the dataset

    # Perform merge
    try:
        success = ogrmerge(
            vector_list,
            out_path,
            single_layer=single_layer,
            driver_name=driver_name,
            t_srs=target_projection,
            overwrite_ds=overwrite,
            skip_failures=skip_failures,
            src_layer_field_name="og_name",
            src_layer_field_content="{LAYER_NAME}"
        )

        if success is None:
            raise RuntimeError("Vector merge operation failed")

        # Verify output file was created
        if not utils_path._check_file_exists(out_path):
            raise RuntimeError("Output file was not created")

        return out_path

    except Exception as e:
        if utils_path._check_file_exists(out_path):
            try:
                utils_gdal._delete_raster_or_vector(out_path)
            except (OSError, PermissionError):
                pass
        raise RuntimeError(f"Vector merge operation failed: {str(e)}") from e
