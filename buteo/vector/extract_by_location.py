"""### Extract vector features by spatial relationship with other geometries. ###

Filter vector features based on spatial relationships like intersect, within, etc.
"""

# Standard library
from typing import Union, Optional, List, Literal, cast

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


SpatialRelationship = Literal[
    "intersects", "contains", "within", "crosses", "touches", "overlaps"
]


def _vector_extract_by_location(
    vector: Union[str, ogr.DataSource],
    reference: Union[str, ogr.DataSource, gdal.Dataset],
    relationship: SpatialRelationship = "intersects",
    out_path: Optional[str] = None,
    invert: bool = False,
    process_layer: int = -1,
    overwrite: bool = True,
) -> str:
    """Internal implementation to extract vector features by spatial relationship."""
    assert utils_gdal._check_is_vector(vector), "Invalid input vector"
    assert utils_gdal._check_is_vector(reference), "Invalid reference vector"
    assert relationship in ["intersects", "contains", "within", "crosses", "touches", "overlaps"], \
        f"Invalid relationship: {relationship}. Must be one of: intersects, contains, within, crosses, touches, overlaps"

    if out_path is None:
        out_path = utils_path._get_temp_filepath(vector, suffix="_extracted_loc", ext="gpkg")

    assert utils_path._check_is_valid_output_filepath(out_path, overwrite=overwrite), "Invalid output path"

    out_format = utils_gdal._get_vector_driver_name_from_path(out_path)
    driver = ogr.GetDriverByName(out_format)

    # Open source vectors
    source_ds = vector_open(vector, writeable=False)
    if isinstance(source_ds, list):
        if not source_ds:
            raise ValueError("No valid vector datasource found")
        source_ds = source_ds[0]
        
    # Open reference - need to handle if it's a gdal.Dataset
    if isinstance(reference, gdal.Dataset):
        # Convert gdal.Dataset to path if possible
        ref_path = reference.GetDescription()
        if ref_path and utils_gdal._check_is_vector(ref_path):
            ref_ds = vector_open(ref_path, writeable=False)
        else:
            raise ValueError("Cannot convert gdal.Dataset to valid vector reference")
    else:
        ref_ds = vector_open(reference, writeable=False)
        
    if isinstance(ref_ds, list):
        if not ref_ds:
            raise ValueError("No valid reference datasource found")
        ref_ds = ref_ds[0]
    
    # Get metadata 
    source_meta = get_metadata_vector(source_ds)
    ref_meta = get_metadata_vector(ref_ds)
    
    # Validate projection match
    source_proj = source_meta["layers"][0]["projection_osr"]
    ref_proj = ref_meta["layers"][0]["projection_osr"]
    
    if source_proj and ref_proj and not source_proj.IsSame(ref_proj):
        raise ValueError("Source and reference vectors have different projections. "
                         "Please reproject one of them to match the other.")
    
    # Determine which layers to process
    layers_to_process = []
    
    if process_layer == -1:
        # Process all layers
        for idx, layer_meta in enumerate(source_meta["layers"]):
            layers_to_process.append((idx, layer_meta["layer_name"]))
    else:
        # Process just the specified layer
        if process_layer >= len(source_meta["layers"]):
            raise ValueError(f"Invalid layer index: {process_layer}. Only {len(source_meta['layers'])} layers available.")
        
        layer_meta = source_meta["layers"][process_layer]
        layers_to_process.append((process_layer, layer_meta["layer_name"]))
    
    # Delete existing file if needed
    utils_io._delete_if_required(out_path, overwrite=overwrite)
    
    # Create destination
    dest_ds = driver.CreateDataSource(out_path)
    if dest_ds is None:
        raise RuntimeError(f"Could not create output datasource: {out_path}")

    # Process each layer
    for _, layer_name in layers_to_process:
        source_layer = source_ds.GetLayerByName(layer_name)
        dest_layer = dest_ds.CreateLayer(
            layer_name, 
            srs=source_layer.GetSpatialRef(),
            geom_type=source_layer.GetGeomType()
        )
        
        # Copy field definitions
        layer_defn = source_layer.GetLayerDefn()
        for i in range(layer_defn.GetFieldCount()):
            dest_layer.CreateField(layer_defn.GetFieldDefn(i))
        
        # For each source feature, check if it satisfies the spatial relationship with any reference feature
        source_layer.ResetReading()
        for source_feat in source_layer:
            source_geom = source_feat.GetGeometryRef()
            if not source_geom:
                continue
                
            # Check against each reference layer
            match_found = False
            for ref_layer_meta in ref_meta["layers"]:
                ref_layer = ref_ds.GetLayerByName(ref_layer_meta["layer_name"])
                ref_layer.ResetReading()
                
                for ref_feat in ref_layer:
                    ref_geom = ref_feat.GetGeometryRef()
                    if not ref_geom:
                        continue
                    
                    # Check spatial relationship
                    satisfies_relation = False
                    if relationship == "intersects":
                        satisfies_relation = source_geom.Intersects(ref_geom)
                    elif relationship == "contains":
                        satisfies_relation = source_geom.Contains(ref_geom)
                    elif relationship == "within":
                        satisfies_relation = source_geom.Within(ref_geom)
                    elif relationship == "crosses":
                        satisfies_relation = source_geom.Crosses(ref_geom)
                    elif relationship == "touches":
                        satisfies_relation = source_geom.Touches(ref_geom)
                    elif relationship == "overlaps":
                        satisfies_relation = source_geom.Overlaps(ref_geom)
                    
                    if satisfies_relation:
                        match_found = True
                        break
                
                if match_found:
                    break
            
            # Add feature if it matches the criteria, considering the invert flag
            if (match_found and not invert) or (not match_found and invert):
                dest_layer.CreateFeature(source_feat)

    dest_ds.FlushCache()
    
    return out_path


def vector_extract_by_location(
    vector: Union[str, ogr.DataSource, gdal.Dataset, List[Union[str, ogr.DataSource, gdal.Dataset]]],
    reference: Union[str, ogr.DataSource, gdal.Dataset],
    relationship: SpatialRelationship = "intersects",
    out_path: Optional[Union[str, List[str]]] = None,
    invert: bool = False,
    process_layer: int = -1,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    overwrite: bool = True,
) -> Union[str, List[str]]:
    """Extract vector features based on spatial relationship with reference geometry.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource, gdal.Dataset, List[Union[str, ogr.DataSource, gdal.Dataset]]]
        The vector(s) to extract features from.

    reference : Union[str, ogr.DataSource, gdal.Dataset]
        The reference vector to check spatial relationships against.

    relationship : str, optional
        The spatial relationship to check. One of:
        "intersects", "contains", "within", "crosses", "touches", "overlaps".
        Default: "intersects"

    out_path : Optional[Union[str, List[str]]], optional
        The output path(s), default: None

    invert : bool, optional
        If True, extract features that do NOT satisfy the relationship.
        Default: False

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
    utils_base._type_check(reference, [ogr.DataSource, gdal.Dataset, str], "reference")
    utils_base._type_check(relationship, [str], "relationship")
    utils_base._type_check(out_path, [str, [str], None], "out_path")
    utils_base._type_check(invert, [bool], "invert")
    utils_base._type_check(process_layer, [int], "process_layer")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(overwrite, [bool], "overwrite")

    if relationship not in ["intersects", "contains", "within", "crosses", "touches", "overlaps"]:
        raise ValueError(f"Invalid relationship: {relationship}. Must be one of: "
                         "intersects, contains, within, crosses, touches, overlaps")

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
            _vector_extract_by_location(
                in_vector,
                reference=reference,
                relationship=relationship,
                out_path=path_list[index],
                invert=invert,
                process_layer=process_layer,
                overwrite=overwrite,
            )
        )

    if input_is_list:
        return output

    return output[0]
