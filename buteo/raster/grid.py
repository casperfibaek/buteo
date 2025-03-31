"""### Create grids from rasters.  ###

Cut rasters to grids. Use vectors or rasters as grids.
"""

# Standard library
from uuid import uuid4
from typing import Union, Optional, List, Tuple
import os

# External
from osgeo import gdal, ogr

# Internal
from buteo.utils import utils_base, utils_gdal, utils_path
# Import necessary bbox functions from their new locations
from buteo.bbox.validation import _check_bboxes_intersect

from buteo.core_raster.core_raster_info import get_metadata_raster
from buteo.core_raster.core_raster_read import _open_raster
from buteo.core_raster.core_raster_stack import raster_stack_vrt_list
from buteo.raster.clip import _raster_clip
from buteo.core_vector.core_vector_read import open_vector
from buteo.core_vector.core_vector_info import get_metadata_vector
from buteo.vector.intersect import _vector_intersect
from buteo.vector.reproject import _vector_reproject
# Import raster_to_vector_extent correctly
from buteo.core_raster.core_raster_extent import raster_to_vector_extent as raster_to_extent

# Type Aliases
Raster = Union[str, gdal.Dataset]
Vector = Union[str, ogr.DataSource]


def raster_to_grid(
    raster: Raster,
    grid: Vector,
    out_dir: str,
    *,
    use_field: Optional[str] = None,
    generate_vrt: bool = True,
    overwrite: bool = True,
    process_layer: int = 0,
    creation_options: Optional[List[str]] = None,
    verbose: int = 0,
) -> Union[List[str], Tuple[List[str], Optional[str]]]:
    """Clips a raster to a grid. Generates .vrt.

    Parameters
    ----------
    raster : Raster
        The input raster.

    grid : Vector
        The grid to use.

    out_dir : str
        The output directory.

    use_field : Optional[str], optional
        A field to use to name the grid cells, default: None.

    generate_vrt : bool, optional
        If True, the output raster will be a .vrt, default: True.

    overwrite : bool, optional
        If True, the output raster will be overwritten, default: True.

    process_layer : int, optional
        The layer from the grid to process, default: 0.

    creation_options : Optional[List[str]], optional
        Creation options for the output raster, default: None.

    verbose : int, optional
        The verbosity level, default: 0.

    Returns
    -------
    Union[List[str], Tuple[List[str], Optional[str]]]
        If generate_vrt is True, returns a tuple of (list of clip paths, vrt path)
        If generate_vrt is False, returns a list of clip paths
    """
    utils_base._type_check(raster, [str, gdal.Dataset], "raster")
    utils_base._type_check(grid, [str, ogr.DataSource], "grid")
    utils_base._type_check(out_dir, [str], "out_dir")
    utils_base._type_check(overwrite, [bool], "overwrite")
    utils_base._type_check(process_layer, [int], "process_layer")
    utils_base._type_check(creation_options, [list, None], "creation_options")
    utils_base._type_check(verbose, [int], "verbose")

    # Open and get metadata
    raster_path = utils_gdal._get_path_from_dataset(raster)
    grid_path = utils_gdal._get_path_from_dataset(grid)

    use_grid_ds = open_vector(grid)
    if use_grid_ds is None: raise ValueError(f"Could not open grid vector: {grid}")
    grid_metadata = get_metadata_vector(use_grid_ds) # type: ignore
    raster_metadata = get_metadata_raster(raster)

    # Reproject grid if necessary.
    grid_proj = grid_metadata["layers"][process_layer]["projection_osr"]
    if not raster_metadata["projection_osr"].IsSame(grid_proj):
        reprojected_grid_path = _vector_reproject(use_grid_ds, raster_metadata["projection_osr"]) # type: ignore
        use_grid_ds = open_vector(reprojected_grid_path)
        if use_grid_ds is None: raise ValueError("Could not open reprojected grid.")
        grid_metadata = get_metadata_vector(use_grid_ds) # type: ignore
        # TODO: Manage deletion of temporary reprojected file if needed

    # Only use the polygons in the grid that intersect the extent of the raster.
    extent_path = raster_to_extent(raster)
    assert isinstance(use_grid_ds, ogr.DataSource), "Internal error: use_grid_ds is not DataSource"
    intersected_grid_path = _vector_intersect(use_grid_ds, extent_path) # type: ignore
    utils_gdal.delete_dataset_if_in_memory(extent_path) # type: ignore

    ref = _open_raster(raster)
    use_grid_intersected_ds = open_vector(intersected_grid_path) # type: ignore
    if use_grid_intersected_ds is None: raise ValueError("Could not open intersected grid vector.")

    layer = use_grid_intersected_ds.GetLayer(process_layer) # type: ignore
    feature_count = layer.GetFeatureCount()
    raster_extent = raster_metadata["bbox"]
    filetype = utils_path._get_ext_from_path(raster_path if raster_path else "tif")
    name = raster_metadata["name"]
    geom_type = grid_metadata["layers"][process_layer]["geom_type_ogr"]

    if use_field is not None:
        if use_field not in grid_metadata["layers"][process_layer]["field_names"]:
            names = grid_metadata["layers"][process_layer]["field_names"]
            raise ValueError(
                f"Requested field not found. Fields available are: {names}"
            )

    generated = []

    if verbose:
        print("Finding intersections.")

    intersections = 0
    for _ in range(feature_count):
        feature = layer.GetNextFeature()
        if feature is None: continue
        geom = feature.GetGeometryRef()
        if geom is None: continue

        if not _check_bboxes_intersect(raster_extent, geom.GetEnvelope()):
            continue

        intersections += 1

    layer.ResetReading()

    if verbose:
        print(f"Found {intersections} intersections.")

    if intersections == 0:
        print("Warning: Found 0 intersections. Returning empty list.")
        if generate_vrt:
            return ([], None)
        return []

    driver = ogr.GetDriverByName("GPKG")
    if driver is None: raise RuntimeError("GPKG Driver not available.")

    clipped = 0
    for _ in range(feature_count):
        feature = layer.GetNextFeature()
        if feature is None: continue
        geom = feature.GetGeometryRef()
        if geom is None: continue

        if not _check_bboxes_intersect(raster_extent, geom.GetEnvelope()):
            continue

        if verbose == 1:
            print(f"Processing: {clipped+1}/{intersections}")

        fid = feature.GetFID()

        temp_clip_path = f"/vsimem/grid_clip_{uuid4().int}.gpkg"
        temp_clip_ds = driver.CreateDataSource(temp_clip_path)
        if temp_clip_ds is None: raise RuntimeError(f"Could not create memory datasource: {temp_clip_path}")
        temp_clip_lyr = temp_clip_ds.CreateLayer(
            "clip_feature",
            geom_type=geom_type,
            srs=raster_metadata["projection_osr"],
        )
        if temp_clip_lyr is None: raise RuntimeError("Could not create memory layer.")
        temp_clip_lyr.CreateFeature(feature.Clone())
        temp_clip_ds.FlushCache()
        temp_clip_ds = None

        out_name = None
        if use_field is not None:
            field_val = feature.GetField(use_field)
            safe_field_val = utils_path._get_filename_from_path(str(field_val), with_ext=False) # Basic sanitization
            out_name = os.path.join(out_dir, f"{name}_{safe_field_val}{filetype}")
        else:
            out_name = os.path.join(out_dir, f"{name}_{fid}{filetype}")

        # Create minimal metadata for the clipping feature
        clip_feature_metadata = {
            "bbox": geom.GetEnvelope(),
            "projection_osr": raster_metadata["projection_osr"],
        }

        _raster_clip(
            ref,
            temp_clip_path,
            clip_metadata=clip_feature_metadata,
            out_path=out_name,
            adjust_bbox=True,
            crop_to_geom=True,
            all_touch=False,
            creation_options=utils_gdal._get_default_creation_options(creation_options),
            verbose=0,
        )

        utils_gdal.delete_dataset_if_in_memory(temp_clip_path)

        generated.append(out_name)
        clipped += 1

    ref = None
    use_grid_intersected_ds = None

    if generate_vrt:
        vrt_name = os.path.join(out_dir, f"{name}.vrt")
        raster_stack_vrt_list(generated, vrt_name, separate=False)
        return (generated, vrt_name)

    return generated
