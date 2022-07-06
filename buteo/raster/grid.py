"""
Cut a rasters to a grid.

TODO:
    - Improve documentation
    - Raster_to_grid without geom
    - Split rasters into grid of x tiles
"""

import sys; sys.path.append("../../") # Path: buteo/raster/grid.py
from uuid import uuid4
from typing import Union, Optional, Tuple, List

from osgeo import gdal, ogr

sys.path.append("../../")

from buteo.vector.io import open_vector, internal_vector_to_metadata
from buteo.vector.intersect import intersect_vector
from buteo.vector.reproject import reproject_vector
from buteo.raster.clip import clip_raster
from buteo.raster.io import open_raster, raster_to_metadata, stack_rasters_vrt
from buteo.utils.gdal_utils import ogr_bbox_intersects, default_options
from buteo.utils.core import path_to_ext, progress, type_check


# TODO: raster_to_grid without geom.
# TODO: split raster in to raster grid of x tiles


def raster_to_grid(
    raster: Union[str, gdal.Dataset],
    grid: Union[str, ogr.DataSource],
    out_dir: str,
    use_field: Optional[str] = None,
    generate_vrt: bool = True,
    overwrite: bool = True,
    process_layer: int = 0,
    creation_options: list = [],
    verbose: int = 1,
) -> Union[List[str], Tuple[Optional[List[str]], Optional[str]]]:
    """Clips a raster to a grid. Generate .vrt.

    Returns:
        The filepath for the newly created raster.
    """
    type_check(raster, [str, gdal.Dataset], "raster")
    type_check(grid, [str, ogr.DataSource], "grid")
    type_check(out_dir, [str], "out_dir")
    type_check(overwrite, [bool], "overwrite")
    type_check(process_layer, [int], "process_layer")
    type_check(creation_options, [list], "creation_options")
    type_check(verbose, [int], "verbose")

    use_grid = open_vector(grid)
    grid_metadata = internal_vector_to_metadata(use_grid)
    raster_metadata = raster_to_metadata(raster, create_geometry=True)

    # Reproject raster if necessary.
    if not raster_metadata["projection_osr"].IsSame(grid_metadata["projection_osr"]):
        use_grid = reproject_vector(grid, raster_metadata["projection_osr"])
        grid_metadata = internal_vector_to_metadata(use_grid)

        if not isinstance(grid_metadata, dict):
            raise Exception("Error while parsing metadata.")

    # Only use the polygons in the grid that intersect the extent of the raster.
    use_grid = intersect_vector(use_grid, raster_metadata["extent_datasource"])

    ref = open_raster(raster)
    use_grid = open_vector(use_grid)

    layer = use_grid.GetLayer(process_layer)
    feature_count = layer.GetFeatureCount()
    raster_extent = raster_metadata["extent_ogr"]
    filetype = path_to_ext(raster)
    name = raster_metadata["name"]
    geom_type = grid_metadata["layers"][process_layer]["geom_type_ogr"]

    if use_field is not None:
        if use_field not in grid_metadata["layers"][process_layer]["field_names"]:
            names = grid_metadata["layers"][process_layer]["field_names"]
            raise ValueError(
                f"Requested field not found. Fields available are: {names}"
            )

    generated = []

    # For the sake of good reporting - lets first establish how many features intersect
    # the raster.

    if verbose:
        print("Finding intersections.")

    intersections = 0
    for _ in range(feature_count):
        feature = layer.GetNextFeature()
        geom = feature.GetGeometryRef()

        if not ogr_bbox_intersects(raster_extent, geom.GetEnvelope()):
            continue

        intersections += 1

    layer.ResetReading()

    if verbose:
        print(f"Found {intersections} intersections.")

    if intersections == 0:
        print("Warning: Found 0 intersections. Returning empty list.")
        return ([], None)

    # TODO: Replace this in gdal. 3.1
    driver = ogr.GetDriverByName("Esri Shapefile")

    clipped = 0
    for _ in range(feature_count):

        feature = layer.GetNextFeature()
        geom = feature.GetGeometryRef()

        if not ogr_bbox_intersects(raster_extent, geom.GetEnvelope()):
            continue

        if verbose == 1:
            progress(clipped, intersections - 1, "clip_grid")

        fid = feature.GetFID()

        test_ds_path = f"/vsimem/grid_{uuid4().int}.shp"
        test_ds = driver.CreateDataSource(test_ds_path)
        test_ds_lyr = test_ds.CreateLayer(
            "mem_layer_grid",
            geom_type=geom_type,
            srs=raster_metadata["projection_osr"],
        )
        test_ds_lyr.CreateFeature(feature.Clone())
        test_ds.FlushCache()

        out_name = None

        if use_field is not None:
            out_name = f"{out_dir}{feature.GetField(use_field)}{filetype}"
        else:
            out_name = f"{out_dir}{name}_{fid}{filetype}"

        clip_raster(
            ref,
            test_ds_path,
            out_path=out_name,
            adjust_bbox=True,
            crop_to_geom=True,
            all_touch=False,
            postfix="",
            prefix="",
            creation_options=default_options(creation_options),
            verbose=0,
        )

        generated.append(out_name)
        clipped += 1

    if generate_vrt:
        vrt_name = f"{out_dir}{name}.vrt"
        stack_rasters_vrt(generated, vrt_name, seperate=False)

        return (generated, vrt_name)

    return generated
