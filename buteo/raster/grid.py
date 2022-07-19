"""
### Create grids from rasters.  ###

Cut rasters to grids. Use vectors or rasters as grids.

TODO:
    * raster_to_grid without geom.
    * split rasters into grid of x tiles.
"""

# Standard library
import sys; sys.path.append("../../")
from uuid import uuid4

# External
from osgeo import gdal, ogr

# Internal
from buteo.utils import bbox_utils, core_utils, gdal_utils
from buteo.raster import core_raster
from buteo.raster.clip import _clip_raster
from buteo.vector import core_vector
from buteo.vector.intersect import _intersect_vector
from buteo.vector.reproject import _reproject_vector



def raster_to_grid(
    raster,
    grid,
    out_dir,
    *,
    use_field=None,
    generate_vrt=True,
    overwrite=True,
    process_layer=0,
    creation_options=None,
    verbose=0,
):
    """
    Clips a raster to a grid. Generate .vrt.

    ## Args:
    `raster` (_str_/_gdal.DataSet_): The input raster. </br>
    `grid` (_str_/ogr.DataSource): The grid to use. </br>
    `out_dir` (_str_): The output directory. </br>

    ## Kwargs:
    `use_field` (_str_/_None_): A field to use to name the grid cells. </br>
    `generate_vrt` (_bool_): If **True**, the output raster will be a .vrt. </br>
    `overwrite` (_bool_): If **True**, the output raster will be overwritten. </br>
    `process_layer` (_int_): The layer from the grid to process. </br>
    `creation_options` (_list_/_None_): Creation options for the output raster. </br>
    `verbose` (_int_): The verbosity level. </br>

    ## Returns:
    (_str_): The filepath for the newly created raster.
    """
    core_utils.type_check(raster, [str, gdal.Dataset], "raster")
    core_utils.type_check(grid, [str, ogr.DataSource], "grid")
    core_utils.type_check(out_dir, [str], "out_dir")
    core_utils.type_check(overwrite, [bool], "overwrite")
    core_utils.type_check(process_layer, [int], "process_layer")
    core_utils.type_check(creation_options, [[str], None], "creation_options")
    core_utils.type_check(verbose, [int], "verbose")

    use_grid = core_vector.open_vector(grid)
    grid_metadata = core_vector._vector_to_metadata(use_grid)
    raster_metadata = core_raster.raster_to_metadata(raster)

    # Reproject raster if necessary.
    if not raster_metadata["projection_osr"].IsSame(grid_metadata["projection_osr"]):
        use_grid = _reproject_vector(grid, raster_metadata["projection_osr"])
        grid_metadata = core_vector._vector_to_metadata(use_grid)

        if not isinstance(grid_metadata, dict):
            raise Exception("Error while parsing metadata.")

    # Only use the polygons in the grid that intersect the extent of the raster.
    use_grid = _intersect_vector(use_grid, raster_metadata["extent_datasource"]())

    ref = core_raster._open_raster(raster)
    use_grid = core_vector.open_vector(use_grid)

    layer = use_grid.GetLayer(process_layer)
    feature_count = layer.GetFeatureCount()
    raster_extent = raster_metadata["bbox"]
    filetype = core_utils.path_to_ext(raster)
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

        if not bbox_utils.bboxes_intersect(raster_extent, geom.GetEnvelope()):
            continue

        intersections += 1

    layer.ResetReading()

    if verbose:
        print(f"Found {intersections} intersections.")

    if intersections == 0:
        print("Warning: Found 0 intersections. Returning empty list.")
        return ([], None)

    driver = ogr.GetDriverByName("FlatGeoBuf")

    clipped = 0
    for _ in range(feature_count):

        feature = layer.GetNextFeature()
        geom = feature.GetGeometryRef()

        if not bbox_utils.bboxes_intersect(raster_extent, geom.GetEnvelope()):
            continue

        if verbose == 1:
            core_utils.progress(clipped, intersections - 1, "clip_grid")

        fid = feature.GetFID()

        test_ds_path = f"/vsimem/grid_{uuid4().int}.fgb"
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

        _clip_raster(
            ref,
            test_ds_path,
            out_path=out_name,
            adjust_bbox=True,
            crop_to_geom=True,
            all_touch=False,
            suffix="",
            prefix="",
            creation_options=gdal_utils.default_creation_options(creation_options),
            verbose=0,
        )

        generated.append(out_name)
        clipped += 1

    if generate_vrt:
        vrt_name = f"{out_dir}{name}.vrt"
        core_raster.stack_rasters_vrt(generated, vrt_name, seperate=False)

        return (generated, vrt_name)

    return generated
