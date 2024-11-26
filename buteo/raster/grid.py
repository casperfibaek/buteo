"""### Create grids from rasters.  ###

Cut rasters to grids. Use vectors or rasters as grids.
"""

# TODO: Verify this function and create more grid options.

# Standard library
from uuid import uuid4
from typing import Union, Optional, List

# External
from osgeo import gdal, ogr

# Internal
from buteo.utils import utils_base, utils_gdal, utils_bbox, utils_path
from buteo.raster import core_raster, stack
from buteo.raster.clip import _raster_clip
from buteo.vector import core_vector
from buteo.vector.intersect import _vector_intersect
from buteo.vector.reproject import _vector_reproject
from buteo.vector.metadata import _vector_to_metadata



def raster_to_grid(
    raster: Union[str, gdal.Dataset],
    grid: Union[str, ogr.DataSource],
    out_dir: str,
    *,
    use_field: bool = None,
    generate_vrt: bool = True,
    overwrite: bool = True,
    process_layer: int = 0,
    creation_options: Optional[List[str]] = None,
    verbose: int = 0,
) -> str:
    """Clips a raster to a grid. Generates .vrt.

    Parameters
    ----------
    raster : Union[str, gdal.Dataset]
        The input raster.

    grid : Union[str, ogr.DataSource]
        The grid to use.

    out_dir : str
        The output directory.

    use_field : bool, optional
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
    str
        The file path for the newly created raster.
    """
    utils_base._type_check(raster, [str, gdal.Dataset], "raster")
    utils_base._type_check(grid, [str, ogr.DataSource], "grid")
    utils_base._type_check(out_dir, [str], "out_dir")
    utils_base._type_check(overwrite, [bool], "overwrite")
    utils_base._type_check(process_layer, [int], "process_layer")
    utils_base._type_check(creation_options, [[str], None], "creation_options")
    utils_base._type_check(verbose, [int], "verbose")

    use_grid = core_vector.vector_open(grid)
    grid_metadata = _vector_to_metadata(use_grid)
    raster_metadata = core_raster.get_metadata_raster(raster)

    # Reproject raster if necessary.
    if not raster_metadata["projection_osr"].IsSame(grid_metadata["projection_osr"]):
        use_grid = _vector_reproject(grid, raster_metadata["projection_osr"])
        grid_metadata = _vector_to_metadata(use_grid)

        if not isinstance(grid_metadata, dict):
            raise RuntimeError("Error while parsing metadata.")

    # Only use the polygons in the grid that intersect the extent of the raster.
    use_grid = _vector_intersect(use_grid, raster_metadata["extent_datasource"]())

    ref = core_raster._open_raster(raster)
    use_grid = core_vector.vector_open(use_grid)

    layer = use_grid.GetLayer(process_layer)
    feature_count = layer.GetFeatureCount()
    raster_extent = raster_metadata["bbox"]
    filetype = utils_path._get_ext_from_path(raster)
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

        if not utils_bbox._check_bboxes_intersect(raster_extent, geom.GetEnvelope()):
            continue

        intersections += 1

    layer.ResetReading()

    if verbose:
        print(f"Found {intersections} intersections.")

    if intersections == 0:
        print("Warning: Found 0 intersections. Returning empty list.")
        return ([], None)

    driver = ogr.GetDriverByName("GPKG")

    clipped = 0
    for _ in range(feature_count):

        feature = layer.GetNextFeature()
        geom = feature.GetGeometryRef()

        if not utils_bbox._check_bboxes_intersect(raster_extent, geom.GetEnvelope()):
            continue

        if verbose == 1:
            utils_base.progress(clipped, intersections - 1, "clip_grid")

        fid = feature.GetFID()

        test_ds_path = f"/vsimem/grid_{uuid4().int}.gpkg"
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

        _raster_clip(
            ref,
            test_ds_path,
            out_path=out_name,
            adjust_bbox=True,
            crop_to_geom=True,
            all_touch=False,
            suffix="",
            prefix="",
            creation_options=utils_gdal._get_default_creation_options(creation_options),
            verbose=0,
        )

        generated.append(out_name)
        clipped += 1

    if generate_vrt:
        vrt_name = f"{out_dir}{name}.vrt"

        stack.raster_stack_vrt_list(generated, vrt_name, separate=False)

        return (generated, vrt_name)

    return generated
