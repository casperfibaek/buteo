import sys

sys.path.append("../../")
from uuid import uuid4
from osgeo import gdal, ogr
from typing import Union, Optional

from buteo.vector.io import vector_to_metadata

from buteo.vector.intersect import intersect_vector
from buteo.vector.reproject import reproject_vector
from buteo.raster.clip import clip_raster
from buteo.raster.io import raster_to_metadata, stack_rasters_vrt

from buteo.utils import (
    path_to_ext,
    progress,
    type_check,
)
from buteo.gdal_utils import (
    ogr_bbox_intersects,
    vector_to_reference,
    raster_to_reference,
    default_options,
)


def raster_to_grid(
    raster: Union[str, gdal.Dataset],
    grid: Union[str, ogr.DataSource],
    out_dir: str,
    use_field: Optional[str] = None,
    generate_vrt: bool = True,
    overwrite: bool = True,
    process_layer: int = 0,
    creation_options: list = [],
    opened: bool = False,
) -> Union[list, tuple]:
    """ Clips a raster to a grid. Generate .vrt.

    Returns:
        The filepath for the newly created raster.
    """
    type_check(raster, [str, gdal.Dataset], "raster")
    type_check(grid, [str, ogr.DataSource], "grid")
    type_check(out_dir, [str], "out_dir")
    type_check(overwrite, [bool], "overwrite")
    type_check(process_layer, [int], "process_layer")
    type_check(creation_options, [list], "creation_options")
    type_check(opened, [bool], "opened")

    raster_metadata = raster_to_metadata(raster, simple=False)
    grid_metadata = vector_to_metadata(grid)

    # This is necessary to ensure mypy that the metadata is a dictionary.
    if not isinstance(raster_metadata, dict):
        raise Exception("Error while parsing metadata.")

    if not isinstance(grid_metadata, dict):
        raise Exception("Error while parsing metadata.")

    # Reproject raster if necessary.
    use_grid = vector_to_reference(grid)
    if not raster_metadata["projection_osr"].IsSame(grid_metadata["projection_osr"]):
        use_grid = reproject_vector(
            grid, raster_metadata["projection_osr"], opened=True
        )
        grid_metadata = vector_to_metadata(use_grid)

        if not isinstance(grid_metadata, dict):
            raise Exception("Error while parsing metadata.")

    # Only use the polygons in the grid that intersect the extent of the raster.
    use_grid = intersect_vector(
        use_grid, raster_metadata["extent_datasource"], opened=True
    )

    ref = raster_to_reference(raster)

    shp_driver = ogr.GetDriverByName("Esri Shapefile")

    layer = use_grid.GetLayer(process_layer)
    feature_count = layer.GetFeatureCount()
    raster_extent = raster_metadata["extent_ogr"]
    filetype = path_to_ext(raster)
    basename = raster_metadata["basename"]
    geom_type = grid_metadata["layers"][process_layer]["geom_type_ogr"]

    if use_field is not None:
        if use_field not in grid_metadata["layers"][process_layer]["field_names"]:
            raise ValueError("Requested field not found.")

    generated = []

    for _ in range(feature_count):
        progress(_, feature_count - 1, "clip_grid")

        feature = layer.GetNextFeature()
        geom = feature.GetGeometryRef()

        if not ogr_bbox_intersects(raster_extent, geom.GetEnvelope()):
            continue

        fid = feature.GetFID()

        test_ds_path = f"/vsimem/grid_{uuid4().int}.shp"
        test_ds = shp_driver.CreateDataSource(test_ds_path)
        test_ds_lyr = test_ds.CreateLayer(
            "test_mem_grid_layer",
            geom_type=geom_type,
            srs=raster_metadata["projection_osr"],
        )
        test_ds_lyr.CreateFeature(feature.Clone())
        test_ds.FlushCache()

        out_name = None

        if use_field is not None:
            out_name = f"{out_dir}{feature.GetField(use_field)}{filetype}"
        else:
            out_name = f"{out_dir}{basename}_{fid}{filetype}"

        generated.append(out_name)

        clipped = clip_raster(
            ref,
            test_ds_path,
            out_path=out_name,
            adjust_bbox=True,
            crop_to_geom=True,
            all_touch=False,
            postfix="",
            prefix="",
            creation_options=creation_options,
            verbose=0,
        )

    if generate_vrt:
        vrt_name = f"{out_dir}{basename}.vrt"
        stack_rasters_vrt(generated, vrt_name, seperate=False)

        return (vrt_name, generated)

    return generated


if __name__ == "__main__":
    folder = "C:/Users/caspe/Desktop/test/out/"

    vector = folder + "denmark_10km_grid.gpkg"
    raster = folder + "predicted_raster_32-16.tif"

    raster_to_grid(raster, vector, folder + "snow/", use_field="KN10kmDK")
