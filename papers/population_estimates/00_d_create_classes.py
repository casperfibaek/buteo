import sys
from glob import glob
import os
import numpy as np

sys.path.append("../../")
from buteo.vector.rasterize import rasterize_vector
from buteo.raster.io import raster_to_array, array_to_raster
from buteo.raster.clip import clip_raster
from buteo.raster.resample import resample_raster


def create_labels(buildings, b4, output, size=0.4):
    cm40 = rasterize_vector(
        buildings,
        size,
        extent=b4,
        attribute="type",
        all_touch=True,
        nodata_value=0,
    )

    resampled = resample_raster(
        cm40,
        b4,
        resample_alg="mode",
        dst_nodata=0,
    )
    return array_to_raster(
        np.rint(raster_to_array(resampled)).astype("uint8"),
        reference=b4,
        out_path=output,
    )


folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/ghana/vector/"
outdir = folder + "classes/"

buildings = folder + "vector/kampala_buildings.gpkg"
b4 = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/ghana/raster/2021_B04_10m.tif"

for grid_cell in glob(folder + "classes/zones/zone_id_*.gpkg"):
    number = os.path.splitext(os.path.basename(grid_cell))[0]
    number = number.split("_")[2]
    building = folder + f"classes/zones/building_zone_{number}.gpkg"
    clipped_b4 = clip_raster(b4, grid_cell, all_touch=False, adjust_bbox=False)
    create_labels(
        building,
        clipped_b4,
        outdir + f"class_{number}.tif",
        size=0.2,
    )
