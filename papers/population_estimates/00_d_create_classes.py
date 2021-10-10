import sys
from glob import glob
import os
import numpy as np

sys.path.append("../../")
from buteo.vector.rasterize import rasterize_vector
from buteo.raster.io import raster_to_array, array_to_raster
from buteo.raster.clip import clip_raster
from buteo.raster.resample import resample_raster
from buteo.utils import progress
from osgeo import gdal


def create_labels(buildings, reference, output, size=0.4):
    cm40 = rasterize_vector(
        buildings,
        size,
        extent=reference,
        attribute="type",
        all_touch=True,
        nodata_value=0,
    )

    cm40_arr = raster_to_array(cm40)

    uninhabited = array_to_raster((cm40_arr == 0).astype("uint8").filled(), cm40)
    residential = array_to_raster((cm40_arr == 1).astype("uint8").filled(), cm40)
    industrial = array_to_raster((cm40_arr == 2).astype("uint8").filled(), cm40)
    self_organised = array_to_raster((cm40_arr == 3).astype("uint8").filled(), cm40)

    resampled_uninhabited = resample_raster(
        uninhabited, reference, resample_alg="average"
    )
    resampled_residential = resample_raster(
        residential, reference, resample_alg="average"
    )
    resampled_industrial = resample_raster(
        industrial, reference, resample_alg="average"
    )
    resampled_self_organised = resample_raster(
        self_organised, reference, resample_alg="average"
    )

    merged = np.concatenate(
        [
            raster_to_array(resampled_uninhabited).filled(0),
            raster_to_array(resampled_residential).filled(0),
            raster_to_array(resampled_industrial).filled(0),
            raster_to_array(resampled_self_organised).filled(0),
        ],
        axis=2,
    ).astype("float32")

    for mem_raster in [
        cm40,
        uninhabited,
        residential,
        industrial,
        self_organised,
        resampled_uninhabited,
        resampled_residential,
        resampled_industrial,
        resampled_self_organised,
    ]:
        gdal.Unlink(mem_raster)

    return array_to_raster(
        merged,
        reference=reference,
        out_path=output,
    )


folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/ghana/vector/"
outdir = folder + "classes_extra/"

buildings = folder + "buildings_type_extra_01.gpkg"
b4 = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/ghana/raster_v2/2021_B04_10m.tif"

zones = glob(outdir + "zones/zone_*.gpkg")
for idx, grid_cell in enumerate(zones):
    progress(idx, len(zones), "Labeling")
    number = os.path.splitext(os.path.basename(grid_cell))[0]
    number = number.split("_")[1]
    clipped_b4 = clip_raster(b4, grid_cell, all_touch=False, adjust_bbox=False)
    create_labels(
        buildings,
        clipped_b4,
        outdir + f"class_{number}.tif",
        size=0.2,
    )
    progress(idx + 1, len(zones), "Labeling")
