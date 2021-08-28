yellow_follow = "C:/Users/caspe/Desktop/buteo/"
import sys
import os

sys.path.append(yellow_follow)

from glob import glob
from buteo.raster.io import stack_rasters
from buteo.machine_learning.patch_extraction import predict_raster
from buteo.raster.io import raster_to_array, array_to_raster
from utils import preprocess_optical, preprocess_sar

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/analysis/ghana/"
raster_folder = (
    "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/ghana/raster/clipped/"
)


rgbn = stack_rasters(
    [
        array_to_raster(
            preprocess_optical(raster_to_array(raster_folder + "2021_B02_10m.tif")),
            reference=raster_folder + "2021_B02_10m.tif",
        ),
        array_to_raster(
            preprocess_optical(raster_to_array(raster_folder + "2021_B03_10m.tif")),
            reference=raster_folder + "2021_B03_10m.tif",
        ),
        array_to_raster(
            preprocess_optical(raster_to_array(raster_folder + "2021_B04_10m.tif")),
            reference=raster_folder + "2021_B04_10m.tif",
        ),
        array_to_raster(
            preprocess_optical(raster_to_array(raster_folder + "2021_B08_10m.tif")),
            reference=raster_folder + "2021_B08_10m.tif",
        ),
    ]
)

swir = stack_rasters(
    [
        array_to_raster(
            preprocess_optical(raster_to_array(raster_folder + "2021_B11_20m.tif")),
            reference=raster_folder + "2021_B11_20m.tif",
        ),
        array_to_raster(
            preprocess_optical(raster_to_array(raster_folder + "2021_B12_20m.tif")),
            reference=raster_folder + "2021_B12_20m.tif",
        ),
    ]
)

sar = stack_rasters(
    [
        array_to_raster(
            preprocess_sar(raster_to_array(raster_folder + "2021_VH.tif")),
            reference=raster_folder + "2021_VH.tif",
        ),
        array_to_raster(
            preprocess_sar(raster_to_array(raster_folder + "2021_VV.tif")),
            reference=raster_folder + "2021_VV.tif",
        ),
    ]
)


def predict_model(model, name):
    predict_raster(
        [rgbn, swir, sar],
        model,
        out_path=folder + f"predictions/{name}.tif",
        offsets=[
            [(16, 16), (32, 32), (48, 48)],
            [(8, 8), (16, 16), (24, 24)],
            [(16, 16), (32, 32), (48, 48)],
        ],
        device="gpu",
        output_size=64,
    )


model = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/analysis/ghana/vector/grid_cells/patches/merged/models/ghana_04_transfer_50"
# model = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/analysis/denmark/models/area_advanced_v07_01"

predict_model(model, "ghana_v03_transfer")
