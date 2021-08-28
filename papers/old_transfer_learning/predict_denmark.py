yellow_follow = "C:/Users/caspe/Desktop/buteo/"
import sys
import os

sys.path.append(yellow_follow)

from glob import glob
from buteo.raster.io import stack_rasters
from buteo.machine_learning.patch_extraction import predict_raster
from buteo.raster.io import raster_to_array, array_to_raster
from utils import preprocess_optical, preprocess_sar

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/analysis/denmark/"
raster_folder = "predictions/roskilde_rasters"
date = "2020"

rgbn = stack_rasters(
    [
        array_to_raster(
            preprocess_optical(
                raster_to_array(folder + f"{raster_folder}/{date}_B02_10m.tif")
            ),
            reference=folder + f"{raster_folder}/{date}_B02_10m.tif",
        ),
        array_to_raster(
            preprocess_optical(
                raster_to_array(folder + f"{raster_folder}/{date}_B03_10m.tif")
            ),
            reference=folder + f"{raster_folder}/{date}_B03_10m.tif",
        ),
        array_to_raster(
            preprocess_optical(
                raster_to_array(folder + f"{raster_folder}/{date}_B04_10m.tif")
            ),
            reference=folder + f"{raster_folder}/{date}_B04_10m.tif",
        ),
        array_to_raster(
            preprocess_optical(
                raster_to_array(folder + f"{raster_folder}/{date}_B08_10m.tif")
            ),
            reference=folder + f"{raster_folder}/{date}_B08_10m.tif",
        ),
    ]
)

swir = stack_rasters(
    [
        array_to_raster(
            preprocess_optical(
                raster_to_array(folder + f"{raster_folder}/{date}_B11_20m.tif")
            ),
            reference=folder + f"{raster_folder}/{date}_B11_20m.tif",
        ),
        array_to_raster(
            preprocess_optical(
                raster_to_array(folder + f"{raster_folder}/{date}_B12_20m.tif")
            ),
            reference=folder + f"{raster_folder}/{date}_B12_20m.tif",
        ),
    ]
)

sar = stack_rasters(
    [
        array_to_raster(
            preprocess_sar(raster_to_array(folder + f"{raster_folder}/{date}_VH.tif")),
            reference=folder + f"{raster_folder}/{date}_VH.tif",
        ),
        array_to_raster(
            preprocess_sar(raster_to_array(folder + f"{raster_folder}/{date}_VV.tif")),
            reference=folder + f"{raster_folder}/{date}_VV.tif",
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


main_model = "area_advanced_v07_01"
for model in glob(folder + f"models/{main_model}*"):
    model_name = os.path.basename(model)

    predict_model(model, model_name + "_" + date)
    # predict_model(model, "2021", model_name + "_2021")
