yellow_follow = "C:/Users/caspe/Desktop/buteo/"
import sys

sys.path.append(yellow_follow)

import numpy as np
from buteo.raster.io import stack_rasters
from buteo.machine_learning.patch_extraction import predict_raster
from buteo.raster.io import raster_to_array, array_to_raster
from utils import preprocess_optical, preprocess_sar


folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/bornholm/"

model = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/machine_learning_data/models/denmark_02"

rgbn = stack_rasters(
    [
        array_to_raster(
            preprocess_optical(raster_to_array(folder + "raster/2021_B02_10m.tif")),
            reference=folder + "raster/2021_B02_10m.tif",
        ),
        array_to_raster(
            preprocess_optical(raster_to_array(folder + "raster/2021_B03_10m.tif")),
            reference=folder + "raster/2021_B03_10m.tif",
        ),
        array_to_raster(
            preprocess_optical(raster_to_array(folder + "raster/2021_B04_10m.tif")),
            reference=folder + "raster/2021_B04_10m.tif",
        ),
        array_to_raster(
            preprocess_optical(raster_to_array(folder + "raster/2021_B08_10m.tif")),
            reference=folder + "raster/2021_B08_10m.tif",
        ),
    ]
)

swir = stack_rasters(
    [
        array_to_raster(
            preprocess_optical(raster_to_array(folder + "raster/2021_B11_20m.tif")),
            reference=folder + "raster/2021_B11_20m.tif",
        ),
        array_to_raster(
            preprocess_optical(raster_to_array(folder + "raster/2021_B12_20m.tif")),
            reference=folder + "raster/2021_B12_20m.tif",
        ),
    ]
)

sar = stack_rasters(
    [
        array_to_raster(
            preprocess_sar(raster_to_array(folder + "raster/2021_VH.tif")),
            reference=folder + "raster/2021_VH.tif",
        ),
        array_to_raster(
            preprocess_sar(raster_to_array(folder + "raster/2021_VV.tif")),
            reference=folder + "raster/2021_VV.tif",
        ),
    ]
)


predict_raster(
    [rgbn, swir, sar],
    model,
    out_path=folder + "predicted_raster_06.tif",
    offsets=[
        [(32, 32), (64, 64), (96, 96)],
        [(16, 16), (32, 32), (48, 48)],
        [(32, 32), (64, 64), (96, 96)],
    ],
    device="gpu",
    output_size=128,
)
