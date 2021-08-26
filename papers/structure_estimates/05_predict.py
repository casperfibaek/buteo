yellow_follow = "C:/Users/caspe/Desktop/buteo/"
import sys
import os

sys.path.append(yellow_follow)

from buteo.raster.io import stack_rasters
from buteo.machine_learning.patch_extraction import predict_raster
from buteo.machine_learning.ml_utils import (
    preprocess_optical,
    preprocess_sar,
    preprocess_coh,
)
from buteo.raster.io import raster_to_array, array_to_raster


def get_offsets(size):
    high_mid = size // 2
    high_low = high_mid // 2
    high_high = high_mid + high_low

    low_low = high_low - (high_low // 2)
    low_mid = high_low
    low_high = low_mid + low_low

    high_row = [
        (0, high_low),
        (0, high_mid),
        (0, high_high),
        (high_low, high_low),
        (high_mid, high_mid),
        (high_high, high_high),
        (high_low, 0),
        (high_mid, 0),
        (high_high, 0),
    ]

    low_row = [
        (0, low_low),
        (0, low_mid),
        (0, low_high),
        (low_low, low_low),
        (low_mid, low_mid),
        (low_high, low_high),
        (low_low, 0),
        (low_mid, 0),
        (low_high, 0),
    ]

    return [
        high_row,
        high_row,
        low_row,
    ]


# def get_offsets(size):
#     return [
#         [],
#         [],
#         [],
#     ]


def predict_model(
    model,
    name,
    folder,
    target,
    output_size,
    offsets=[],
    target_low=0,
    target_high=1,
    optical_top=6000,
):

    higher_01 = stack_rasters(
        [
            array_to_raster(
                preprocess_optical(
                    raster_to_array(f"{folder}raster/{target}/B02.tif"),
                    target_low=target_low,
                    target_high=target_high,
                    cutoff_high=optical_top,
                ),
                reference=f"{folder}raster/{target}/B02.tif",
            ),
            array_to_raster(
                preprocess_optical(
                    raster_to_array(f"{folder}raster/{target}/B03.tif"),
                    target_low=target_low,
                    target_high=target_high,
                    cutoff_high=optical_top,
                ),
                reference=f"{folder}raster/{target}/B03.tif",
            ),
            array_to_raster(
                preprocess_optical(
                    raster_to_array(f"{folder}raster/{target}/B04.tif"),
                    target_low=target_low,
                    target_high=target_high,
                    cutoff_high=optical_top,
                ),
                reference=f"{folder}raster/{target}/B04.tif",
            ),
            array_to_raster(
                preprocess_optical(
                    raster_to_array(f"{folder}raster/{target}/B08.tif"),
                    target_low=target_low,
                    target_high=target_high,
                    cutoff_high=optical_top,
                ),
                reference=f"{folder}raster/{target}/B08.tif",
            ),
        ]
    )

    lower = stack_rasters(
        [
            array_to_raster(
                preprocess_optical(
                    raster_to_array(f"{folder}raster/{target}/B05.tif"),
                    target_low=target_low,
                    target_high=target_high,
                    cutoff_high=optical_top,
                ),
                reference=f"{folder}raster/{target}/B05.tif",
            ),
            array_to_raster(
                preprocess_optical(
                    raster_to_array(f"{folder}raster/{target}/B06.tif"),
                    target_low=target_low,
                    target_high=target_high,
                    cutoff_high=optical_top,
                ),
                reference=f"{folder}raster/{target}/B06.tif",
            ),
            array_to_raster(
                preprocess_optical(
                    raster_to_array(f"{folder}raster/{target}/B07.tif"),
                    target_low=target_low,
                    target_high=target_high,
                    cutoff_high=optical_top,
                ),
                reference=f"{folder}raster/{target}/B07.tif",
            ),
            array_to_raster(
                preprocess_optical(
                    raster_to_array(f"{folder}raster/{target}/B11.tif"),
                    target_low=target_low,
                    target_high=target_high,
                    cutoff_high=optical_top,
                ),
                reference=f"{folder}raster/{target}/B11.tif",
            ),
            array_to_raster(
                preprocess_optical(
                    raster_to_array(f"{folder}raster/{target}/B12.tif"),
                    target_low=target_low,
                    target_high=target_high,
                    cutoff_high=optical_top,
                ),
                reference=f"{folder}raster/{target}/B12.tif",
            ),
        ]
    )

    higher_02 = stack_rasters(
        [
            array_to_raster(
                preprocess_sar(
                    raster_to_array(f"{folder}raster/{target}/VV_asc.tif"),
                    target_low=target_low,
                    target_high=target_high,
                ),
                reference=f"{folder}raster/{target}/VV_asc.tif",
            ),
            array_to_raster(
                preprocess_sar(
                    raster_to_array(f"{folder}raster/{target}/VV_desc.tif"),
                    target_low=target_low,
                    target_high=target_high,
                ),
                reference=f"{folder}raster/{target}/VV_desc.tif",
            ),
            array_to_raster(
                preprocess_coh(
                    raster_to_array(f"{folder}raster/{target}/COH_asc.tif"),
                    target_low=target_low,
                    target_high=target_high,
                ),
                reference=f"{folder}raster/{target}/COH_asc.tif",
            ),
            array_to_raster(
                preprocess_coh(
                    raster_to_array(f"{folder}raster/{target}/COH_desc.tif"),
                    target_low=target_low,
                    target_high=target_high,
                ),
                reference=f"{folder}raster/{target}/COH_desc.tif",
            ),
        ]
    )

    predict_raster(
        [higher_01, higher_02, lower],
        model,
        out_path=folder + f"predictions/{name}.tif",
        offsets=offsets,
        device="gpu",
        output_size=output_size,
        batch_size=128,
    )


folder = "C:/Users/caspe/Desktop/paper_2_Structural_Volume/data/"

for target_area in ["holsterbro", "aarhus", "samsoe"]:
    predict_model(
        folder + "models/128_area",
        f"{target_area}_area_128x128_9_overlaps",
        folder,
        target_area,
        128,
        offsets=get_offsets(128),
        target_low=0,
        target_high=1,
        optical_top=6000,
    )

    predict_model(
        folder + "models/64_area",
        f"{target_area}_area_64x64_9_overlaps",
        folder,
        target_area,
        64,
        offsets=get_offsets(64),
        target_low=0,
        target_high=1,
        optical_top=6000,
    )

    predict_model(
        folder + "models/32_area",
        f"{target_area}_area_32x32_9_overlaps",
        folder,
        target_area,
        32,
        offsets=get_offsets(32),
        target_low=0,
        target_high=1,
        optical_top=6000,
    )

    predict_model(
        folder + "models/16_area",
        f"{target_area}_area_16x16_9_overlaps",
        folder,
        target_area,
        16,
        offsets=get_offsets(16),
        target_low=0,
        target_high=1,
        optical_top=6000,
    )
