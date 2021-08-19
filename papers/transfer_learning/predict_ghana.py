yellow_follow = "C:/Users/caspe/Desktop/buteo/"
import sys
import os

sys.path.append(yellow_follow)

import numpy as np
from glob import glob
from buteo.raster.io import stack_rasters, raster_to_array
from buteo.machine_learning.patch_extraction import predict_raster
from buteo.raster.io import raster_to_array, array_to_raster, stack_rasters_vrt
from buteo.raster.clip import internal_clip_raster
from utils import preprocess_optical, preprocess_sar

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/analysis/ghana/"
vector_folder = (
    "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/analysis/ghana/vector/regions/"
)
raster_folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/ghana/raster/"
model = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/analysis/ghana/vector/grid_cells4/patches/merged/models/ghana_seed_06_54_best"

# NOTE: all_touch and adjust_bbox needs to be false for region 10.

for region in glob(vector_folder + "id_*.gpkg"):
    region_name = os.path.splitext(os.path.basename(region))[0]

    # completed
    if region_name in ["id_10"]:
        continue

    print(f"Processing region: {region_name}")

    print("Clipping SWIR.")
    B11_clipped = internal_clip_raster(
        raster_folder + "2021_B11_20m.tif",
        clip_geom=region,
        # adjust_bbox=False,
        # all_touch=False,
    )
    B12_clipped = internal_clip_raster(
        raster_folder + "2021_B12_20m.tif",
        clip_geom=region,
        # adjust_bbox=False,
        # all_touch=False,
    )

    print("Stacking SWIR.")
    swir = stack_rasters(
        [
            array_to_raster(
                preprocess_optical(raster_to_array(B11_clipped)),
                reference=B11_clipped,
            ),
            array_to_raster(
                preprocess_optical(raster_to_array(B12_clipped)),
                reference=B12_clipped,
            ),
        ],
        dtype="float32",
    )

    print("Clipping RGBN.")
    B02_clipped = internal_clip_raster(
        raster_folder + "2021_B02_10m.tif",
        clip_geom=B11_clipped,
        # adjust_bbox=False,
        # all_touch=False,
    )
    B03_clipped = internal_clip_raster(
        raster_folder + "2021_B03_10m.tif",
        clip_geom=B11_clipped,
        # adjust_bbox=False,
        # all_touch=False,
    )
    B04_clipped = internal_clip_raster(
        raster_folder + "2021_B04_10m.tif",
        clip_geom=B11_clipped,
        # adjust_bbox=False,
        # all_touch=False,
    )
    B08_clipped = internal_clip_raster(
        raster_folder + "2021_B08_10m.tif",
        clip_geom=B11_clipped,
        # adjust_bbox=False,
        # all_touch=False,
    )

    print("Stacking RGBN.")
    rgbn = stack_rasters(
        [
            array_to_raster(
                preprocess_optical(raster_to_array(B02_clipped)),
                reference=B02_clipped,
            ),
            array_to_raster(
                preprocess_optical(raster_to_array(B03_clipped)),
                reference=B03_clipped,
            ),
            array_to_raster(
                preprocess_optical(raster_to_array(B04_clipped)),
                reference=B04_clipped,
            ),
            array_to_raster(
                preprocess_optical(raster_to_array(B08_clipped)),
                reference=B08_clipped,
            ),
        ],
        dtype="float32",
    )

    B02_clipped = None
    B03_clipped = None
    B04_clipped = None
    B08_clipped = None

    print("Clipping SAR.")
    vh_clipped = internal_clip_raster(
        raster_folder + "2021_VH_10m.tif",
        clip_geom=B11_clipped,
        # adjust_bbox=False,
        # all_touch=False,
    )
    vv_clipped = internal_clip_raster(
        raster_folder + "2021_VV_10m.tif",
        clip_geom=B11_clipped,
        # adjust_bbox=False,
        # all_touch=False,
    )

    print("Stacking SAR.")
    sar = stack_rasters(
        [
            array_to_raster(
                preprocess_sar(raster_to_array(vh_clipped)),
                reference=vh_clipped,
            ),
            array_to_raster(
                preprocess_sar(raster_to_array(vv_clipped)),
                reference=vv_clipped,
            ),
        ],
        dtype="float32",
    )

    B11_clipped = None
    B12_clipped = None
    vh_clipped = None
    vv_clipped = None

    print("Ready for predictions.")

    outname = os.path.splitext(os.path.basename(region))[0]

    predict_raster(
        [rgbn, swir, sar],
        model,
        target_raster=B04_clipped,
        out_path=folder + f"predictions/{outname}.tif",
        offsets=[
            [(16, 16), (32, 32), (48, 48)],
            [(8, 8), (16, 16), (24, 24)],
            [(16, 16), (32, 32), (48, 48)],
        ],
        device="gpu",
        batch_size=64,
        output_size=64,
        # merge_method="average",
    )

# mosaic = stack_rasters_vrt(
#     glob(folder + f"predictions/*.tif"), "/vsimem/vrt_predictions.vrt", seperate=False
# )
# mosaic = "/vsimem/vrt_predictions.vrt"

# rounded = array_to_raster(
#     np.clip(np.rint(raster_to_array(mosaic)), 0, 100).astype("uint8"), mosaic
# )
# internal_clip_raster(
#     rounded,
#     folder + "vector/ghana_buffered_1k.gpkg",
#     out_path=folder + "predictions/Ghana_uint8_v3.tif",
#     dst_nodata=255,
# )
