yellow_follow = "C:/Users/caspe/Desktop/buteo/"
import sys
import os

sys.path.append(yellow_follow)

from glob import glob
from buteo.raster.io import stack_rasters
from buteo.machine_learning.patch_extraction import predict_raster
from buteo.raster.io import raster_to_array, array_to_raster
from buteo.raster.clip import internal_clip_raster
from utils import preprocess_optical, preprocess_sar

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/analysis/ghana/"
vector_folder = (
    "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/analysis/ghana/vector/regions/"
)
raster_folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/ghana/raster/"
model = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/analysis/ghana/vector/grid_cells/patches/merged/models/ghana_seed_01_11_best"


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


# for region in glob(vector_folder + "id_*.gpkg"):
#     region_name = os.path.splitext(os.path.basename(region))[0]

#     if region_name in ["id_1", "id_10", "id_11", "id_12", "id_13"]:
#         continue

#     print(f"Processing region: {region_name}")

#     print("Clipping SWIR.")
#     B11_clipped = internal_clip_raster(
#         raster_folder + "2021_B11_20m.tif", clip_geom=region
#     )
#     B12_clipped = internal_clip_raster(
#         raster_folder + "2021_B12_20m.tif", clip_geom=region
#     )

#     print("Stacking SWIR.")
#     swir = stack_rasters(
#         [
#             array_to_raster(
#                 preprocess_optical(raster_to_array(B11_clipped)),
#                 reference=B11_clipped,
#             ),
#             array_to_raster(
#                 preprocess_optical(raster_to_array(B12_clipped)),
#                 reference=B12_clipped,
#             ),
#         ],
#         dtype="float32",
#     )

#     print("Clipping RGBN.")
#     B02_clipped = internal_clip_raster(
#         raster_folder + "2021_B02_10m.tif", clip_geom=B11_clipped
#     )
#     B03_clipped = internal_clip_raster(
#         raster_folder + "2021_B03_10m.tif", clip_geom=B11_clipped
#     )
#     B04_clipped = internal_clip_raster(
#         raster_folder + "2021_B04_10m.tif", clip_geom=B11_clipped
#     )
#     B08_clipped = internal_clip_raster(
#         raster_folder + "2021_B08_10m.tif", clip_geom=B11_clipped
#     )

#     print("Stacking RGBN.")
#     rgbn = stack_rasters(
#         [
#             array_to_raster(
#                 preprocess_optical(raster_to_array(B02_clipped)),
#                 reference=B02_clipped,
#             ),
#             array_to_raster(
#                 preprocess_optical(raster_to_array(B03_clipped)),
#                 reference=B03_clipped,
#             ),
#             array_to_raster(
#                 preprocess_optical(raster_to_array(B04_clipped)),
#                 reference=B04_clipped,
#             ),
#             array_to_raster(
#                 preprocess_optical(raster_to_array(B08_clipped)),
#                 reference=B08_clipped,
#             ),
#         ],
#         dtype="float32",
#     )

#     B02_clipped = None
#     B03_clipped = None
#     B04_clipped = None
#     B08_clipped = None

#     print("Clipping SAR.")
#     vh_clipped = internal_clip_raster(
#         raster_folder + "2021_VH_10m.tif", clip_geom=B11_clipped
#     )
#     vv_clipped = internal_clip_raster(
#         raster_folder + "2021_VV_10m.tif", clip_geom=B11_clipped
#     )

#     print("Stacking SAR.")
#     sar = stack_rasters(
#         [
#             array_to_raster(
#                 preprocess_sar(raster_to_array(vh_clipped)),
#                 reference=vh_clipped,
#             ),
#             array_to_raster(
#                 preprocess_sar(raster_to_array(vv_clipped)),
#                 reference=vv_clipped,
#             ),
#         ],
#         dtype="float32",
#     )

#     B11_clipped = None
#     B12_clipped = None
#     vh_clipped = None
#     vv_clipped = None

#     print("Ready for predictions.")
#     predict_model(model, os.path.splitext(os.path.basename(region))[0])

import numpy as np

mosaic = folder + "predictions/Ghana_mosaic.tif"

rounded = array_to_raster(
    np.clip(np.rint(raster_to_array(mosaic)), 0, 100).astype("uint8"), mosaic
)
internal_clip_raster(
    rounded,
    folder + "vector/ghana_buffered_1280.gpkg",
    out_path=folder + "predictions/Ghana_uint8.tif",
    dst_nodata=255,
)

# post processing:
# np.clip(np.rint(img), 0, 100).astype("uint8")
# set nodata to -1
