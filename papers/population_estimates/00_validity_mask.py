import sys, os
from glob import glob

from numpy import dtype

sys.path.append("../../")
from buteo.vector.rasterize import rasterize_vector
from buteo.raster.io import raster_to_array, array_to_raster
from buteo.raster.clip import clip_raster
from buteo.raster.resample import resample_raster


def create_valid_mask(vector_mask, b4, output):
    rasterized = rasterize_vector(vector_mask, 10.0, extent=b4)
    return array_to_raster(
        raster_to_array(
            clip_raster(
                rasterized, clip_geom=b4, postfix="", all_touch=False, adjust_bbox=False
            ),
        ).astype("uint8"),
        reference=b4,
        out_path=output,
    )


def create_ground_truth(buildings, b4, output, size=0.4):
    cm40 = rasterize_vector(buildings, size, extent=b4)
    resampled = resample_raster(cm40, b4, resample_alg="average", dtype="float32")
    return array_to_raster(
        raster_to_array(resampled) * 100, reference=b4, out_path=output
    )


folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/tanzania_kigoma/"
# create_valid_mask(
#     folder + "vector/kigoma_valid_mask.gpkg",
#     folder + "B04_10m.tif",
#     folder + "validation_mask.tif",
# )

create_ground_truth(
    folder + "vector/kigoma_buildings.gpkg",
    folder + "B04_10m.tif",
    folder + "label_area.tif",
    size=0.5,
)
