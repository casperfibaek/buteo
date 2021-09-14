import sys

sys.path.append("../../")

from buteo.raster.align import align_rasters
from buteo.raster.clip import clip_raster
from buteo.raster.nodata import raster_set_nodata

# from buteo.raster.io import raster_to_array, array_to_raster
from glob import glob

# folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/ghana/raster/"
folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/tanzania_mwanza/"
aligned = folder + "aligned/"

target = folder + "august_B12_20m.jp2"

align_rasters(
    clip_raster(
        glob(folder + "*10m.*"),
        clip_geom=target,
        postfix="",
        all_touch=False,
        adjust_bbox=False,
        dst_nodata=0,
    ),
    aligned,
    postfix="",
    dst_nodata=False,
    ram="80%",
    bounding_box=target,
)

align_rasters(
    clip_raster(
        glob(folder + "*20m.*"),
        clip_geom=target,
        postfix="",
        all_touch=False,
        adjust_bbox=False,
    ),
    aligned,
    postfix="",
    ram="80%",
    dst_nodata=False,
    bounding_box=target,
)
