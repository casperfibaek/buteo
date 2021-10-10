import sys

sys.path.append("../../")

from buteo.orfeo_toolbox import merge_rasters
from glob import glob
import os

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/"

raster_folder = folder + "ghana/raster/"
dst_folder = folder + "ghana/raster_v2/"
tmp_folder = dst_folder + "tmp/"

accra = folder + "accra/"
lake = folder + "lake/"
kumasi = folder + "kumasi/"

for raster in glob(raster_folder + "2021_B*.tif"):
    name = os.path.splitext(os.path.basename(raster))[0]
    name = name.split("_")[1] + "_" + name.split("_")[2]

    if os.path.exists(dst_folder + "2021_" + name + ".tif"):
        continue

    print(f"Processing: {name}")

    og_path = raster
    accra_path = accra + name + ".tif"
    lake_path = lake + name + ".tif"
    kumasi_path = kumasi + name + ".tif"

    merge_rasters(
        [og_path, accra_path, lake_path, kumasi_path],
        dst_folder + "2021_" + name + ".tif",
        tmp=tmp_folder,
        harmonisation=True,
        nodata_value=None,
    )
