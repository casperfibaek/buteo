""" Bug-fixing script """

# pylint: skip-file

# Standard library
import sys; sys.path.append("../")
import os
from glob import glob

# External
import numpy as np
from osgeo import gdal

# Internal
from buteo import raster_to_array, array_to_raster, raster_to_metadata
from buteo.utils import split_into_offsets


folder_raster = r"C:/Users/casper.fibaek/OneDrive - ESA/Desktop/Unicef/"
folder_dst = r"C:/Users/casper.fibaek/OneDrive - ESA\Desktop/Unicef/nightlights_2021/trend/parts/"

nl_path = os.path.join(folder_raster, "VNL_v21_npp_2014-2021_global_vcmslcfg_202205302300_median_54009_500m.tif")
metadata = raster_to_metadata(nl_path)
nl_shape = metadata["shape"]

offsets = split_into_offsets(nl_shape, 5, 3)

for idx, offset in enumerate(offsets):
    print(f"Processing offset {idx + 1} of {len(offsets)}")
    
    name = f"offset_{idx + 1}.tif"
    out_path = os.path.join(folder_dst, name)

    arr = raster_to_array(nl_path, pixel_offsets=offset)[:, :, 0]
    array_to_raster(arr, reference=nl_path, out_path=out_path, pixel_offsets=offset)
