yellow_follow = "C:/Users/caspe/Desktop/buteo/"
import sys
import os

from numpy.ma import array

sys.path.append(yellow_follow)

import numpy as np
from glob import glob
from osgeo import gdal
from buteo.raster.io import stack_rasters, raster_to_array
from buteo.machine_learning.patch_extraction_v2 import predict_raster
from buteo.vector.rasterize import rasterize_vector
from buteo.raster.clip import clip_raster
from buteo.raster.io import (
    raster_to_array,
    array_to_raster,
    stack_rasters_vrt,
)
from buteo.orfeo_toolbox import merge_rasters, obt_bandmath
from buteo.filters.convolutions import filter_array
from buteo.filters.kernel_generator import create_circle_kernel

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/ghana/predictions/"

area = folder + "Ghana_uint8_v9.tif"
volume = folder + "Ghana_volume_float32_v5.tif"

area_arr = raster_to_array(area)
# vol_arr = raster_to_array(volume)
# area_adj = (area_arr * 2).astype("float32")

# vol_arr_adj = np.where(area_adj > vol_arr, area_adj, vol_arr)

# array_to_raster(vol_arr_adj, reference=area, out_path=folder + "tmp_vol_adj_01.tif")

# vol_adj = folder + "tmp_vol_adj_01.tif"
# vol_adj_arr = raster_to_array(vol_adj)

# array_to_raster(
#     (area_arr > 0) * vol_adj_arr,
#     reference=vol_adj,
#     out_path=folder + "tmp_vol_adj_02.tif",
# )

# vol_adj2 = folder + "tmp_vol_adj_02.tif"
# vol_adj2_arr = raster_to_array(vol_adj2)

# array_to_raster(
#     vol_adj2_arr / (area_arr + 0.0000001),
#     reference=area,
#     out_path=folder + "tmp_height.tif",
# )

array_to_raster((area_arr / 100), reference=area, out_path=folder + "tmp_weights.tif")
array_to_raster(
    raster_to_array(folder + "tmp_weights.tif")
    * raster_to_array(folder + "tmp_height.tif"),
    reference=area,
    out_path=folder + "tmp_height_weighted.tif",
)

# raster = folder + "vector/population/m2_per_person_coarse_100m.tif"

# pops = glob(folder + "predictions/population*.tif")

# for img in pops:
#     out = folder + "/predictions/" + os.path.splitext(os.path.basename(img))[0] + "_clipped.tif"

#     clip_raster(
#         img,
#         folder + "vector/ghana_buffered_1k.gpkg",
#         out_path=out,
#         all_touch=False,
#         adjust_bbox=False,
#     )

# kernel = list(create_circle_kernel(25, 25))
# kernel[2] = kernel[2] / kernel[2].sum()
# kernel = tuple(kernel)

# array_to_raster(
#     filter_array(
#         raster_to_array(raster),
#         (25, 25, 1),
#         nodata=True,
#         nodata_value=-9999,
#         kernel=kernel,
#     ),
#     reference=raster,
#     out_path=folder + "vector/population/m2_per_person_smooth_100m.tif",
# )

# for zone in glob(folder + "vector/population/not_buffered/*.gpkg"):
#     name = os.path.splitext(os.path.basename(zone))[0] + "_not_buffered.tif"
#     rasterize_vector(
#         zone,
#         10,
#         out_path=folder + "vector/population/" + name,
#         extent=zone,
#         dtype="float32",
#         nodata_value=-9999.0,
#         attribute="m2_per_person2",
#     )

# merge_rasters(
#     glob(folder + "vector/population/*.tif"),
#     folder + "vector/population/smooth.tif",
#     out_datatype="float",
#     tmp=folder + "vector/population/tmp/",
#     nodata_value=-9999.0,
#     pixel_height=10.0,
#     pixel_width=10.0,
# )

# obt_bandmath(
#     [folder + "Ghana_classification_float32_v2.tif", folder + "Ghana_float32_v9.tif"],
#     "((im1b2 + im1b3 + im1b4) == 0) ? im2b1 : (im1b2 / (im1b2 + im1b3 + im1b4)) * im2b1",
#     folder + "area_residential.tif",
#     ram=32000,
# )

# obt_bandmath(
#     [folder + "Ghana_classification_float32_v2.tif", folder + "Ghana_float32_v9.tif"],
#     "((im1b2 + im1b3 + im1b4) == 0) ? 0.0 : (im1b3 / (im1b2 + im1b3 + im1b4)) * im2b1",
#     folder + "area_industrial.tif",
#     ram=32000,
# )

# obt_bandmath(
#     [folder + "Ghana_classification_float32_v2.tif", folder + "Ghana_float32_v9.tif"],
#     "((im1b2 + im1b3 + im1b4) == 0) ? 0.0 : (im1b4 / (im1b2 + im1b3 + im1b4)) * im2b1",
#     folder + "area_slum.tif",
#     ram=32000,
# )

# obt_bandmath(
#     [
#         folder + "area_residential_clipped.tif",
#         folder + "area_industrial_clipped.tif",
#         folder + "area_slum_clipped.tif",
#         folder + "m2_per_person_smooth.tif",
#     ],
#     "((im1b1 * 1.0) + (im2b1 * 1.0) + (im3b1 * 1.0)) / im4b1",  # unweighted
#     # "((im1b1 * 1.0238) + (im2b1 * 0.4) + (im3b1 * 1.3)) / im4b1", # afternoon
#     # "((im1b1 * 1.0405) + (im2b1 * 0.01) + (im3b1 * 1.3)) / im4b1", # nighttime
#     # "((im1b1 * 0.9575) + (im2b1 * 2.0) + (im3b1 * 1.1)) / im4b1",  # daytime
#     folder + "population_area_unweighted.tif",
#     ram=32000,
# )
