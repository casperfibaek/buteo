yellow_follow = "C:/Users/caspe/Desktop/buteo/"
import sys

sys.path.append(yellow_follow)

from buteo.filters.convolutions import filter_array
from buteo.filters.kernel_generator import create_circle_kernel
from buteo.raster.io import raster_to_array, array_to_raster

folder = (
    "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/analysis/denmark/raster/people/"
)

people_raster = folder + "fid_1_rasterized.tif"
people_arr = raster_to_array(people_raster)

summed = filter_array(
    people_arr,
    # (21, 21),
    (10, 10),
    # normalised=False,
    # distance_calc=False,
    # radius_method="2d",
    operation="sum",
    kernel=create_circle_kernel(10, 10),
)
# array_to_raster(people_arr, people_raster, folder + "people_summed_100m.tif")
array_to_raster(summed, people_raster, folder + "people_within_100m_radius_v2.tif")

# 1 / create_circle_kernel(5, 5)[2].sum()
# 0.012773023374632774
