import sys; sys.path.append("../")


import os
import numpy as np
import buteo as beo

FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s2_data/S2A_MSIL2A_20220107T090351_N0301_R007_T36UVB_20220107T120342.SAFE/GRANULE/L2A_T36UVB_A034181_20220107T090345/IMG_DATA/R10m/"

jp2_file = os.path.join(FOLDER, "T36UVB_20220107T090351_TCI_10m.jp2")

jp2_arr = beo.raster_to_array(jp2_file, pixel_offsets=[500, 500, 1000, 1000])
print(jp2_arr.max())

jp2_arr_f32 = beo.raster_to_array(jp2_file, pixel_offsets=[500, 500, 1000, 1000], cast=np.float32)
print(jp2_arr_f32.max())
