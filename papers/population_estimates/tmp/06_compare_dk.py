yellow_follow = "C:/Users/caspe/Desktop/buteo/"
import sys, os
from token import DOUBLESTAR

sys.path.append(yellow_follow)

from buteo.raster.io import raster_to_array, array_to_raster
from buteo.raster.clip import clip_raster
from glob import glob

import numpy as np

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/"

pred_path = folder + "bornholm/predictions/big_model_dk_08_2020.tif"
truth_path = folder + "bornholm/raster/2020_label_area.tif"

truth = raster_to_array(truth_path)
pred = np.clip(raster_to_array(pred_path), 0, 100)


sum_dif = ((np.sum(pred) - np.sum(truth)) / np.sum(truth)) * 100
mae = np.mean(np.abs(pred - truth))
mse = np.mean(np.power(pred - truth, 2))
binary = (((np.rint(pred) == 0) == (np.rint(truth) == 0)).sum() / truth.size) * 100

print("")
print("TSUM: " + str(np.sum(truth)))
print("PSUM: " + str(np.sum(pred)))
print(f"truth_size: " + str(len(truth)))
print(f"pred_size:  " + str(len(pred)))
print("MAE: " + str(mae))
print("MSE: " + str(mse))
print("SUM: " + str(sum_dif))
print("BIN: " + str(round(binary, 4)))
print("")

# ("aarhus_area@1" > 0 AND "aarhus_area_32x32_9_overlaps@1" > 0) *
# ((("aarhus_area_32x32_9_overlaps@1" - "aarhus_area@1") + 0.00000001) / ("aarhus_area@1" + 0.00000001)) + ("aarhus_area_32x32_9_overlaps@1" <  0.00000001 and "aarhus_area@1" >  0.00000001) * "aarhus_area@1" * -1
