yellow_follow = "C:/Users/caspe/Desktop/buteo/"
import sys, os
from token import DOUBLESTAR

sys.path.append(yellow_follow)

from buteo.raster.io import raster_to_array, array_to_raster
from buteo.raster.clip import clip_raster
from glob import glob

import numpy as np

folder = "C:/Users/caspe/Desktop/paper_2_Structural_Volume/data/"

predictions = folder + "predictions/"

target = "area"
truth_aarhus = raster_to_array(folder + f"raster/aarhus/label_{target}.tif")
truth_holsterbro = raster_to_array(folder + f"raster/holsterbro/label_{target}.tif")
# truth_holsterbro = raster_to_array(folder + f"raster/holsterbro/label_{target}.tif")[
#     1:
# ][:][:]
truth_samsoe = raster_to_array(folder + f"raster/samsoe/label_{target}.tif")

for pred_path in glob(predictions + f"big_model_dk_08_*.tif"):
    name = os.path.splitext(os.path.basename(pred_path))[0]
    pred = raster_to_array(pred_path)

    if target == "people":
        pred = pred / 100

    if "aarhus" in name:
        truth = truth_aarhus
    elif "holsterbro" in name:
        truth = truth_holsterbro
        # pred = pred[1:][:][:]  # due the nodata row on top of raster.
    elif "samsoe" in name:
        truth = truth_samsoe

    sum_dif = ((np.sum(pred) - np.sum(truth)) / np.sum(truth)) * 100
    mae = np.mean(np.abs(pred - truth))
    mse = np.mean(np.power(pred - truth, 2))

    print(name)
    print("TSUM: " + str(np.sum(truth)))
    print("PSUM: " + str(np.sum(pred)))
    print("MAE:  " + str(mae))
    print("MSE:  " + str(mse))
    print("SUM:  " + str(sum_dif))
    print("")

# ("aarhus_area@1" > 0 AND "aarhus_area_32x32_9_overlaps@1" > 0) *
# ((("aarhus_area_32x32_9_overlaps@1" - "aarhus_area@1") + 0.00000001) / ("aarhus_area@1" + 0.00000001)) + ("aarhus_area_32x32_9_overlaps@1" <  0.00000001 and "aarhus_area@1" >  0.00000001) * "aarhus_area@1" * -1
