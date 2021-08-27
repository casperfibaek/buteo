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

target = "people"
truth_aarhus = raster_to_array(folder + f"raster/aarhus/label_{target}.tif")
truth_holsterbro = raster_to_array(folder + f"raster/holsterbro/label_{target}.tif")[
    1:
][:][:]
truth_samsoe = raster_to_array(folder + f"raster/samsoe/label_{target}.tif")

for pred in glob(predictions + "*big_model_people_v2_01*.tif"):
    name = os.path.splitext(os.path.basename(pred))[0]
    test = raster_to_array(pred)

    if target == "people":
        test = test / 100

    if "aarhus" in name:
        truth = truth_aarhus
    elif "holsterbro" in name:
        truth = truth_holsterbro
        test = test[1:][:][:]  # due the nodata row on top of raster.
    elif "samsoe" in name:
        truth = truth_samsoe

    # mask = np.logical_or(
    #     np.sum(truth > 0, axis=(1, 2)),
    #     np.sum(test > 0, axis=(1, 2)),
    # )
    # truth = truth[mask]
    # test = test[mask]

    # import pdb

    # pdb.set_trace()

    mae = np.mean(np.abs(test - truth))
    mse = np.mean(np.power(test - truth, 2))

    dif = np.nanmean((test / truth))

    sum_dif = np.sum(test) - np.sum(truth)
    total_difference = (sum_dif / np.sum(truth)) * 100

    # print(name)
    # print("MAE: " + str(mae))
    # print("MSE: " + str(mse))
    # print("DIF: " + str(dif))
    print("Difference: " + str(round(total_difference, 3)) + "%")
    # print("")

# ("aarhus_area@1" > 0 AND "aarhus_area_32x32_9_overlaps@1" > 0) *
# ((("aarhus_area_32x32_9_overlaps@1" - "aarhus_area@1") + 0.00000001) / ("aarhus_area@1" + 0.00000001)) + ("aarhus_area_32x32_9_overlaps@1" <  0.00000001 and "aarhus_area@1" >  0.00000001) * "aarhus_area@1" * -1
