yellow_follow = "C:/Users/caspe/Desktop/buteo/"
import sys, os

sys.path.append(yellow_follow)

from buteo.raster.io import raster_to_array, array_to_raster
from buteo.raster.clip import clip_raster
from glob import glob

import numpy as np

folder = "C:/Users/caspe/Desktop/paper_2_Structural_Volume/data/"

predictions = folder + "predictions/"

truth_aarhus = raster_to_array(folder + "raster/aarhus/label_area.tif")
truth_holsterbro = raster_to_array(folder + "raster/holsterbro/label_area.tif")[1:][:][
    :
]
truth_samsoe = raster_to_array(folder + "raster/samsoe/label_area.tif")

for pred in glob(predictions + "*.tif"):
    name = os.path.splitext(os.path.basename(pred))[0]
    test = raster_to_array(pred)

    if "aarhus" in name:
        truth = truth_aarhus
    elif "holsterbro" in name:
        truth = truth_holsterbro
        test = test[1:][:][:]  # due the nodata row on top of raster.
    elif "samsoe" in name:
        truth = truth_samsoe

    mae = np.mean(np.abs(test - truth))
    mse = np.mean(np.power(test - truth, 2))

    dif = np.nanmean((test / truth))

    print(name)
    print("MAE: " + str(mae))
    print("MSE: " + str(mse))
    print("DIF: " + str(dif))
    print("")
