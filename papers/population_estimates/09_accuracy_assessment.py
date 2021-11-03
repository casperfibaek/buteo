import sys

sys.path.append("../../")
import numpy as np
from glob import glob
import os

np.set_printoptions(suppress=True)

from buteo.utils import progress
from buteo.raster.io import raster_to_array, array_to_raster
from buteo.raster.resample import resample_raster
from buteo.raster.reproject import reproject_raster
from buteo.raster.clip import clip_raster
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/ghana/aux_files/"

# for img in glob(folder + "gha_ppp*.tif"):
#     name = os.path.splitext(os.path.basename(img))[0]
#     arr = raster_to_array(img)
#     arr.mask = arr == -99999.0
#     arr = arr.filled(0)

#     output = folder + "adj_" + name + ".tif"

#     nullfix = array_to_raster(arr, img)

#     reproject_raster(nullfix, 32630, output)


# folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/ghana/vector/comparisons/resampled/"
# folder2 = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/ghana/aux_files/"

# base = folder2 + "ESA_WorldCover_builtup_aligned.tif"
# grids = glob(folder + "grid_id_*.tif")

# for idx, grid_cell in enumerate(grids):
#     progress(idx, len(grids), "Rasterizing")

#     name = os.path.basename(grid_cell)
#     number = os.path.splitext(name)[0].split("_")[-1]
#     clip_raster(base, grid_cell, folder + f"/ewc/ecw_{number}.tif")

# exit()

# for raster in glob(folder + "ghls2*.tif"):
#     name = os.path.splitext(os.path.basename(raster))[0]
#     resample_raster(
#         raster,
#         100,
#         out_path=folder + f"resampled/{name}.tif",
#         resample_alg="average",
#         postfix="",
#     )


def print_result(number, name, pred, truth):
    pred = pred.flatten()
    truth = truth.flatten()

    _mae = "{0:.3f}".format(round(mean_absolute_error(truth, pred), 3))
    _rmse = "{0:.3f}".format(round(mean_squared_error(truth, pred, squared=False), 3))
    _exvar = "{0:.3f}".format(round(explained_variance_score(truth, pred), 3))

    print(f"{number} - {name}. MAE: {_mae} - RMSE: {_rmse} - Exp.var: {_exvar}")


truth_arr = []
pred_arr = []
osm_arr = []
google_90_arr = []
google_85_arr = []
google_80_arr = []
google_00_arr = []
ghls2_arr = []
ecw_arr = []

for area in sorted(glob(folder + "ground_truth_*.tif")):
    name = os.path.splitext(os.path.basename(area))[0]
    number = name.split("_")[-1]

    pred = raster_to_array(f"{folder}prediction_{number}.tif")
    mask = pred == -9999.9
    pred.mask = mask
    pred = pred.filled(0)
    pred_arr.append(pred.flatten())

    truth = raster_to_array(area)
    truth.mask = mask
    truth = truth.filled(0)
    truth_arr.append(truth.flatten())

    osm = raster_to_array(f"{folder}OSM_{number}.tif")
    osm.mask = mask
    osm = osm.filled(0)
    osm_arr.append(osm.flatten())

    google_90 = raster_to_array(f"{folder}Google90_{number}.tif")
    google_90.mask = mask
    google_90 = google_90.filled(0)
    google_90_arr.append(google_90.flatten())

    google_85 = raster_to_array(f"{folder}Google85_{number}.tif")
    google_85.mask = mask
    google_85 = google_85.filled(0)
    google_85_arr.append(google_85.flatten())

    google_80 = raster_to_array(f"{folder}Google80_{number}.tif")
    google_80.mask = mask
    google_80 = google_80.filled(0)
    google_80_arr.append(google_80.flatten())

    google_00 = raster_to_array(f"{folder}Google00_{number}.tif")
    google_00.mask = mask
    google_00 = google_00.filled(0)
    google_00_arr.append(google_00.flatten())

    ghls2 = raster_to_array(f"{folder}ghls2_{number}.tif")
    ghls2.mask = mask
    ghls2 = ghls2.filled(0)
    ghls2_arr.append(ghls2.flatten())

    ecw = raster_to_array(f"{folder}ecw_{number}.tif")
    ecw.mask = mask
    ecw = ecw.filled(0)
    ecw_arr.append(ecw.flatten())

    # print_result(number, "Prediction", pred, truth)
    # print_result(number, "OSM       ", osm, truth)
    # print_result(number, "Google 90 ", google_90, truth)
    # print_result(number, "Google 85 ", google_85, truth)
    # print_result(number, "Google 80 ", google_80, truth)
    # print_result(number, "Google All", google_00, truth)
    # print_result(number, "GHSL S2   ", ghls2, truth)
    # print_result(number, "ECW       ", ecw, truth)

truth_c = np.concatenate(truth_arr)

print_result("100", "Prediction", np.concatenate(pred_arr), truth_c)
print_result("100", "OSM       ", np.concatenate(osm_arr), truth_c)
print_result("100", "Google 90 ", np.concatenate(google_90_arr), truth_c)
print_result("100", "Google 85 ", np.concatenate(google_85_arr), truth_c)
print_result("100", "Google 80 ", np.concatenate(google_80_arr), truth_c)
print_result("100", "Google All", np.concatenate(google_00_arr), truth_c)
# print_result("100", "GHSL S2   ", np.concatenate(ghls2_arr), truth_c)
# print_result("100", "ECW       ", np.concatenate(ecw_arr), truth_c)

bin_truth = np.array(truth_c > 0.5, dtype="uint8")
bin_pred = np.array(np.concatenate(pred_arr) > 0.5, dtype="uint8")
bin_osm = np.array(np.concatenate(osm_arr) > 0.5, dtype="uint8")
bin_google90 = np.array(np.concatenate(google_90_arr) > 0.5, dtype="uint8")
bin_google85 = np.array(np.concatenate(google_85_arr) > 0.5, dtype="uint8")
bin_google80 = np.array(np.concatenate(google_80_arr) > 0.5, dtype="uint8")
bin_google00 = np.array(np.concatenate(google_00_arr) > 0.5, dtype="uint8")
bin_ghls2 = np.array(np.concatenate(ghls2_arr) > 0, dtype="uint8")
bin_ghls2_50 = np.array(np.concatenate(ghls2_arr) > 50, dtype="uint8")
bin_ecw = np.array(np.concatenate(ecw_arr) > 0, dtype="uint8")


def print_result_bin(name, _truth, _pred):
    bacc = round(balanced_accuracy_score(_truth, _pred), 3)
    acc = round(accuracy_score(_truth, _pred), 3)
    f1 = round(f1_score(_truth, _pred), 3)
    precision = round(precision_score(_truth, _pred), 3)
    recall = round(recall_score(_truth, _pred), 3)
    print(
        f"{name} Binary || Bal. Accuracy: {bacc} - Accuracy: {acc} - F1: {f1} - Precision: {precision} - Recall: {recall}"
    )


print_result_bin("predition ", bin_truth, bin_pred)
print_result_bin("OSM       ", bin_truth, bin_osm)
print_result_bin("Google 90 ", bin_truth, bin_google90)
print_result_bin("Google 85 ", bin_truth, bin_google85)
print_result_bin("Google 80 ", bin_truth, bin_google80)
print_result_bin("Google All", bin_truth, bin_google00)
print_result_bin("GHLS2 > 0 ", bin_truth, bin_ghls2)
print_result_bin("GHLS2 > 50", bin_truth, bin_ghls2_50)
print_result_bin("ECW       ", bin_truth, bin_ecw)
