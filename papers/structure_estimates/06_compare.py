yellow_follow = "C:/Users/caspe/Desktop/buteo/"
import sys, os

sys.path.append(yellow_follow)

from buteo.raster.io import array_to_raster, raster_to_array, raster_to_metadata
from buteo.raster.resample import resample_raster
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
)

import numpy as np


def round_to_decimals(x):
    return f"{np.round(x, 4):.4f}"


def metrics(truth, pred, name, resample=False, target=None):
    if not isinstance(truth, list):
        truth = [truth]
    if not isinstance(pred, list):
        pred = [pred]

    if len(truth) != len(pred):
        raise ValueError("Length of truth and pred must be equal")

    processed_truth = []
    processed_pred = []

    for idx in range(len(truth)):
        if (
            raster_to_metadata(truth[idx])["size"]
            != raster_to_metadata(pred[idx])["size"]
        ):
            print(f"{name} rasters are not the same size")
            return

        if resample:
            truth[idx] = resample_raster(truth[idx], 100, resample_alg="sum")
            pred[idx] = resample_raster(pred[idx], 100, resample_alg="sum")

        arr_truth = raster_to_array(truth[idx])
        arr_pred = raster_to_array(pred[idx])

        mask = np.logical_or(arr_truth == -9999.0, arr_pred == -9999.0)

        arr_truth.mask = mask
        arr_pred.mask = mask

        arr_truth = arr_truth.compressed()
        arr_pred = arr_pred.compressed()

        processed_truth.append(arr_truth.ravel())
        processed_pred.append(arr_pred.ravel())

    tarr = np.concatenate(processed_truth)
    tarr = tarr.ravel()
    parr = np.concatenate(processed_pred)
    parr = parr.ravel()

    mae = round_to_decimals(mean_absolute_error(tarr, parr))
    mse = round_to_decimals(mean_squared_error(tarr, parr))
    tpe = round_to_decimals(((np.sum(parr) - np.sum(tarr)) / np.sum(tarr)) * 100)

    if target == "people":
        tarr = np.array(tarr > 0.01, dtype=np.uint8)
        parr = np.array(parr > 0.01, dtype=np.uint8)
    else:
        tarr = np.array(tarr >= 1.0, dtype=np.uint8)
        parr = np.array(parr >= 1.0, dtype=np.uint8)

    acc = round_to_decimals(accuracy_score(tarr, parr))
    bacc = round_to_decimals(balanced_accuracy_score(tarr, parr))
    prec = round_to_decimals(precision_score(tarr, parr))
    rec = round_to_decimals(recall_score(tarr, parr))
    f1 = round_to_decimals(f1_score(tarr, parr))

    adjust_name = name.ljust(10, " ")

    print(f"{adjust_name} (reg) - MAE: {mae}, MSE: {mse}, TPE: {tpe}")
    print(
        f"{adjust_name} (bin) - ACC: {acc}, BACC: {bacc}, PREC: {prec}, REC: {rec}, F1: {f1}"
    )


base = "C:/Users/caspe/Desktop/paper_2_Structural_Volume/data/"
folder = base + "predictions/"

target = "people"
resample = True

truth_aarhus = folder + f"aarhus_label_{target}.tif"
truth_holsterbro = folder + f"holsterbro_label_{target}.tif"
truth_samsoe = folder + f"samsoe_label_{target}.tif"

pred_aarhus = folder + f"aarhus_prediction_{target}.tif"
pred_holsterbro = folder + f"holsterbro_prediction_{target}.tif"
pred_samsoe = folder + f"samsoe_prediction_{target}.tif"

metrics(truth_aarhus, pred_aarhus, "Aarhus", resample=resample, target=target)
metrics(
    truth_holsterbro, pred_holsterbro, "Holsterbro", resample=resample, target=target
)
metrics(truth_samsoe, pred_samsoe, "Samsoe", resample=resample, target=target)
metrics(
    [
        truth_aarhus,
        truth_holsterbro,
        truth_samsoe,
    ],
    [
        pred_aarhus,
        pred_holsterbro,
        pred_samsoe,
    ],
    "All",
    resample=resample,
    target=target,
)
