import sys
import numpy as np

sys.path.append("../")
sys.path.append("../../")
np.set_printoptions(suppress=True)

from buteo.raster.io import raster_to_array, array_to_raster
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


def print_result(number, name, pred, truth, zfill=6):
    pred = pred.flatten()
    truth = truth.flatten()

    _mae = str(round(mean_absolute_error(truth, pred), 3)).zfill(zfill)
    _rmse = str(round(mean_squared_error(truth, pred, squared=False), 3)).zfill(zfill)
    _exvar = str(round(explained_variance_score(truth, pred), 3)).zfill(zfill)
    tpe = str(np.round((pred.sum() / truth.sum()) * 100, 3)).zfill(zfill)

    print(
        f"{number} - {name}. MAE: {_mae} - RMSE: {_rmse} - Exp.var: {_exvar} - TPE: {tpe}"
    )


def print_result_bin(name, _pred, _truth, zfill=6):
    bacc = str(round(balanced_accuracy_score(_truth, _pred), 3)).zfill(zfill)
    acc = str(round(accuracy_score(_truth, _pred), 3)).zfill(zfill)
    f1 = str(round(f1_score(_truth, _pred), 3)).zfill(zfill)
    precision = str(round(precision_score(_truth, _pred, zero_division=True), 3)).zfill(
        zfill
    )
    recall = str(round(recall_score(_truth, _pred, zero_division=True), 3)).zfill(zfill)
    tpe = str(np.round((_pred.sum() / _truth.sum()) * 100, 3)).zfill(zfill)
    print(
        f"{name} Binary || Bal. Accuracy: {bacc} - Accuracy: {acc} - F1: {f1} - Precision: {precision} - Recall: {recall} - TPE: {tpe}"
    )


folder = "C:/Users/caspe/Desktop/new_labels/"

truth = raster_to_array(folder + "4_label_area.tif").flatten()

mean = raster_to_array(folder + "id_888_mean.tif").flatten()
median = raster_to_array(folder + "id_888_median.tif").flatten()
mad = raster_to_array(folder + "id_888_mad.tif").flatten()

print_result("1", "Mean", mean, truth)
print_result("2", "Median", median, truth)
print_result("3", "Mad", mad, truth)

limit = 0.5
print_result_bin("Mean", mean >= limit, truth >= limit)
print_result_bin("Median", median >= limit, truth >= limit)
print_result_bin("Mad", mad >= limit, truth >= limit)
