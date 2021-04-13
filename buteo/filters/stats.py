import numpy as np
from numba import jit


def stats_to_ints(names):
    ints = np.zeros(len(names), dtype="uint8")

    for idx, name in enumerate(names):
        if name == "count":
            ints[idx] = 1
        elif name == "range":
            ints[idx] = 2
        elif name == "min":
            ints[idx] = 3
        elif name == "max":
            ints[idx] = 4
        elif name == "sum":
            ints[idx] = 5
        elif name == "mean":
            ints[idx] = 6
        elif name == "avg":
            ints[idx] = 6
        elif name == "average":
            ints[idx] = 6
        elif name == "var":
            ints[idx] = 7
        elif name == "variance":
            ints[idx] = 7
        elif name == "std":
            ints[idx] = 8
        elif name == "stdev":
            ints[idx] = 8
        elif name == "standard_deviation":
            ints[idx] = 8
        elif name == "skew":
            ints[idx] = 9
        elif name == "kurtosis":
            ints[idx] = 10
        elif name == "median":
            ints[idx] = 11
        elif name == "med":
            ints[idx] = 11
        elif name == "iqr":
            ints[idx] = 12
        elif name == "q02":
            ints[idx] = 13
        elif name == "q98":
            ints[idx] = 14
        elif name == "q1":
            ints[idx] = 15
        elif name == "q3":
            ints[idx] = 16
        elif name == "mad":
            ints[idx] = 17
        elif name == "median_absolute_deviation":
            ints[idx] = 16
        elif name == "mode":
            ints[idx] = 18
        elif name == "snr":
            ints[idx] = 19
        elif name == "eff":
            ints[idx] = 20
        elif name == "cv":
            ints[idx] = 21
        else:
            raise ValueError("unable to parse the name of the statistics.")

    return ints


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def calculate_array_stats(arr, stats):
    stats_length = int(len(stats))
    result = np.zeros((stats_length), dtype="float32")

    for idx in range(stats_length):
        if stats[idx] == 1:  # count
            result[idx] = arr.size
        elif stats[idx] == 2:  # range
            result[idx] = np.ptp(arr)
        elif stats[idx] == 3:  # min
            result[idx] = np.min(arr)
        elif stats[idx] == 4:  # max
            result[idx] = np.max(arr)
        elif stats[idx] == 5:  # sum
            result[idx] = np.sum(arr)
        elif stats[idx] == 6:  # mean
            result[idx] = np.mean(arr)
        elif stats[idx] == 7:  # var
            result[idx] = np.var(arr)
        elif stats[idx] == 8:  # std
            result[idx] = np.std(arr)
        elif stats[idx] == 9:  # skew
            mean = np.mean(arr)
            std = np.std(arr)
            if std == 0:
                result[idx] = 0.0
                continue
            deviations = np.sum(np.power(arr - mean, 3))
            result[idx] = (deviations * (1 / arr.size)) / (np.power(std, 3))
        elif stats[idx] == 10:  # kurt
            mean = np.mean(arr)
            std = np.std(arr)
            if std == 0:
                result[idx] = 0.0
                continue
            deviations = np.sum(np.power(arr - mean, 4))
            result[idx] = (deviations * (1 / arr.size)) / (np.power(std, 4))
        elif stats[idx] == 11:  # median
            result[idx] = np.median(arr)
        elif stats[idx] == 12:  # iqr
            result[idx] = np.quantile(
                arr, np.array([0.25, 0.75], dtype="float32")
            ).sum()
        elif stats[idx] == 13:  # q02
            result[idx] = np.quantile(arr, 0.02)
        elif stats[idx] == 14:  # q92
            result[idx] = np.quantile(arr, 0.98)
        elif stats[idx] == 15:  # q1
            result[idx] = np.quantile(arr, 0.25)
        elif stats[idx] == 16:  # q3
            result[idx] = np.quantile(arr, 0.75)
        elif stats[idx] == 17:  # mad
            median = np.median(arr)
            absdev = np.abs(arr - median)
            result[idx] = np.median(absdev)
        elif stats[idx] == 18:  # mode
            uniques = np.unique(arr)
            counts = np.zeros_like(uniques, dtype="uint64")

            # numba does not support return count of uniques.
            for idx_values in range(len(arr)):
                val = arr[idx_values]
                for idx_uniques in range(len(uniques)):
                    if val == uniques[idx_uniques]:
                        counts[idx_uniques] = counts[idx_uniques] + 1
            index = np.argmax(counts)
            result[idx] = uniques[index]
        elif stats[idx] == 19:  # snr
            std = np.std(arr)
            if std == 0:
                result[idx] = 0.0
                continue
            result[idx] = np.mean(arr) / std
        elif stats[idx] == 20:  # eff
            mean = np.mean(arr)
            if mean == 0:
                result[idx] = 0.0
                continue
            result[idx] = np.var(arr) / np.power(mean, 2)
        elif stats[idx] == 21:  # cv
            mean = np.mean(arr)
            if mean == 0:
                result[idx] = 0.0
                continue
            result[idx] = np.std(arr) / mean

    return result
