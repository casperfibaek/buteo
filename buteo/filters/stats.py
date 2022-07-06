"""
Calculates various statistics for a given array.

TODO:
    - Improve documentation
    - Add tests
"""

import numpy as np
from enum import Enum

from numba import jit


class stat(Enum):
    count = 1
    range = 2
    min = 3
    max = 4
    sum = 5
    mean = 6
    avg = 6
    average = 6
    var = 7
    variance = 7
    std = 8
    stdev = 8
    standard_deviation = 8
    skew = 9
    kurtosis = 10
    median = 11
    med = 11
    iqr = 12
    q02 = 13
    q98 = 14
    q1 = 15
    q3 = 16
    mad = 17
    median_absolute_deviation = 17
    mode = 18
    snr = 19
    eff = 20
    cv = 21


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def calculate_array_stats(arr, stats):
    stats_length = int(len(stats))
    result = np.zeros((stats_length), dtype="float32")

    for idx in range(stats_length):
        if stat(stats[idx]) == 1:  # count
            result[idx] = arr.size
        elif stat(stats[idx]) == 2:  # range
            result[idx] = np.ptp(arr)
        elif stat(stats[idx]) == 3:  # min
            result[idx] = np.min(arr)
        elif stat(stats[idx]) == 4:  # max
            result[idx] = np.max(arr)
        elif stat(stats[idx]) == 5:  # sum
            result[idx] = np.sum(arr)
        elif stat(stats[idx]) == 6:  # mean
            result[idx] = np.mean(arr)
        elif stat(stats[idx]) == 7:  # var
            result[idx] = np.var(arr)
        elif stat(stats[idx]) == 8:  # std
            result[idx] = np.std(arr)
        elif stat(stats[idx]) == 9:  # skew
            mean = np.mean(arr)
            std = np.std(arr)
            if std == 0:
                result[idx] = 0.0
                continue
            deviations = np.sum(np.power(arr - mean, 3))
            result[idx] = (deviations * (1 / arr.size)) / (np.power(std, 3))
        elif stat(stats[idx]) == 10:  # kurt
            mean = np.mean(arr)
            std = np.std(arr)
            if std == 0:
                result[idx] = 0.0
                continue
            deviations = np.sum(np.power(arr - mean, 4))
            result[idx] = (deviations * (1 / arr.size)) / (np.power(std, 4))
        elif stat(stats[idx]) == 11:  # median
            result[idx] = np.median(arr)
        elif stat(stats[idx]) == 12:  # iqr
            result[idx] = np.quantile(
                arr, np.array([0.25, 0.75], dtype="float32")
            ).sum()
        elif stat(stats[idx]) == 13:  # q02
            result[idx] = np.quantile(arr, 0.02)
        elif stat(stats[idx]) == 14:  # q92
            result[idx] = np.quantile(arr, 0.98)
        elif stat(stats[idx]) == 15:  # q1
            result[idx] = np.quantile(arr, 0.25)
        elif stat(stats[idx]) == 16:  # q3
            result[idx] = np.quantile(arr, 0.75)
        elif stat(stats[idx])== 17:  # mad
            median = np.median(arr)
            absdev = np.abs(arr - median)
            result[idx] = np.median(absdev)
        elif stat(stats[idx]) == 18:  # mode
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
        elif stat(stats[idx]) == 19:  # snr
            std = np.std(arr)
            if std == 0:
                result[idx] = 0.0
                continue
            result[idx] = np.mean(arr) / std
        elif stat(stats[idx]) == 20:  # eff
            mean = np.mean(arr)
            if mean == 0:
                result[idx] = 0.0
                continue
            result[idx] = np.var(arr) / np.power(mean, 2)
        elif stat(stats[idx]) == 21:  # cv
            mean = np.mean(arr)
            if mean == 0:
                result[idx] = 0.0
                continue
            result[idx] = np.std(arr) / mean

    return result
