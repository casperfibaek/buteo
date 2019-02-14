import gdal
import numpy as np
from enum import Enum
from numba import jit
from scipy.stats import stats
import matplotlib.pyplot as plt
import math
from rasterUtils.rasterToArray import rasterToArray


# TODO: Testing of this function
def bincount(arr, iqr=None, rng=None):
    if len(arr) > 1000:
        if iqr is None:
            q1 = np.quantile(arr, 0.25)
            q3 = np.quantile(arr, 0.75)
            iqr = q3 - q1

        if rng is None:
            rng = np.range(arr)

        # Freedman Diaconis Estimator
        fd = round(math.ceil(rng / (2 * (iqr / math.pow(len(arr), 1 / 3)) + 1)))
        return fd
    else:
        # Sturges
        return round(math.log2(len(arr)) + 1)


@jit(nopython=True, parallel=True, fastmath=True)
def calcStats2(data, statistics: list):
    results = []
    for statType in statistics:

        if statType is 'min':
            results.append(np.min(data))

        if statType is 'max':
            results.append(np.max(data))

    return results


def calcStats(data, statistics=('std', 'mean', 'median')):
    holder = {}

    if 'min' in statistics:
        holder['min'] = np.min(data)

    if 'max' in statistics:
        holder['max'] = np.max(data)

    if 'count' in statistics:
        holder['count'] = len(data)

    if 'range' in statistics:
        if 'min' in holder:
            _min = holder['min']
        else:
            _min = np.min(data)
        if 'max' in holder:
            _max = holder['max']
        else:
            _max = np.max(data)
        holder['range'] = _max - _min

    if 'mean' in statistics:
        holder['mean'] = np.mean(data)

    if 'median' in statistics:
        holder['median'] = np.median(data)

    if 'std' in statistics:
        holder['std'] = np.std(data)

    if 'kurtosis' in statistics:
        holder['kurtosis'] = stats.kurtosis(data)

    if 'skew' in statistics:
        holder['skew'] = stats.skew(data)

    if 'npskew' in statistics:
        if 'mean' in holder:
            _mean = holder['mean']
        else:
            _mean = np.mean(data)
        if 'median' in holder:
            _median = holder['median']
        else:
            _median = np.median(data)
        if 'std' in holder:
            _std = holder['std']
        else:
            _std = np.std(data)
        holder['npskew'] = (_mean - _median) / _std

    if 'skewratio' in statistics:
        if 'mean' in holder:
            _mean = holder['mean']
        else:
            _mean = np.mean(data)
        if 'median' in holder:
            _median = holder['median']
        holder['skewratio'] = 1 - (_median / _mean)

    if 'variation' in statistics:
        holder['variation'] = stats.variation(data)

    if 'q1' in statistics:
        holder['q1'] = np.quantile(data, 0.25)

    if 'q3' in statistics:
        holder['q3'] = np.quantile(data, 0.75)

    if 'iqr' in statistics:
        if 'q1' in holder and 'q3' in statistics:
            holder['iqr'] = holder['q3'] - holder['q1']
        else:
            holder['iqr'] = np.quantile(data, 0.75) - np.quantile(data, 0.25)

    if 'mad' in statistics:
        if 'median' in holder:
            deviations = np.abs(holder['median'] - data)
        else:
            deviations = np.abs(np.median(data) - data)

        holder['mad'] = np.median(deviations)

    if 'madstd' in statistics:
        if 'mad' in holder:
            holder['madstd'] = holder['mad'] * 1.4826
        else:
            if 'median' in holder:
                deviations = np.abs(holder['median'] - data)
            else:
                deviations = np.abs(np.median(data) - data)

            holder['madstd'] = np.median(deviations) * 1.4826

    if 'within3std' in statistics:
        if 'std' in holder:
            _std = holder['std']
        else:
            _std = np.std(data)
        if 'mean' in holder:
            _mean = holder['mean']
        else:
            _mean = np.mean(data)
        if 'count' in holder:
            _count = holder['count']
        else:
            _count = len(data)

        _std3 = _std * 3
        lowerbound = _mean - _std3
        higherbound = _mean + _std3

        belowLimit = _count - len(data[data > lowerbound])
        aboveLimit = _count - len(data[data < higherbound])

        percent = 100 - (((belowLimit + aboveLimit) / _count) * 100)
        holder['within3std'] = percent

    if 'within3std_mad' in statistics:
        if 'median' in holder:
            _median = holder['median']
        else:
            _median = np.median(data)
        if 'madstd' in holder:
            _madstd = holder['madstd']
        else:
            _madstd = np.median(np.abs(_median - data)) * 1.4826
        if 'count' in holder:
            _count = holder['count']
        else:
            _count = len(data)

        _madstd3 = _madstd * 3
        lowerbound = _median - _madstd3
        higherbound = _median + _madstd3

        belowLimit = _count - len(data[data > lowerbound])
        aboveLimit = _count - len(data[data < higherbound])

        percent = 100 - (((belowLimit + aboveLimit) / _count) * 100)
        holder['within3std_mad'] = percent

    return holder


class StatTypes(Enum):
    min = 1
    max = 2
    count = 3
    range = 4
    mean = 5
    median = 6
    std = 7
    kurtosis = 8
    skew = 9
    npskew = 10
    skewratio = 11
    variation = 12
    q1 = 13
    q3 = 14
    iqr = 15
    mad = 16
    madstd = 17
    within3std = 18
    within3std_mad = 19


def rasterStats(inRaster, cutline=None, srcNoDataValue=None,
                histogram=False, cutlineAllTouch=True, bandToCalculate=1,
                quiet=False, statistics=('std', 'mean', 'median')):

    data = rasterToArray(
        inRaster,
        cutline=cutline,
        cutlineAllTouch=cutlineAllTouch,
        compressed=True,
        srcNoDataValue=srcNoDataValue,
        bandToClip=bandToCalculate,
        quiet=quiet,
        calcBandStats=False,
    )
    # stats = calcStats(data, statistics)
    stats = calcStats2(data, ('min', 'max'))

    if histogram is True:
        plt.hist(data, bins='auto')
        plt.show()

    return stats
