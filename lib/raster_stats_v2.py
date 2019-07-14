import numpy as np
from scipy.stats import stats
from raster_to_array import raster_to_array


def _calc_stats(data, statistics):
    calculated_names = np.array([], dtype=str)
    calculated_values = np.array([], dtype=float)

    for stat_type in statistics:
        if stat_type == 'min':
            calculated_names.append('min')
            calculated_values.append(np.min(data))

        elif stat_type == 'max':
            calculated_names.append('max')
            calculated_values.append(np.max(data))

        elif stat_type == 'count':
            calculated_names.append('count')
            calculated_values.append(data.size)

        elif stat_type == 'range':
            if 'min' not in calculated_names:
                calculated_names.append('min')
                calculated_values.append(np.min(data))

            if 'max' not in calculated_names:
                calculated_names.append('max')
                calculated_values.append(np.max(data))

            calculated_names.append('range')
            calculated_values.append(
                calculated_values[calculated_names.index('max')] - calculated_values[calculated_names.index('min')]
            )

        elif stat_type == 'mean':
            calculated_names.append('mean')
            calculated_values.append(np.mean(data))

        elif stat_type == 'med':
            calculated_names.append('med')
            calculated_values.append(np.median(data))

        elif stat_type == 'std':
            calculated_names.append('std')
            calculated_values.append(np.std(data))

        elif stat_type == 'kurt':
            calculated_names.append('kurt')
            calculated_values.append(stats.kurtosis(data))

        elif stat_type == 'skew':
            calculated_names.append('skew')
            calculated_values.append(stats.skew(data))

        elif stat_type == 'npskew':
            if 'mean' not in calculated_names:
                calculated_names.append('mean')
                calculated_values.append(np.mean(data))

            if 'med' not in calculated_names:
                calculated_names.append('med')
                calculated_values.append(np.median(data))

            if 'std' not in calculated_names:
                calculated_names.append('std')
                calculated_values.append(np.std(data))

            calculated_names.append('npskew')
            calculated_values.append(
                (calculated_values[calculated_names.index('mean')] - calculated_values[calculated_names.index('med')]) / calculated_values[calculated_names.index('std')]
            )

        elif stat_type == 'skewratio':
            if 'mean' not in calculated_names:
                calculated_names.append('mean')
                calculated_values.append(np.mean(data))

            if 'med' not in calculated_names:
                calculated_names.append('med')
                calculated_values.append(np.median(data))

            calculated_names.append('skewratio')
            calculated_values.append(
                (calculated_values[calculated_names.index('med')] / calculated_values[calculated_names.index('mean')])
            )

        elif stat_type == 'variation':
            calculated_names.append('variation')
            calculated_values.append(stats.variation(data))

        elif stat_type == 'q1':
            calculated_names.append('q1')
            calculated_values.append(np.quantile(data, 0.25))

        elif stat_type == 'q3':
            calculated_names.append('q3')
            calculated_values.append(np.quantile(data, 0.75))

        elif stat_type == 'q98':
            calculated_names.append('q98')
            calculated_values.append(np.quantile(data, 0.98))

        elif stat_type == 'q02':
            calculated_names.append('q02')
            calculated_values.append(np.quantile(data, 0.02))

        elif stat_type == 'iqr':
            if 'q1' not in calculated_names:
                calculated_names.append('q1')
                calculated_values.append(np.quantile(data, 0.25))

            if 'q3' not in calculated_names:
                calculated_names.append('q3')
                calculated_values.append(np.quantile(data, 0.75))

            calculated_names.append('iqr')
            calculated_values.append(
                (calculated_values[calculated_names.index('q3')] - calculated_values[calculated_names.index('q1')])
            )

        elif stat_type == 'mad':
            if 'med' not in calculated_names:
                calculated_names.append('med')
                calculated_values.append(np.median(data))

            deviations = np.abs(np.subtract(calculated_values[calculated_names.index('med')], data))

            calculated_names.append('mad')
            calculated_values.append(np.median(deviations))

        elif stat_type == 'madstd':
            if 'med' not in calculated_names:
                calculated_names.append('med')
                calculated_values.append(np.median(data))

            if 'mad' not in calculated_names:
                deviations = np.abs(np.subtract(calculated_values[calculated_names.index('med')], data))

                calculated_names.append('mad')
                calculated_values.append(np.median(deviations))

            calculated_names.append('madstd')
            calculated_values.append(calculated_values[calculated_names.index('med')] * 1.4826)

    out_stats = np.array([], dtype=float)

    for index, value in enumerate(calculated_names):
        if value in statistics:
            out_stats.append(calculated_values[index])

    return out_stats

calc_stats = np.vectorize(_calc_stats, otypes=[float])
