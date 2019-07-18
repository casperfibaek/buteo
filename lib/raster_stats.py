import numpy as np
from raster_io import raster_to_array


def calc_stats(data, translated_statistics):
    calc_value = np.zeros(20, dtype=np.float64)
    calc_names = np.zeros(20, dtype=np.int8)

    for stat_type in translated_statistics:
        if stat_type == 1:
            calc_names[1] = 1
            calc_value[1] = np.min(data)

        elif stat_type == 2:
            calc_names[2] = 1
            calc_value[2] = np.max(data)

        elif stat_type == 3:
            calc_names[3] = 1
            calc_value[3] = data.size

        elif stat_type == 4:
            if calc_names[1] is not 1:
                calc_names[1] = 1
                calc_value[1] = np.min(data)

            if calc_names[2] is not 1:
                calc_names[2] = 1
                calc_value[2] = np.max(data)

            calc_names[4] = 1
            calc_value[4] = calc_value[2] - calc_value[1]

        elif stat_type == 5:
            calc_names[5] = 1
            calc_value[5] = np.mean(data)

        elif stat_type == 6:
            calc_names[6] = 1
            calc_value[6] = np.median(data)

        elif stat_type == 7:
            calc_names[7] = 1
            calc_value[7] = np.std(data)

        elif stat_type == 8:
            if calc_names[5] is not 1:
                calc_names[5] = 1
                calc_value[5] = np.mean(data)

            if calc_names[7] is not 1:
                calc_names[7] = 1
                calc_value[7] = np.std(data)

            with np.errstate(divide='ignore', invalid='ignore'):
                dev = np.subtract(data, calc_value[5])
                m2 = np.divide(np.power(dev, 2).sum(), data.shape[0])
                m4 = np.divide(np.power(dev, 4).sum(), data.shape[0])

                calc_names[8] = 1
                calc_value[8] = (m4 / (m2 ** 2.0)) - 3.0

        elif stat_type == 9:
            if calc_names[5] is not 1:
                calc_names[5] = 1
                calc_value[5] = np.mean(data)

            if calc_names[7] is not 1:
                calc_names[7] = 1
                calc_value[7] = np.std(data)

            with np.errstate(divide='ignore', invalid='ignore'):
                dev = np.subtract(data, calc_value[5])
                m2 = np.divide(np.power(dev, 2).sum(), data.shape[0])
                m3 = np.divide(np.power(dev, 3).sum(), data.shape[0])

                calc_names[9] = 1
                calc_value[9] = (m3 / (m2 ** 1.5))

        elif stat_type == 10:
            if calc_names[5] is not 1:
                calc_names[5] = 1
                calc_value[5] = np.mean(data)

            if calc_names[6] is not 1:
                calc_names[6] = 1
                calc_value[6] = np.median(data)

            if calc_names[7] is not 1:
                calc_names[7] = 1
                calc_value[7] = np.std(data)

            with np.errstate(divide='ignore', invalid='ignore'):
                calc_names[10] = 1
                calc_value[10] = (calc_value[5] - calc_value[6]) / calc_value[7]

        elif stat_type == 11:
            if calc_names[5] is not 1:
                calc_names[5] = 1
                calc_value[5] = np.mean(data)

            if calc_names[6] is not 1:
                calc_names[6] = 1
                calc_value[6] = np.median(data)

            with np.errstate(divide='ignore', invalid='ignore'):
                calc_names[11] = 1
                calc_value[11] = calc_value[6] / calc_value[5]

        elif stat_type == 12:
            calc_names[12] = 1
            calc_value[12] = np.var(data)

        elif stat_type == 13:
            calc_names[13] = 1
            calc_value[13] = np.quantile(data, 0.25)

        elif stat_type == 14:
            calc_names[14] = 1
            calc_value[14] = np.quantile(data, 0.75)

        elif stat_type == 15:
            calc_names[15] = 1
            calc_value[15] = np.quantile(data, 0.98)

        elif stat_type == 16:
            calc_names[16] = 1
            calc_value[16] = np.quantile(data, 0.02)

        elif stat_type == 17:
            if calc_names[13] is not 1:
                calc_names[13] = 1
                calc_value[13] = np.quantile(data, 0.25)

            if calc_names[14] is not 1:
                calc_names[14] = 1
                calc_value[14] = np.quantile(data, 0.75)

            calc_names[17] = 1
            calc_value[17] = calc_value[14] - calc_value[13]

        elif stat_type == 18:
            if calc_names[6] is not 1:
                calc_names[6] = 1
                calc_value[6] = np.median(data)

            deviations = np.abs(np.subtract(calc_value[5], data))

            calc_names[18] = 1
            calc_value[18] = np.median(deviations)

        elif stat_type == 19:
            if calc_names[18] is not 1:
                if calc_names[6] is not 1:
                    calc_names[6] = 1
                    calc_value[6] = np.median(data)

                deviations = np.abs(np.subtract(calc_value[5], data))

                calc_names[18] = 1
                calc_value[18] = np.median(deviations)

            calc_names[19] = 1
            calc_value[19] = calc_value[18] * 1.4826

    out_stats = np.empty(len(translated_statistics), dtype=np.float32)

    for index, value in enumerate(translated_statistics):
        out_stats[index] = calc_value[value]

    return out_stats


def translate_stats(statistics):
    translated = np.array([], dtype=np.int8)

    for stat in statistics:
        if stat is 'min':
            translated = np.append(translated, 1)
        if stat is 'max':
            translated = np.append(translated, 2)
        if stat is 'count':
            translated = np.append(translated, 3)
        if stat is 'range':
            translated = np.append(translated, 4)
        if stat is 'mean':
            translated = np.append(translated, 5)
        if stat is 'med':
            translated = np.append(translated, 6)
        if stat is 'std':
            translated = np.append(translated, 7)
        if stat is 'kurt':
            translated = np.append(translated, 8)
        if stat is 'skew':
            translated = np.append(translated, 9)
        if stat is 'npskew':
            translated = np.append(translated, 10)
        if stat is 'skewratio':
            translated = np.append(translated, 11)
        if stat is 'var':
            translated = np.append(translated, 12)
        if stat is 'q1':
            translated = np.append(translated, 13)
        if stat is 'q3':
            translated = np.append(translated, 14)
        if stat is 'q98':
            translated = np.append(translated, 15)
        if stat is 'q02':
            translated = np.append(translated, 16)
        if stat is 'iqr':
            translated = np.append(translated, 17)
        if stat is 'mad':
            translated = np.append(translated, 18)
        if stat is 'madstd':
            translated = np.append(translated, 19)

    return translated
