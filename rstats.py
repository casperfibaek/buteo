from osgeo import gdal
import numpy as np
import numpy.ma as ma
from scipy.stats import mstats
import matplotlib.pyplot as plt


def rstats(geometry, inRaster, outRaster='memory', nodata=None,
           statistics=('std', 'mean', 'median'), histogram=False):
    cut = gdal.Warp(
        outRaster,
        inRaster,
        format='MEM',
        cutlineDSName=geometry,
        cropToCutline=True,
        multithread=True,
    )

    band = cut.GetRasterBand(1)
    bandNDV = band.GetNoDataValue()
    data = band.ReadAsArray().flatten()

    cut = None # release memory

    if nodata is None and bandNDV is not None:
        ndv = bandNDV
    elif nodata is not None:
        ndv = nodata
    else:
        ndv = None

    mdata = ma.masked_where(data == ndv, data)

    holder = {}

    if 'min' in statistics:
        holder['min'] = np.ma.min(mdata)
    if 'max' in statistics:
        holder['max'] = np.ma.max(mdata)
    if 'count' in statistics:
        holder['count'] = np.ma.count(mdata)
    if 'range' in statistics:
        holder['range'] = np.ma.max(mdata) - np.ma.min(mdata)
    if 'mean' in statistics:
        holder['mean'] = np.ma.mean(mdata)
    if 'median' in statistics:
        holder['median'] = np.ma.median(mdata)
    if 'std' in statistics:
        holder['std'] = np.ma.std(mdata)
    if 'kurtosis' in statistics:
        holder['kurtosis'] = mstats.kurtosis(mdata)
    if 'skew' in statistics:
        holder['skew'] = mstats.skew(mdata).data.sum()
    if 'npskew' in statistics:
        holder['npskew'] = (np.ma.mean(mdata) - np.ma.median(mdata)) / np.ma.std(mdata)
    if 'skewratio' in statistics:
        holder['skewratio'] = 1 - (np.ma.median(mdata) / np.ma.mean(mdata))
    if 'variation' in statistics:
        holder['variation'] = mstats.variation(mdata)
    if 'mad' in statistics:
        deviations = np.abs(np.ma.mean(mdata) - mdata)
        holder['mad'] = np.ma.median(deviations)
    if 'madstd' in statistics:
        if 'mad' in statistics:
            holder['madstd'] = np.ma.median(deviations) * 1.4826
        else:
            deviations = np.abs(np.ma.median(mdata) - mdata)
            madstd = np.ma.median(deviations) * 1.4826
            holder['madstd'] = madstd

    if histogram is True:
        plt.hist(data[data != ndv], bins='auto')
        plt.title(f'{geometry} on {inRaster}')
        plt.show()

    return holder
