from osgeo import gdal, ogr, osr
from osgeo.gdalconst import *
import numpy as np
import multiprocessing
import sys
import time

sys.path.append('../lib')
from raster_stats import raster_stats
from raster_to_array import raster_to_array
from utils import divide_steps, step_ranges


gdal.PushErrorHandler('CPLQuietErrorHandler')


def _get_stats(block):
    start = block['start']
    stop = block['stop']
    vector_path = block['vector_path']
    raster_path = block['raster_path']

    raster_datasource = gdal.Open(raster_path, GA_ReadOnly)

    stats = {}
    for fid in range(start, stop + 1):
        stat = raster_stats(
            raster_datasource,
            cutline=vector_path,
            cutline_all_touch=False,
            quiet=True,
            statistics=['mean', 'std'],
            cutlineWhere=f'FID={fid}',
        )

        stats[f"{fid}"] = {'mean': stat['mean'], 'std': stat['std']}
        stat = None

    return stats


def zonal_stats(vector_path, raster_path):
    vector_datasource = ogr.Open(vector_path)
    vector_layer = vector_datasource.GetLayer(0)

    total_features = vector_layer.GetFeatureCount()

    cpus = int(multiprocessing.cpu_count())
    steps = divide_steps(total_features, cpus)
    ranges = step_ranges(steps)
    for obj in ranges:
        obj['vector_path'] = vector_path
        obj['raster_path'] = raster_path

    pool = multiprocessing.Pool(cpus)

    composite_parts = pool.map(_get_stats, ranges)
    pool.close()

    return composite_parts

if __name__ == '__main__':
    vect = 'D:\\PhD\\Projects\\SavingsAtTheFrontiers\\pilot_analysis\\segmentation\\S2_PCA_RGB5-NIR.shp'
    rast = 'D:\\PhD\\Projects\\SavingsAtTheFrontiers\\pilot_analysis\\data\\MSI\\R10m_NDVI.tif'
    stats = zonal_stats(vect, rast)
    print(stats)
