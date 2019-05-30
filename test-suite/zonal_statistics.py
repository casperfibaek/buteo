from osgeo import gdal, ogr, osr
from osgeo.gdalconst import *
from collections import ChainMap
import numpy as np
import multiprocessing
import sys
import time

sys.path.append('../lib')
from raster_stats import raster_stats
from raster_to_array import raster_to_array
from utils import divide_steps, step_ranges


def _get_stats(block):
    print(f'Calculating Zones - {block["id"]}. (STARTED)')
    raster_datasource = gdal.Open(block['raster_path'], GA_ReadOnly)
    vector_datasource = ogr.Open(block['vector_path'])
    vector_layer = vector_datasource.GetLayer(0)
    vector_driver = ogr.GetDriverByName('ESRI Shapefile')
    vector_projection = osr.SpatialReference()
    vector_projection.ImportFromWkt(str(vector_layer.GetSpatialRef()))

    stats = {}

    for fid in range(block['start'], block['stop']):
        vector_feature = vector_layer.GetFeature(fid)
        temp_vector_dataSource = vector_driver.CreateDataSource(f'/vsimem/temp_vector_{fid}.shp')
        temp_vector_layer = temp_vector_dataSource.CreateLayer('temp_polygon', vector_projection, ogr.wkbPolygon)
        vector_geometry = vector_feature.GetGeometryRef()

        if vector_geometry.IsValid():
            temp_vector_layer.CreateFeature(vector_feature.Clone())
        else:
            vector_feature_fixed = ogr.Feature(temp_vector_layer.GetLayerDefn())
            vector_feature_fixed.SetGeometry(vector_geometry.Buffer(0))
            temp_vector_layer.CreateFeature(vector_feature_fixed)
            new_feat = None

        del temp_vector_dataSource, temp_vector_layer, vector_feature

        stats[fid] = raster_stats(
            raster_datasource,
            cutline=f'/vsimem/temp_vector_{fid}.shp',
            cutline_all_touch=False,
            quiet=True,
            statistics=block['statistics'],
        )

    print(f'Calculating Zones - {block["id"]}. (COMPLETED)')

    del raster_datasource, vector_datasource, vector_layer, vector_driver, vector_projection
    return stats


def zonal_stats(vector_path, raster_path, prefix='_', statistics=['mean', 'std']):
    vector_datasource = ogr.Open(vector_path, 1)
    vector_layer = vector_datasource.GetLayer(0)
    total_features = vector_layer.GetFeatureCount()

    vector_field_names = [field.name for field in vector_layer.schema]

    cpus = int(multiprocessing.cpu_count())
    steps = divide_steps(total_features, cpus)
    ranges = step_ranges(steps)

    for obj in ranges:
        obj['vector_path'] = vector_path
        obj['raster_path'] = raster_path
        obj['statistics'] = statistics

    pool = multiprocessing.Pool(cpus)

    composite_parts = pool.map(_get_stats, ranges)
    all_stats = dict(ChainMap(*composite_parts))
    pool.close()

    # Create field if they do not exist.
    for statistic in statistics:
        if f"{prefix}{statistic}" in vector_field_names:
            continue

        statField = ogr.FieldDefn(f"{prefix}{statistic}", ogr.OFTReal)
        vector_layer.CreateField(statField)

    # Updates feature fields.
    for feature in vector_layer:
        for statistic in statistics:
            fid = feature.GetFID()
            feature.SetField(f'{prefix}{statistic}', float(all_stats[feature.GetFID()][statistic]))
        vector_layer.SetFeature(feature)

    del vector_datasource, vector_layer, total_features, cpus, steps, ranges, pool, composite_parts, all_stats
    return


if __name__ == '__main__':
    vect = 'D:\\PhD\\Projects\\SavingsAtTheFrontiers\\pilot_analysis\\segmentation\\pca_rural.shp'
    rast = 'D:\\PhD\\Projects\\SavingsAtTheFrontiers\\pilot_analysis\\data\\MSI\\R10m_NDVI.tif'

    before = time.time()
    zones = zonal_stats(vect, rast, prefix='t1_')
    after = time.time()
    print(f"Zonal_stats took: {(after - before) / 60} m")
