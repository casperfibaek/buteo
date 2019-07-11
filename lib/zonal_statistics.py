from osgeo import gdal, ogr, osr
from osgeo.gdalconst import *
import multiprocessing
from glob import glob
import sys
import time

sys.path.append('../lib')
from raster_stats import raster_stats
from utils import timing


def zonal_stats(task):
    raster_path = task['raster_path']
    vector_path = task['vector_path']
    statistics = task['statistics']
    prefix = task['prefix']
    cutline_all_touch = task['cutline_all_touch']
    vect_key = task['vect_key']

    raster_datasource = gdal.Open(raster_path, GA_ReadOnly)
    vector_datasource = ogr.Open(vector_path, 0)
    vector_layer = vector_datasource.GetLayer(0)
    vector_driver = ogr.GetDriverByName('ESRI Shapefile')
    vector_projection = osr.SpatialReference()
    vector_projection.ImportFromWkt(str(vector_layer.GetSpatialRef()))
    vector_feature_count = vector_layer.GetFeatureCount()
    vector_field_names = [field.name for field in vector_layer.schema]

    if vect_key is None:
        vect_key = 'fid'
    else:
        if vect_key not in vector_field_names:
            raise RuntimeError('Key not in shapefile.') from None

    csv_lines = ''

    for fid in range(vector_feature_count):
        vector_feature = vector_layer.GetFeature(fid)
        temp_vector_dataSource = vector_driver.CreateDataSource(f'/vsimem/temp_vector_{fid}.shp')
        temp_vector_layer = temp_vector_dataSource.CreateLayer('temp_polygon', vector_projection, ogr.wkbPolygon)
        vector_geometry = vector_feature.GetGeometryRef()
        if vect_key is 'fid':
            feature_key_value = fid
        else:
            feature_key_value = vector_feature.GetField(vect_key)

        gdal.PushErrorHandler('CPLQuietErrorHandler')
        if vector_geometry.IsValid():
            temp_vector_layer.CreateFeature(vector_feature.Clone())
        else:
            vector_feature_fixed = ogr.Feature(temp_vector_layer.GetLayerDefn())
            vector_feature_fixed.SetGeometry(vector_geometry.Buffer(0))
            temp_vector_layer.CreateFeature(vector_feature_fixed)
        gdal.PopErrorHandler()

        temp_vector_layer.SyncToDisk()

        stats = raster_stats(
            raster_datasource,
            cutline=f'/vsimem/temp_vector_{fid}.shp',
            cutline_all_touch=cutline_all_touch,
            quiet=True,
            statistics=statistics,
        )

        csv_lines += f'{feature_key_value},' + ','.join(str(x) for x in stats.values()) + '\n'

    # Create temp csv-file
    vector_folder = vector_path.rsplit('\\', 1)[0]
    vector_name = vector_path.rsplit('\\', 1)[1].rsplit('.', 1)[0]
    csv_path = vector_folder + '\\' + vector_name + '_' + prefix[:-1] + '.csv'
    csv = open(csv_path, 'a')
    csv.write(f'{vect_key},{prefix}{f",{prefix}".join(statistics)}\n')  # Write the header
    csv.write(csv_lines)
    csv.close()

    print('Finished: ', prefix)
    return None


if __name__ == '__main__':
    before = time.time()

    vect = 'E:\\SATF\\phase_II_urban-seperation\\initial_segmentation.shp'
    vect_key = 'DN'

    statistics = ['mean', 'med', 'mad', 'std', 'skew', 'kurt', 'iqr']

    rasters = glob('E:\\SATF\\data\\*.tif')
    tasks = []

    for index, raster in enumerate(rasters):
        tasks.append(
            {'vect_key': vect_key, 'vector_path': vect, 'raster_path': raster, 'prefix': f'{index}_', 'statistics': statistics, 'cutline_all_touch': False},
        )

    pool = multiprocessing.Pool(6, maxtasksperchild=1)
    pool.map(zonal_stats, tasks, chunksize=1)

    pool.close()
    pool.join()

    timing(before)
