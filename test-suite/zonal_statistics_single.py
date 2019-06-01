from osgeo import gdal, ogr, osr
from osgeo.gdalconst import *
import multiprocessing
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

    vect = 'D:\\PhD\\Projects\\SavingsAtTheFrontiers\\pilot_analysis\\segmentation\\S2_MSI_ALL.shp'
    vect_key = 'DN'
    msi_base = 'D:\\PhD\\Projects\\SavingsAtTheFrontiers\\pilot_analysis\\data\\MSI\\'
    B02 = msi_base + 'R10m_B02.tif'
    B03 = msi_base + 'R10m_B03.tif'
    B04 = msi_base + 'R10m_B04.tif'
    B05 = msi_base + 'R10m_B05.tif'
    B06 = msi_base + 'R10m_B06.tif'
    B07 = msi_base + 'R10m_B07.tif'
    B08 = msi_base + 'R10m_B08.tif'
    B8A = msi_base + 'R10m_B8A.tif'
    B11 = msi_base + 'R10m_B11.tif'
    B12 = msi_base + 'R10m_B12.tif'
    sar_vv_coh = 'D:\\PhD\\Projects\\SavingsAtTheFrontiers\\pilot_analysis\\data\\SAR\\VV_coh.tif'
    sar_vv_db = 'D:\\PhD\\Projects\\SavingsAtTheFrontiers\\pilot_analysis\\data\\SAR\\VV_backscatter.tif'
    sar_vh_coh = 'D:\\PhD\\Projects\\SavingsAtTheFrontiers\\pilot_analysis\\data\\SAR\\VH_coh.tif'
    sar_vh_db = 'D:\\PhD\\Projects\\SavingsAtTheFrontiers\\pilot_analysis\\data\\SAR\\VH_backscatter.tif'
    statistics = ['min', 'max', 'mean', 'med', 'std', 'mad', 'skew', 'kurt', 'q1', 'q3']
    cutline_all_touch = False

    tasks = [
        {'vect_key': vect_key, 'vector_path': vect, 'raster_path': sar_vv_coh, 'prefix': 'vvc_', 'statistics': statistics, 'cutline_all_touch': False},
        {'vect_key': vect_key, 'vector_path': vect, 'raster_path': sar_vv_db, 'prefix': 'vvb_', 'statistics': statistics, 'cutline_all_touch': False},
        {'vect_key': vect_key, 'vector_path': vect, 'raster_path': sar_vh_coh, 'prefix': 'vhc_', 'statistics': statistics, 'cutline_all_touch': False},
        {'vect_key': vect_key, 'vector_path': vect, 'raster_path': sar_vh_db, 'prefix': 'vhb_', 'statistics': statistics, 'cutline_all_touch': False},
        {'vect_key': vect_key, 'vector_path': vect, 'raster_path': B02, 'prefix': 'b02_', 'statistics': statistics, 'cutline_all_touch': False},
        {'vect_key': vect_key, 'vector_path': vect, 'raster_path': B03, 'prefix': 'b03_', 'statistics': statistics, 'cutline_all_touch': False},
        {'vect_key': vect_key, 'vector_path': vect, 'raster_path': B04, 'prefix': 'b04_', 'statistics': statistics, 'cutline_all_touch': False},
        {'vect_key': vect_key, 'vector_path': vect, 'raster_path': B05, 'prefix': 'b05_', 'statistics': statistics, 'cutline_all_touch': False},
        {'vect_key': vect_key, 'vector_path': vect, 'raster_path': B06, 'prefix': 'b06_', 'statistics': statistics, 'cutline_all_touch': False},
        {'vect_key': vect_key, 'vector_path': vect, 'raster_path': B07, 'prefix': 'b07_', 'statistics': statistics, 'cutline_all_touch': False},
        {'vect_key': vect_key, 'vector_path': vect, 'raster_path': B08, 'prefix': 'b08_', 'statistics': statistics, 'cutline_all_touch': False},
        {'vect_key': vect_key, 'vector_path': vect, 'raster_path': B8A, 'prefix': 'b8A_', 'statistics': statistics, 'cutline_all_touch': False},
        {'vect_key': vect_key, 'vector_path': vect, 'raster_path': B11, 'prefix': 'b11_', 'statistics': statistics, 'cutline_all_touch': False},
        {'vect_key': vect_key, 'vector_path': vect, 'raster_path': B12, 'prefix': 'b12_', 'statistics': statistics, 'cutline_all_touch': False},
    ]

    # for task in tasks:
    #     zonal_stats(task)

    pool = multiprocessing.Pool(7)
    pool.map(zonal_stats, tasks, chunksize=1)

    timing(before)
