from osgeo import gdal, ogr, osr
from osgeo.gdalconst import *
import numpy as np
import multiprocessing
import sys
import time

sys.path.append('../lib')
from raster_stats import raster_stats
from raster_to_array import raster_to_array


gdal.PushErrorHandler('CPLQuietErrorHandler')

def _getStats(vector_list, raster_datasource):
    pool = multiprocessing.Pool(8)



def zonal_stats(vector_path, raster_path, csv=True, prefix='_'):
    raster_datasource = gdal.Open(raster_path, GA_ReadOnly)

    raster_projection = raster_datasource.GetProjection()
    raster_projectionOSR = osr.SpatialReference()
    raster_projectionOSR.ImportFromWkt(str(raster_projection))

    raster_band = raster_datasource.GetRasterBand(1)
    raster_transform = raster_datasource.GetGeoTransform()

    vector_datasource = ogr.Open(vector_path, 1)
    vector_layer = vector_datasource.GetLayer(0)
    vector_projection = osr.SpatialReference()
    vector_projection.ImportFromWkt(str(vector_layer.GetSpatialRef()))

    assert(raster_projectionOSR.IsSame(vector_projection))

    # vector_driver = ogr.GetDriverByName('GeoJSON')
    vector_driver = ogr.GetDriverByName("Esri Shapefile")
    raster_driver = gdal.GetDriverByName('MEM')

    if csv:
        csv_path = vector_path.rsplit('.', 1)[0] + '.csv'
        csv_file = f"fid,{prefix}stdev,{prefix}mean\n"
        f = open(csv_path, "a")
        f.write(csv_file)

    # Loop the features
    for feature in vector_layer:
        mem_vector_datasource = vector_driver.CreateDataSource('/vsimem/temp_vector.shp')
        mem_vector_layer = mem_vector_datasource.CreateLayer('poly', vector_projection, ogr.wkbPolygon)
        mem_vector_layer.CreateFeature(feature.Clone())
        mem_vector_datasource = None

        # stats = raster_stats(raster_datasource, cutline='/vsimem/temp_vector.shp', cutline_all_touch=False, quiet=True, statistics=['mean', 'std'])
        stats = raster_stats(raster_datasource, cutline='/vsimem/temp_vector.shp', cutline_all_touch=False, quiet=True, statistics=['mean', 'std'])
        print(stats)

        exit()
        if csv:
            f.write(f"{int(feature.GetFID())},{stats['std']},{stats['mean']}\n")

    if csv:
        f.close()

    return

vect = 'D:\\PhD\\Projects\\SavingsAtTheFrontiers\\pilot_analysis\\segmentation\\S2_PCA_RGB5-NIR.shp'
rast = 'D:\\PhD\\Projects\\SavingsAtTheFrontiers\\pilot_analysis\\data\\MSI\\R10m_NDVI.tif'
stats = zonal_stats(vect, rast, prefix='_', csv=True)
