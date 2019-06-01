from osgeo import gdal, ogr, osr
from osgeo.gdalconst import *
from raster_stats import raster_stats


def _get_stats(block):
    raster_datasource = gdal.Open(block['raster_path'], GA_ReadOnly)
    vector_datasource = ogr.Open(block['vector_path'])
    vector_layer = vector_datasource.GetLayer(0)
    vector_driver = ogr.GetDriverByName('ESRI Shapefile')
    vector_projection = osr.SpatialReference()
    vector_projection.ImportFromWkt(str(vector_layer.GetSpatialRef()))

    csv = open(block['csv_path'], 'a')

    for fid in block['fids']:
        vector_feature = vector_layer.GetFeature(fid)
        temp_vector_dataSource = vector_driver.CreateDataSource(f'/vsimem/temp_vector_{fid}.shp')
        temp_vector_layer = temp_vector_dataSource.CreateLayer('temp_polygon', vector_projection, ogr.wkbPolygon)
        vector_geometry = vector_feature.GetGeometryRef()

        gdal.PushErrorHandler('CPLQuietErrorHandler')
        if vector_geometry.IsValid():
            temp_vector_layer.CreateFeature(vector_feature.Clone())
        else:
            vector_feature_fixed = ogr.Feature(temp_vector_layer.GetLayerDefn())
            vector_feature_fixed.SetGeometry(vector_geometry.Buffer(0))
            temp_vector_layer.CreateFeature(vector_feature_fixed)
            del vector_feature_fixed
        gdal.PopErrorHandler()

        temp_vector_layer.SyncToDisk()
        # del temp_vector_dataSource, temp_vector_layer, vector_feature, vector_geometry

        stats = raster_stats(
            raster_datasource,
            cutline=f'/vsimem/temp_vector_{fid}.shp',
            cutline_all_touch=block['cutline_all_touch'],
            quiet=True,
            statistics=block['statistics'],
        )

        csv_line = f'{fid},' + ','.join(str(x) for x in stats.values()) + '\n'
        csv.write(csv_line)

    csv.close()

    return

    # del raster_datasource, vector_datasource, vector_layer, vector_driver, vector_projection
    # return
