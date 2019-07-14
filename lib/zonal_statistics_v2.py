from osgeo import ogr, gdal
from osgeo.gdalconst import GA_ReadOnly
from uuid import uuid4

import sys
import numpy.ma as ma
from glob import glob

sys.path.append('../lib')
from raster_stats import calc_stats
from utils import get_extent, get_intersection, create_geotransform, progress


def zonal(vect, rast, prefix='', stats=['mean', 'med', 'std'], layer_number=0):
    raster = gdal.Open(rast, GA_ReadOnly)
    raster_driver = gdal.GetDriverByName('MEM')
    raster_band = raster.GetRasterBand(1)
    raster_nodata_value = raster_band.GetNoDataValue()
    raster_datatype = raster_band.DataType
    raster_transform = raster.GetGeoTransform()
    raster_projection = raster.GetProjection()
    raster_extent = get_extent(raster)

    vector = ogr.Open(vect, 1)
    vector_layer = vector.GetLayer(layer_number)
    vector_layer_defn = vector_layer.GetLayerDefn()
    vector_feature_count = vector_layer.GetFeatureCount()
    vector_projection = vector_layer.GetSpatialRef()
    vector_driver = ogr.GetDriverByName('ESRI Shapefile')
    vector_id = uuid4()

    for stat in stats:
        if vector_layer_defn.GetFieldIndex(stat) is -1:
            field_defn = ogr.FieldDefn(f'{prefix}{stat}', ogr.OFTReal)
            vector_layer.CreateField(field_defn)

    gdal.PushErrorHandler('CPLQuietErrorHandler')

    for n in range(vector_feature_count):
        vector_feature = vector_layer.GetNextFeature()

        temp_vector_datasource = vector_driver.CreateDataSource(f'/vsimem/{vector_id}.shp')
        temp_vector_layer = temp_vector_datasource.CreateLayer('temp_polygon', vector_projection, ogr.wkbPolygon)
        temp_vector_layer.CreateFeature(vector_feature.Clone())
        temp_vector_layer.SyncToDisk()

        vector_extent = vector_layer.GetExtent()
        vector_extent = (vector_extent[0], vector_extent[2], vector_extent[1], vector_extent[3])

        vector_intersection = get_intersection(raster_extent, vector_extent)
        clip_transform = create_geotransform(raster_transform, vector_intersection)

        raster_target = raster_driver.Create(
            'in_memory_raster',                         # Location of the saved raster, ignored if driver is memory.
            clip_transform['RasterXSize'],              # Dataframe width in pixels (e.g. 1920px).
            clip_transform['RasterYSize'],              # Dataframe height in pixels (e.g. 1280px).
            1,                                          # The number of bands required.
            raster_datatype,                            # Datatype of the destination.
        )

        raster_target.SetGeoTransform(clip_transform['Transform'])
        raster_target.SetProjection(raster_projection)

        gdal.Warp(
            raster_target,
            raster,
            format='MEM',
            xRes=raster_transform[1],
            yRes=raster_transform[5],
            srcNodata=raster_nodata_value,
            cutlineDSName=f'/vsimem/{vector_id}.shp',
        )

        clipped_band = raster_target.GetRasterBand(1)

        if raster_nodata_value is None:
            data = ma.array(clipped_band.ReadAsArray(), fill_value=0).compressed()
        else:
            data = ma.masked_equal(clipped_band.ReadAsArray(), raster_nodata_value).compressed()

        zonal_stats = calc_stats(data, stats)

        for key in zonal_stats.keys():
            vector_feature.SetField(f'{prefix}{key}', zonal_stats[key])

        # import pdb; pdb.set_trace()
        vector_layer.SetFeature(vector_feature)
        # vector_layer.Update()

        gdal.Unlink(f'/vsimem/{vector_id}.shp')
        gdal.Unlink('in_memory_raster')

        progress(n, vector_feature_count - 1, 'mean')

    vector_layer.SyncToDisk()
    return


if __name__ == '__main__':
    vect = 'C:\\Users\\CFI\\Desktop\\scratch_pad\\urban_classification_zonal.shp'
    rast = 'C:\\Users\\CFI\\Desktop\\scratch_pad\\dry_b04.tif'
    # vect = 'E:\\SATF\\phase_IV_urban-classification\\urban_classification_zonal.shp'
    # rast_folder = 'E:\\SATF\\data\\s2\\'
    # rasters = glob(rast_folder + '*.tif')

    zonal(vect, rast, stats=['mean', 'med', 'mad', 'std', 'kurt', 'skew', 'iqr'])

    # for index, value in enumerate(rasters):
    #     zonal(vect, rasters[index], prefix=f"{index}_", stats=['mean', 'med', 'mad', 'std', 'kurt', 'skew', 'iqr'])
