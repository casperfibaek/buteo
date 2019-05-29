from osgeo import gdal, ogr
from osgeo.gdalconst import *
import numpy as np
import sys
import time
gdal.PushErrorHandler('CPLQuietErrorHandler')


def bbox_to_pixel_offsets(gt, bbox):
    originX = gt[0]
    originY = gt[3]
    pixel_width = gt[1]
    pixel_height = gt[5]
    x1 = int((bbox[0] - originX) / pixel_width)
    x2 = int((bbox[1] - originX) / pixel_width) + 1

    y1 = int((bbox[3] - originY) / pixel_height)
    y2 = int((bbox[2] - originY) / pixel_height) + 1

    xsize = x2 - x1
    ysize = y2 - y1
    return (x1, y1, xsize, ysize)


def zonal_stats(vector_path, raster_path, csv=True, prefix='_'):
    raster_datasource = gdal.Open(raster_path, GA_ReadOnly)
    raster_band = raster_datasource.GetRasterBand(1)
    raster_transform = raster_datasource.GetGeoTransform()

    vector_datasource = ogr.Open(vector_path, 1)
    vector_layer = vector_datasource.GetLayer(0)

    vector_driver = ogr.GetDriverByName('Memory')
    raster_driver = gdal.GetDriverByName('MEM')

    if csv:
        csv_path = vector_path.rsplit('.', 1)[0] + '.csv'
        csv_file = f"fid,{prefix}stdev,{prefix}mean\n"
        f = open(csv_path, "a")
        f.write(csv_file)

    # Loop through vectors
    feature = vector_layer.GetNextFeature()
    while feature is not None:

        feature_envelope = feature.geometry().GetEnvelope()
        src_offset = bbox_to_pixel_offsets(raster_transform, feature_envelope)
        src_array = raster_band.ReadAsArray(*src_offset)

        # Create a temporary vector layer in memory
        mem_ds = vector_driver.CreateDataSource('out')
        mem_layer = mem_ds.CreateLayer('poly', None, ogr.wkbPolygon)
        mem_layer.CreateFeature(feat.Clone())

        # Rasterize it
        rvds = raster_driver.Create('', src_offset[2], src_offset[3], 1, gdal.GDT_Byte)
        rvds.SetGeoTransform(new_gt)
        gdal.RasterizeLayer(rvds, [1], mem_layer, burn_values=[1])
        rv_array = rvds.ReadAsArray()

        # Mask the source data array with our current feature
        # we take the logical_not to flip 0<->1 to get the correct mask effect
        # we also mask out nodata values explictly
        masked = np.ma.MaskedArray(src_array, mask=np.logical_not(rv_array))

        if csv:
            f.write(f"{int(feat.GetFID())},{float(masked.std())},{float(masked.mean())}\n")
        else:
            feat.SetField(f'{prefix}stdev', float(masked.std()))
            feat.SetField(f'{prefix}mean', float(masked.mean()))
            vlyr.SetFeature(feat)

        rvds = None
        mem_ds = None
        feat = vlyr.GetNextFeature()

    vds = None
    rds = None

    if csv:
        f.close()

    return

vect = 'D:\\PhD\\Projects\\SavingsAtTheFrontiers\\pilot_analysis\\segmentation\\S2_PCA_RGB5-NIR.shp'
rast = 'D:\\PhD\\Projects\\SavingsAtTheFrontiers\\pilot_analysis\\data\\MSI\\R10m_NDVI.tif'

before = time.time()
stats = zonal_stats(vect, rast, prefix='ndvi_')
after = time.time()
print((after - before) / 60)

# print(stats)
