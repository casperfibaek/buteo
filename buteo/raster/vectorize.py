"""
Module to vectorize rasters using the GDAL tools.

TODO:
    - Improve documentation
    - Improve function
    - Type check
"""

import sys; sys.path.append("../../") # Path: buteo/raster/vectorize.py

from osgeo import gdal, ogr

from buteo.raster.io import open_raster, raster_to_metadata


def vectorize_raster(raster, out_path_shp, band=1):
    meta = raster_to_metadata(raster)
    opened = open_raster(raster)
    srcband = opened.GetRasterBand(band)

    projection = meta["projection_osr"]

    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(out_path_shp)
    dst_layer = dst_ds.CreateLayer(out_path_shp, srs=projection)

    gdal.Polygonize(srcband, None, dst_layer, 0)

    return out_path_shp
