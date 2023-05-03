from osgeo import gdal, osr
import numpy as np
from uuid import uuid4



def create_sample_raster(
        width=10,
        height=10,
        bands=1,
        pixel_width=1,
        pixel_height=1,
        x_min=None,
        y_max=None,
        epsg_code=4326,
        datatype=gdal.GDT_Byte,
        nodata=None,
    ):
    """ Create a sample raster file for testing purposes. """
    raster_path = f"/vsimem/mem_raster_{uuid4().int}.tif"
    driver = gdal.GetDriverByName("GTiff")
    raster = driver.Create(raster_path, width, height, bands, datatype)

    if y_max is None:
        y_max = height * pixel_height
    if x_min is None:
        x_min = 0

    raster.SetGeoTransform((x_min, pixel_width, 0, y_max, 0, -pixel_height))

    for band in range(1, bands + 1):
        raster.GetRasterBand(band).WriteArray(np.random.randint(0, 255, (height, width), dtype=np.uint8))

    if nodata is not None:
        for band in range(1, bands + 1):
            raster.GetRasterBand(band).SetNoDataValue(float(nodata))

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg_code)
    raster.SetProjection(srs.ExportToWkt())
    raster.FlushCache()
    raster = None

    return raster_path
