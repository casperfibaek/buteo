import numpy as np
import numpy.ma as ma
from osgeo import gdal
from raster_io import raster_to_array, array_to_raster


def shift_raster(in_raster, out_raster, x, y):
    raster = gdal.Open(in_raster, 0)
    raster_transform = raster.GetGeoTransform()
    raster_projection = raster.GetProjection()
    raster_band = raster.GetRasterBand(1)
    raster_nodata_value = raster_band.GetNoDataValue()
    raster_datatype = raster_band.DataType

    raster_driver = gdal.GetDriverByName('GTiff')

    raster_target = raster_driver.Create(
        out_raster,                         # Location of the saved raster, ignored if driver is memory.
        raster.RasterXSize,                 # Dataframe width in pixels (e.g. 1920px).
        raster.RasterYSize,                 # Dataframe height in pixels (e.g. 1280px).
        1,                                  # The number of bands required.
        raster_datatype,                    # Datatype of the destination.
    )

    new_transform = list(raster_transform)
    new_transform[0] += x
    new_transform[3] += y
    raster_target.SetGeoTransform(tuple(new_transform))
    raster_target.SetProjection(raster_projection)
    raster_target_band = raster_target.GetRasterBand(1)
    raster_target_band.WriteArray(raster_band.ReadAsArray())

    if raster_nodata_value is not None:
        raster_target_band.SetNoDataValue(raster_nodata_value)

    raster_target = None
    raster = None

    return out_raster

if __name__ == '__main__':
    rast = 'E:\\sentinel_1_data\\ghana\\slc\\step2\\wet_season.data\\coh_VV_15Jun2019_21Jun2019.img'
    out_rast = 'E:\\sentinel_1_data\\ghana\\slc\\step2\\wet_season.data\\coh_VV_15Jun2019_21Jun2019_moved.tif'

    shift_raster(rast, out_rast, 80, 0)
