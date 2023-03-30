""" Tests for core_raster.py """


# Standard library
import sys; sys.path.append("../")
from uuid import uuid4

# External
import numpy as np
from osgeo import gdal, osr

# Internal
from buteo.raster.reproject import match_raster_projections

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


def test_match_raster_projections_same_projection():
    """Test: match_raster_projections when rasters have the same projection as master"""
    raster1 = create_sample_raster(width=10, height=10, x_min=0, y_max=10, epsg_code=4326)
    raster2 = create_sample_raster(width=10, height=10, x_min=0, y_max=10, epsg_code=4326)
    rasters = [raster1, raster2]
    master = create_sample_raster(width=10, height=10, x_min=0, y_max=10, epsg_code=4326)

    matched_rasters = match_raster_projections(rasters, master)

    for matched_raster in matched_rasters:
        assert isinstance(matched_raster, str)
        raster_ds = gdal.Open(matched_raster)
        assert raster_ds.GetProjection() == gdal.Open(master).GetProjection()

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)
    gdal.Unlink(master)

def test_match_raster_projections_different_projection():
    """ Test: match_raster_projections when rasters have a different projection than master """
    raster1 = create_sample_raster(width=10, height=10, x_min=0, y_max=10, epsg_code=4326)
    raster2 = create_sample_raster(width=10, height=10, x_min=0, y_max=10, epsg_code=4326)
    rasters = [raster1, raster2]
    master = create_sample_raster(width=10, height=10, x_min=0, y_max=10, epsg_code=3857)

    matched_rasters = match_raster_projections(rasters, master)

    for matched_raster in matched_rasters:
        assert isinstance(matched_raster, str)
        raster_ds = gdal.Open(matched_raster)
        assert raster_ds.GetProjection() == gdal.Open(master).GetProjection()

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)
    gdal.Unlink(master)

def test_match_raster_projections_all_different():
    """ Test: match_raster_projections when all rasters have a different projection than master """
    raster1 = create_sample_raster(width=10, height=10, x_min=0, y_max=10, epsg_code=4326)
    raster2 = create_sample_raster(width=10, height=10, x_min=0, y_max=10, epsg_code=32632)
    rasters = [raster1, raster2]
    master = create_sample_raster(width=10, height=10, x_min=0, y_max=10, epsg_code=3857)

    matched_rasters = match_raster_projections(rasters, master)

    for matched_raster in matched_rasters:
        assert isinstance(matched_raster, str)
        raster_ds = gdal.Open(matched_raster)
        assert raster_ds.GetProjection() == gdal.Open(master).GetProjection()

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)
    gdal.Unlink(master)
