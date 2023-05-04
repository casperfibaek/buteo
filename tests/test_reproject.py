""" Tests for core_raster.py """

# Standard library
import sys; sys.path.append("../")

# External
from osgeo import gdal

# Internal
from utils_tests import create_sample_raster
from buteo.raster.reproject import raster_match_projections



def test_match_raster_projections_same_projection():
    """Test: match_raster_projections when rasters have the same projection as master"""
    raster1 = create_sample_raster(width=10, height=10, x_min=0, y_max=10, epsg_code=4326)
    raster2 = create_sample_raster(width=10, height=10, x_min=0, y_max=10, epsg_code=4326)
    rasters = [raster1, raster2]
    master = create_sample_raster(width=10, height=10, x_min=0, y_max=10, epsg_code=4326)

    matched_rasters = raster_match_projections(rasters, master)

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

    matched_rasters = raster_match_projections(rasters, master)

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

    matched_rasters = raster_match_projections(rasters, master)

    for matched_raster in matched_rasters:
        assert isinstance(matched_raster, str)
        raster_ds = gdal.Open(matched_raster)
        assert raster_ds.GetProjection() == gdal.Open(master).GetProjection()

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)
    gdal.Unlink(master)
