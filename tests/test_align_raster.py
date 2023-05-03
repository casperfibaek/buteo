""" Tests for core_raster.py """


# Standard library
import sys; sys.path.append("../")
from uuid import uuid4

# External
import numpy as np
import pytest
from osgeo import gdal, osr

# Internal
from buteo.raster.align import raster_align_to_reference, raster_align
from buteo.raster.core_raster import check_rasters_are_aligned, raster_to_metadata


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

def test_align_rasters_single_raster():
    """ Test: align_rasters with a single raster """
    raster1 = create_sample_raster()
    aligned_raster_path = raster_align_to_reference(raster1, reference=raster1)

    assert check_rasters_are_aligned(aligned_raster_path + [raster1])
    assert gdal.Open(aligned_raster_path[0]) is not None

    gdal.Unlink(raster1)
    gdal.Unlink(aligned_raster_path[0])


def test_align_rasters_multiple_rasters_reference():
    """ Test: align_rasters with multiple rasters """
    raster1 = create_sample_raster(width=10, height=10)
    raster2 = create_sample_raster(width=15, height=15)
    raster_ref = create_sample_raster(width=13, height=13)
    rasters = [raster1, raster2]
    aligned_rasters = raster_align_to_reference(rasters, reference=raster_ref)

    assert len(aligned_rasters) == len(rasters)
    assert check_rasters_are_aligned(aligned_rasters + [raster_ref])

    for aligned_raster in aligned_rasters:
        assert gdal.Open(aligned_raster) is not None
        gdal.Unlink(aligned_raster)

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)
    gdal.Unlink(raster_ref)


def test_align_rasters_different_projections():
    """ Test: align_rasters when rasters have different projections """
    raster1 = create_sample_raster(width=10, height=10, epsg_code=4326)
    raster2 = create_sample_raster(width=15, height=15, epsg_code=32632)
    raster_ref = create_sample_raster(width=15, height=15, epsg_code=4326)
    rasters = [raster1, raster2]
    aligned_rasters = raster_align_to_reference(rasters, reference=raster_ref)

    assert len(aligned_rasters) == len(rasters)
    assert check_rasters_are_aligned(aligned_rasters)

    for aligned_raster in aligned_rasters:
        assert gdal.Open(aligned_raster) is not None
        gdal.Unlink(aligned_raster)

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)
    gdal.Unlink(raster_ref)


def test_align_rasters_multiple_rasters():
    """ Test: align_rasters with multiple rasters """
    raster1 = create_sample_raster(width=10, height=10)
    raster2 = create_sample_raster(width=15, height=16)
    raster3 = create_sample_raster(width=13, height=13)
    rasters = [raster1, raster2, raster3]
    aligned_rasters = raster_align(rasters)

    assert len(aligned_rasters) == len(rasters)
    assert check_rasters_are_aligned(aligned_rasters)

    metadata = raster_to_metadata(aligned_rasters[0])

    assert metadata["width"] == 15
    assert metadata["height"] == 16

    for aligned_raster in aligned_rasters:
        assert gdal.Open(aligned_raster) is not None
        gdal.Unlink(aligned_raster)

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)
    gdal.Unlink(raster3)


def test_align_rasters_multiple_rasters_no_overlap():
    """ Test: align_rasters with multiple rasters """
    raster1 = create_sample_raster(width=10, height=10, x_min=25)
    raster2 = create_sample_raster(width=15, height=16)
    raster3 = create_sample_raster(width=13, height=13)
    rasters = [raster1, raster2, raster3]

    with pytest.raises(AssertionError):
        raster_align(rasters)

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)
    gdal.Unlink(raster3)


def test_align_rasters_same_size_and_projection():
    """ Test: align_rasters when rasters have the same size and projection """
    raster1 = create_sample_raster(width=10, height=10)
    raster2 = create_sample_raster(width=10, height=10)
    raster_ref = create_sample_raster(width=10, height=10)
    rasters = [raster1, raster2]
    aligned_rasters = raster_align_to_reference(rasters, reference=raster_ref)

    assert len(aligned_rasters) == len(rasters)
    assert check_rasters_are_aligned(aligned_rasters)

    for aligned_raster in aligned_rasters:
        assert gdal.Open(aligned_raster) is not None
        gdal.Unlink(aligned_raster)

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)
    gdal.Unlink(raster_ref)

def test_align_rasters_different_sizes_same_projection():
    """ Test: align_rasters when rasters have different sizes but the same projection """
    raster1 = create_sample_raster(width=10, height=10)
    raster2 = create_sample_raster(width=15, height=12)
    raster_ref = create_sample_raster(width=20, height=20)
    rasters = [raster1, raster2]
    aligned_rasters = raster_align_to_reference(rasters, reference=raster_ref)

    assert len(aligned_rasters) == len(rasters)
    assert check_rasters_are_aligned(aligned_rasters)

    for aligned_raster in aligned_rasters:
        assert gdal.Open(aligned_raster) is not None
        gdal.Unlink(aligned_raster)

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)
    gdal.Unlink(raster_ref)

def test_align_rasters_different_epsg_codes():
    """ Test: align_rasters when rasters have different EPSG codes """
    raster1 = create_sample_raster(width=10, height=10, epsg_code=4326)
    raster2 = create_sample_raster(width=10, height=10, epsg_code=32632)
    raster_ref = create_sample_raster(width=10, height=10, epsg_code=4326)
    rasters = [raster1, raster2]
    aligned_rasters = raster_align_to_reference(rasters, reference=raster_ref)

    assert len(aligned_rasters) == len(rasters)
    assert check_rasters_are_aligned(aligned_rasters)

    for aligned_raster in aligned_rasters:
        assert gdal.Open(aligned_raster) is not None
        gdal.Unlink(aligned_raster)

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)
    gdal.Unlink(raster_ref)
