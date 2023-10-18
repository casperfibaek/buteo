""" Tests for core_raster.py """
# pylint: disable=missing-function-docstring, bare-except

# Standard library
import sys; sys.path.append("../")

# External
from osgeo import gdal, osr

# Internal
from utils_tests import create_sample_raster
from buteo.raster.reproject import raster_reproject, _find_common_projection, _raster_reproject
from buteo.utils import utils_projection


def test_match_raster_projections_same_projection():
    """Test: match_raster_projections when rasters have the same projection as master"""
    raster1 = create_sample_raster(width=10, height=10, x_min=0, y_max=10, epsg_code=4326)
    raster2 = create_sample_raster(width=10, height=10, x_min=0, y_max=10, epsg_code=4326)
    rasters = [raster1, raster2]
    master = create_sample_raster(width=10, height=10, x_min=0, y_max=10, epsg_code=4326)

    matched_rasters = raster_reproject(rasters, master)

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

    matched_rasters = raster_reproject(rasters, master)

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

    matched_rasters = raster_reproject(rasters, master)

    for matched_raster in matched_rasters:
        assert isinstance(matched_raster, str)
        raster_ds = gdal.Open(matched_raster)
        assert raster_ds.GetProjection() == gdal.Open(master).GetProjection()

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)
    gdal.Unlink(master)

def test_find_common_projection():
    # Create sample raster files
    raster_paths = [create_sample_raster() for _ in range(2)]

    common_projection = _find_common_projection(raster_paths)
    assert common_projection.GetAuthorityCode(None) == "4326", "Common projection not found correctly."

def test_raster_reproject():
    # Create a sample raster file
    raster_path = create_sample_raster()

    # Test raster reproject
    reprojected_raster_path = _raster_reproject(
        raster=raster_path,
        projection=4326,
        out_path=None,
        resample_alg="nearest",
        copy_if_same=True,
        overwrite=True,
        creation_options=None,
        dst_nodata="infer",
        dtype=None,
        prefix="",
        suffix="reprojected",
        add_uuid=False,
        add_timestamp=True,
        memory=0.8,
    )
    reprojected_raster = gdal.Open(reprojected_raster_path)
    assert reprojected_raster is not None, "Reprojected raster is not created."

    srs = osr.SpatialReference()
    srs.ImportFromWkt(reprojected_raster.GetProjection())
    assert srs.GetAuthorityCode(None) == "4326", "Projection is not set correctly."

def test_raster_reproject_different_projection():
    raster_path = create_sample_raster(epsg_code=3857)
    reprojected_raster_path = _raster_reproject(raster_path, projection=4326)

    reprojected_raster = gdal.Open(reprojected_raster_path)
    assert reprojected_raster is not None, "Reprojected raster is not created."

    srs = osr.SpatialReference()
    srs.ImportFromWkt(reprojected_raster.GetProjection())
    assert srs.GetAuthorityCode(None) == "4326", "Projection is not set correctly."

def test_raster_reproject_no_copy_same_projection():
    raster_path = create_sample_raster(epsg_code=3857)
    reprojected_raster_path = _raster_reproject(raster_path, projection=3857, copy_if_same=False)

    assert raster_path == reprojected_raster_path, "Output path should be the same as the input path when copy_if_same is False and projections match."

def test_raster_reproject_copy_same_projection():
    raster_path = create_sample_raster(epsg_code=3857)
    reprojected_raster_path = _raster_reproject(raster_path, projection=3857, copy_if_same=True)

    assert raster_path != reprojected_raster_path, "Output path should be different from the input path when copy_if_same is True."

def test_raster_reproject_resample_alg():
    raster_path = create_sample_raster(epsg_code=3857)
    reprojected_raster_path = _raster_reproject(raster_path, projection=4326, resample_alg="bilinear")

    reprojected_raster = gdal.Open(reprojected_raster_path)
    assert reprojected_raster is not None, "Reprojected raster is not created."

    srs = osr.SpatialReference()
    srs.ImportFromWkt(reprojected_raster.GetProjection())
    assert srs.GetAuthorityCode(None) == "4326", "Projection is not set correctly."

def test_raster_reproject_dtype():
    raster_path = create_sample_raster(epsg_code=3857, datatype=gdal.GDT_Int16)
    reprojected_raster_path = _raster_reproject(raster_path, projection=4326, dtype="Int32")

    reprojected_raster = gdal.Open(reprojected_raster_path)
    assert reprojected_raster is not None, "Reprojected raster is not created."

    raster_band = reprojected_raster.GetRasterBand(1)
    assert raster_band.DataType == gdal.GDT_Int32, "Data type is not set correctly."

def test_get_utm_zone_from_latlng():
    longitudes = [-7,-6,-3,-1,0,1,3,6,7]
    latitudes = [-1,0,1]

    for long in longitudes:
        for lat in latitudes:
            if long < -6:
                zone = '29'
            elif long >= -6 and 0 > long:
                zone = '30'
            elif long >= 0 and 6 > long:
                zone = '31'
            elif long >= 6:
                zone = '32'
            if lat >= 0:
                ns = '6'
            else:
                ns = '7'

            epsg = utils_projection._get_utm_zone_from_latlng([lat,long],return_epsg=True)
            assert epsg.lower() == f'32{ns}{zone}', f"UTM zone for latlong point is wrong {lat}, {long}."
                
