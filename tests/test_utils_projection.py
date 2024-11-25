# pylint: skip-file
# type: ignore

# Standard library
import sys; sys.path.append("../")

import pytest
from osgeo import osr, gdal, ogr
from buteo.utils import utils_projection


# Sample fixtures
@pytest.fixture
def sample_wkt():
    """Returns a sample WKT projection string (EPSG:4326)."""
    return 'GEOGCS["WGS 84",DATUM["WGS_1984",' \
           'SPHEROID["WGS 84",6378137,298.257223563]],' \
           'PRIMEM["Greenwich",0],' \
           'UNIT["degree",0.0174532925199433],' \
           'AUTHORITY["EPSG","4326"]]'

@pytest.fixture
def sample_osr():
    """Returns a sample OSR SpatialReference object (EPSG:4326)."""
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    return srs

@pytest.fixture
def sample_dataset(tmp_path):
    """Creates a sample raster dataset with EPSG:4326 projection."""
    filename = str(tmp_path / "sample.tif")
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(filename, 100, 100, 1, gdal.GDT_Byte)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    ds.SetGeoTransform([0, 1, 0, 0, 0, -1])
    ds.GetRasterBand(1).Fill(0)
    ds = None
    return filename

@pytest.fixture
def sample_vector(tmp_path):
    """Creates a sample vector dataset with EPSG:32633 projection."""
    filename = str(tmp_path / "sample.shp")
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.CreateDataSource(filename)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32633)
    layer = ds.CreateLayer("layer", srs, ogr.wkbPoint)
    feature_def = layer.GetLayerDefn()
    feature = ogr.Feature(feature_def)
    wkt = "POINT (0 0)"
    geometry = ogr.CreateGeometryFromWkt(wkt)
    feature.SetGeometry(geometry)
    layer.CreateFeature(feature)
    feature = None
    ds = None
    return filename

# Fixtures
@pytest.fixture
def sample_bbox():
    """Returns a sample bounding box in OGR format [x_min, x_max, y_min, y_max]."""
    return [0, 10, 0, 10]

@pytest.fixture
def sample_point():
    """Returns a sample point [x, y]."""
    return [12.0, 55.0]  # Coordinates near Copenhagen

@pytest.fixture
def wgs84_srs():
    """Returns an OSR SpatialReference object for EPSG:4326."""
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    return srs

@pytest.fixture
def utm_zone_srs():
    """Returns an OSR SpatialReference object for UTM zone 32N (EPSG:32632)."""
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32632)
    return srs

@pytest.fixture
def sample_raster(tmp_path):
    """Creates a sample raster dataset."""
    filepath = str(tmp_path / "sample.tif")
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(filepath, 100, 100, 1, gdal.GDT_Byte)
    ds.SetGeoTransform([0, 1, 0, 0, 0, -1])
    ds.GetRasterBand(1).Fill(0)
    ds.FlushCache()
    ds = None
    return filepath

# Test _get_default_projection
def test_get_default_projection():
    wkt = utils_projection._get_default_projection()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    assert srs.IsGeographic() == 1
    assert srs.GetAuthorityCode(None) == '4326'

# Test _get_default_projection_osr
def test_get_default_projection_osr():
    srs = utils_projection._get_default_projection_osr()
    assert isinstance(srs, osr.SpatialReference)
    assert srs.IsGeographic() == 1
    assert srs.GetAuthorityCode(None) == '4326'

# Test _get_pseudo_mercator_projection
def test_get_pseudo_mercator_projection():
    wkt = utils_projection._get_pseudo_mercator_projection()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    assert srs.IsProjected() == 1
    assert srs.GetAuthorityCode(None) == '3857'

# Test _get_pseudo_mercator_projection_osr
def test_get_pseudo_mercator_projection_osr():
    srs = utils_projection._get_pseudo_mercator_projection_osr()
    assert isinstance(srs, osr.SpatialReference)
    assert srs.IsProjected() == 1
    assert srs.GetAuthorityCode(None) == '3857'

# Test _get_esri_projection
def test_get_esri_projection():
    wkt = utils_projection._get_esri_projection('ESRI:54009')
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    assert srs.Validate() == 0
    projcs = srs.GetAttrValue("PROJCS")
    assert projcs == "World_Mollweide"

# Test parse_projection
def test_parse_projection_epsg():
    srs = utils_projection.parse_projection(4326)
    assert srs.IsGeographic() == 1
    assert srs.GetAuthorityCode(None) == '4326'

def test_parse_projection_wkt(sample_wkt):
    srs = utils_projection.parse_projection(sample_wkt)
    assert srs.IsGeographic() == 1
    assert srs.GetAuthorityCode(None) == '4326'

def test_parse_projection_osr(sample_osr):
    srs = utils_projection.parse_projection(sample_osr)
    assert srs.IsSame(sample_osr) == 1

def test_parse_projection_dataset(sample_dataset):
    ds = gdal.Open(sample_dataset)
    srs = utils_projection.parse_projection(ds)
    assert srs.GetAuthorityCode(None) == '4326'
    ds = None

def test_parse_projection_invalid():
    with pytest.raises(ValueError):
        utils_projection.parse_projection(None)

# Test parse_projection_wkt
def test_parse_projection_wkt(sample_osr):
    wkt = utils_projection.parse_projection_wkt(sample_osr)
    assert isinstance(wkt, str)
    assert 'WGS 84' in wkt

# Test _projection_is_latlng
def test_projection_is_latlng(sample_osr):
    result = utils_projection._projection_is_latlng(sample_osr)
    assert result is True

def test_projection_is_latlng_projected():
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    result = utils_projection._projection_is_latlng(srs)
    assert result is False

# Test _check_projections_match
def test_check_projections_match(sample_osr):
    srs1 = sample_osr
    srs2 = osr.SpatialReference()
    srs2.ImportFromEPSG(4326)
    assert utils_projection._check_projections_match(srs1, srs2) is True

def test_check_projections_mismatch():
    srs1 = osr.SpatialReference()
    srs1.ImportFromEPSG(4326)
    srs2 = osr.SpatialReference()
    srs2.ImportFromEPSG(3857)
    assert utils_projection._check_projections_match(srs1, srs2) is False

# Test _get_projection_from_raster
def test_get_projection_from_raster(sample_dataset):
    srs = utils_projection._get_projection_from_raster(sample_dataset)
    assert srs.GetAuthorityCode(None) == '4326'

# Test _get_projection_from_vector
def test_get_projection_from_vector(sample_vector):
    srs = utils_projection._get_projection_from_vector(sample_vector)
    assert srs.GetAuthorityCode(None) == '32633'

# Test reproject_bbox
def test_reproject_bbox():
    bbox = [-10, -10, 10, 10]  # xmin, ymin, xmax, ymax
    source_srs = osr.SpatialReference()
    source_srs.ImportFromEPSG(4326)
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(3857)

    new_bbox = utils_projection.reproject_bbox(bbox, source_srs, target_srs)
    assert len(new_bbox) == 4
    # Check that coordinates have been transformed (roughly)
    assert float(new_bbox[0]) != float(bbox[0])
    assert float(new_bbox[1]) != float(bbox[1])

# Test _reproject_point
def test_reproject_point():
    point = [-10, 10]  # lon, lat
    source_srs = osr.SpatialReference()
    source_srs.ImportFromEPSG(4326)
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(3857)

    new_point = utils_projection._reproject_point(point, source_srs, target_srs)
    assert len(new_point) == 2
    # Check that coordinates have been transformed
    assert float(new_point[0]) != float(point[0])
    assert float(new_point[1]) != float(point[1])

# Test _get_utm_epsg_from_latlng
def test_get_utm_epsg_from_latlng():
    lat = 55.495972
    lng = 9.473052
    epsg_code = utils_projection._get_utm_epsg_from_latlng([lat, lng])
    assert epsg_code == '32632'  # UTM zone 32N

# Test _get_transformer
def test_get_transformer():
    source_srs = osr.SpatialReference()
    source_srs.ImportFromEPSG(4326)
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(3857)
    transformer = utils_projection._get_transformer(source_srs, target_srs)
    assert isinstance(transformer, osr.CoordinateTransformation)

# Test set_projection
def test_set_projection(tmp_path):
    # Create a raster without projection
    filename = str(tmp_path / "unproj.tif")
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(filename, 100, 100, 1, gdal.GDT_Byte)
    ds.SetGeoTransform([0, 1, 0, 0, 0, -1])
    ds.GetRasterBand(1).Fill(0)
    ds.FlushCache()
    ds = None

    # Verify no projection
    ds = gdal.Open(filename)
    assert ds.GetProjection() == ''
    ds = None

    # Set projection
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    result = utils_projection.set_projection_raster(filename, srs)
    assert result is True

    # Verify projection set
    ds = gdal.Open(filename)
    assert ds.GetProjection() != ''
    srs_result = osr.SpatialReference()
    srs_result.ImportFromWkt(ds.GetProjection())
    assert srs_result.GetAuthorityCode(None) == '4326'
    ds = None

# Tests for reproject_bbox
def test_reproject_bbox_valid(sample_bbox, wgs84_srs, utm_zone_srs):
    bbox = sample_bbox  # [0, 10, 0, 10]
    reprojected_bbox = utils_projection.reproject_bbox(
        bbox,
        source_projection=wgs84_srs,
        target_projection=utm_zone_srs
    )
    assert len(reprojected_bbox) == 4
    # Ensure that coordinates have changed
    assert reprojected_bbox != bbox

def test_reproject_bbox_invalid_bbox():
    # Invalid bbox (not a list)
    with pytest.raises(ValueError):
        utils_projection.reproject_bbox(
            bbox_ogr=None,
            source_projection=4326,
            target_projection=3857
        )
    # Invalid bbox length
    with pytest.raises(ValueError):
        utils_projection.reproject_bbox(
            bbox_ogr=[0, 10, 0],  # Only 3 elements
            source_projection=4326,
            target_projection=3857
        )
    # Non-numeric values
    with pytest.raises(ValueError):
        utils_projection.reproject_bbox(
            bbox_ogr=[0, 10, 'a', 10],
            source_projection=4326,
            target_projection=3857
        )

def test_reproject_bbox_invalid_projections(sample_bbox):
    # Invalid source projection
    with pytest.raises(ValueError):
        utils_projection.reproject_bbox(
            bbox_ogr=sample_bbox,
            source_projection=None,
            target_projection=3857
        )
    # Invalid target projection
    with pytest.raises(ValueError):
        utils_projection.reproject_bbox(
            bbox_ogr=sample_bbox,
            source_projection=4326,
            target_projection=None
        )

def test_reproject_bbox_same_projection(sample_bbox):
    # Reprojecting to the same projection should return the original bbox
    result_bbox = utils_projection.reproject_bbox(
        bbox_ogr=sample_bbox,
        source_projection=4326,
        target_projection=4326
    )
    assert result_bbox == sample_bbox

def test_reproject_bbox_invalid_bbox_order():
    # x_min > x_max
    with pytest.raises(ValueError):
        utils_projection.reproject_bbox(
            bbox_ogr=[10, 0, 0, 10],
            source_projection=4326,
            target_projection=3857
        )
    # y_min > y_max
    with pytest.raises(ValueError):
        utils_projection.reproject_bbox(
            bbox_ogr=[0, 10, 10, 0],
            source_projection=4326,
            target_projection=3857
        )

# Tests for _reproject_point
def test_reproject_point_valid(sample_point, wgs84_srs, utm_zone_srs):
    point = sample_point  # [12.0, 55.0]
    reprojected_point = utils_projection._reproject_point(
        point,
        source_projection=wgs84_srs,
        target_projection=utm_zone_srs
    )
    assert len(reprojected_point) == 2
    # Ensure that coordinates have changed
    assert reprojected_point != point

def test_reproject_point_invalid_point():
    # Point is None
    with pytest.raises(ValueError):
        utils_projection._reproject_point(
            p=None,
            source_projection=4326,
            target_projection=3857
        )
    # Point is not a list or tuple
    with pytest.raises(ValueError):
        utils_projection._reproject_point(
            p="invalid",
            source_projection=4326,
            target_projection=3857
        )
    # Point has incorrect length
    with pytest.raises(ValueError):
        utils_projection._reproject_point(
            p=[0, 1, 2],
            source_projection=4326,
            target_projection=3857
        )
    # Point contains non-numeric values
    with pytest.raises(ValueError):
        utils_projection._reproject_point(
            p=[0, "a"],
            source_projection=4326,
            target_projection=3857
        )

def test_reproject_point_same_projection(sample_point):
    # Reprojecting to the same projection should return the original point
    result_point = utils_projection._reproject_point(
        p=sample_point,
        source_projection=4326,
        target_projection=4326
    )
    assert result_point == [float(coord) for coord in sample_point]

def test_reproject_point_invalid_projections(sample_point):
    # Invalid source projection
    with pytest.raises(ValueError):
        utils_projection._reproject_point(
            p=sample_point,
            source_projection=None,
            target_projection=3857
        )
    # Invalid target projection
    with pytest.raises(ValueError):
        utils_projection._reproject_point(
            p=sample_point,
            source_projection=4326,
            target_projection=None
        )

def test_get_utm_epsg_from_latlng_invalid_latlng():
    # latlng is None
    with pytest.raises(ValueError):
        utils_projection._get_utm_epsg_from_latlng(None)
    # latlng is not list or array
    with pytest.raises(ValueError):
        utils_projection._get_utm_epsg_from_latlng("invalid")
    # latlng has incorrect length
    with pytest.raises(ValueError):
        utils_projection._get_utm_epsg_from_latlng([55.0])
    # latlng contains non-numeric values
    with pytest.raises(ValueError):
        utils_projection._get_utm_epsg_from_latlng(["a", "b"])
    # Latitude out of range
    with pytest.raises(ValueError):
        utils_projection._get_utm_epsg_from_latlng([91.0, 12.0])
    # Longitude out of range
    with pytest.raises(ValueError):
        utils_projection._get_utm_epsg_from_latlng([55.0, 181.0])

# Tests for _reproject_latlng_point_to_utm
def test_reproject_latlng_point_to_utm_valid():
    latlng = [55.0, 12.0]
    utm_coords = utils_projection._reproject_latlng_point_to_utm(latlng)
    assert len(utm_coords) == 2
    # Ensure coordinates are numeric
    assert all(isinstance(coord, (int, float)) for coord in utm_coords)

def test_reproject_latlng_point_to_utm_invalid_latlng():
    # latlng is None
    with pytest.raises(ValueError):
        utils_projection._reproject_latlng_point_to_utm(None)
    # latlng is not list or array
    with pytest.raises(ValueError):
        utils_projection._reproject_latlng_point_to_utm("invalid")
    # latlng has incorrect length
    with pytest.raises(ValueError):
        utils_projection._reproject_latlng_point_to_utm([55.0])
    # latlng contains non-numeric values
    with pytest.raises(ValueError):
        utils_projection._reproject_latlng_point_to_utm(["a", "b"])
    # Latitude out of range
    with pytest.raises(ValueError):
        utils_projection._reproject_latlng_point_to_utm([91.0, 12.0])
    # Longitude out of range
    with pytest.raises(ValueError):
        utils_projection._reproject_latlng_point_to_utm([55.0, 181.0])

# Tests for set_projection
def test_set_projection_raster(sample_raster, wgs84_srs):
    # Ensure the raster has no projection
    ds = gdal.Open(sample_raster)
    assert ds.GetProjection() == ''
    ds = None

    # Set projection
    result = utils_projection.set_projection_raster(sample_raster, wgs84_srs)

    assert result is True

    # Verify projection is set
    ds = gdal.Open(sample_raster)
    assert ds.GetProjection() != ''
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    assert srs.IsSame(wgs84_srs)
    ds = None

def test_set_projection_invalid_inputs():
    # dataset is None
    with pytest.raises(ValueError):
        utils_projection.set_projection_raster(
            dataset=None,
            projection=4326
        )
    # projection is None
    with pytest.raises(ValueError):
        utils_projection.set_projection_raster(
            dataset="sample.tif",
            projection=None
        )
    # dataset invalid type
    with pytest.raises(ValueError):
        utils_projection.set_projection_raster(
            dataset=123,
            projection=4326
        )
    # projection invalid type
    with pytest.raises(ValueError):
        utils_projection.set_projection_raster(
            dataset="sample.tif",
            projection=123.456
        )

# Tests for _get_transformer
def test_get_transformer_valid(wgs84_srs, utm_zone_srs):
    transformer = utils_projection._get_transformer(wgs84_srs, utm_zone_srs)
    assert isinstance(transformer, osr.CoordinateTransformation)

def test_get_transformer_invalid_inputs():
    # proj_source is None
    with pytest.raises(ValueError):
        utils_projection._get_transformer(
            proj_source=None,
            proj_target=4326
        )
    # proj_target is None
    with pytest.raises(ValueError):
        utils_projection._get_transformer(
            proj_source=4326,
            proj_target=None
        )

def test_get_transformer_same_projection(wgs84_srs):
    transformer = utils_projection._get_transformer(wgs84_srs, wgs84_srs)
    assert isinstance(transformer, osr.CoordinateTransformation)
