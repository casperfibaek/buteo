# Standard library
import sys
import os
import pytest
import numpy as np
from osgeo import gdal, ogr, osr
from typing import List, Tuple, Sequence, Union

# Add the parent directory to sys.path to allow imports from the buteo package
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

# Type Aliases
BboxType = Sequence[Union[int, float]]
GeoTransformType = Sequence[Union[int, float]]

# Fixtures
@pytest.fixture
def sample_bbox_ogr() -> List[float]:
    """Sample OGR bbox: [x_min, x_max, y_min, y_max]."""
    return [0.0, 1.0, 0.0, 1.0]

@pytest.fixture
def sample_bbox_latlng() -> List[float]:
    """Sample lat/long bbox."""
    return [-10.0, 10.0, -10.0, 10.0]

@pytest.fixture
def sample_geotransform() -> List[float]:
    """Sample GDAL geotransform."""
    return [0.0, 1.0, 0.0, 10.0, 0.0, -1.0]

@pytest.fixture
def wgs84_srs() -> osr.SpatialReference:
    """WGS84 spatial reference fixture."""
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    return srs

@pytest.fixture
def utm32n_srs() -> osr.SpatialReference:
    """UTM 32N spatial reference fixture."""
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32632)
    return srs

@pytest.fixture
def create_temp_raster(tmp_path, utm32n_srs):
    """Factory fixture to create a temporary raster file."""
    def _create_raster(
        filename: str = "test_raster.tif",
        width: int = 10,
        height: int = 10,
        geotransform: GeoTransformType = (1000.0, 10.0, 0.0, 2000.0, 0.0, -10.0),
        srs: osr.SpatialReference = utm32n_srs,
        dtype: int = gdal.GDT_Byte,
        bands: int = 1,
    ) -> str:
        """Creates a temporary raster file."""
        out_path = tmp_path / filename
        driver = gdal.GetDriverByName("GTiff")
        ds = driver.Create(str(out_path), width, height, bands, dtype)
        if ds is None:
            raise IOError(f"Could not create temporary raster: {out_path}")
        ds.SetGeoTransform(geotransform)
        ds.SetProjection(srs.ExportToWkt())
        for i in range(1, bands + 1):
            band = ds.GetRasterBand(i)
            band.WriteArray(np.ones((height, width), dtype=np.uint8) * i)
            band.FlushCache()
        ds = None # Close dataset
        return str(out_path)
    return _create_raster

@pytest.fixture
def create_temp_vector(tmp_path, utm32n_srs):
    """Factory fixture to create a temporary vector file (GeoPackage)."""
    def _create_vector(
        filename: str = "test_vector.gpkg",
        bbox: BboxType = (1000.0, 1100.0, 1900.0, 2000.0), # Corresponds to 10x10 raster with GT above
        srs: osr.SpatialReference = utm32n_srs,
        layer_name: str = "test_layer",
    ) -> str:
        """Creates a temporary vector file with a single polygon feature."""
        out_path = tmp_path / filename
        driver = ogr.GetDriverByName("GPKG")
        if driver is None:
            raise RuntimeError("GPKG driver not available.")
        if out_path.exists():
            driver.DeleteDataSource(str(out_path))

        ds = driver.CreateDataSource(str(out_path))
        if ds is None:
            raise IOError(f"Could not create temporary vector: {out_path}")

        layer = ds.CreateLayer(layer_name, srs, ogr.wkbPolygon)
        if layer is None:
            raise RuntimeError("Could not create layer.")

        # Create polygon geometry from bbox
        x_min, x_max, y_min, y_max = bbox
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(x_min, y_min)
        ring.AddPoint(x_max, y_min)
        ring.AddPoint(x_max, y_max)
        ring.AddPoint(x_min, y_max)
        ring.AddPoint(x_min, y_min)
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)

        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(poly)
        if layer.CreateFeature(feature) != ogr.OGRERR_NONE:
             raise RuntimeError("Failed to create feature.")

        feature.Destroy()
        ds.FlushCache()
        ds = None # Close dataset
        return str(out_path)
    return _create_vector
