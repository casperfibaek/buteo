# pylint: skip-file
# type: ignore

# Standard library
import sys; sys.path.append("../../")

import pytest
import numpy as np
from osgeo import gdal
from pathlib import Path

from buteo.core_raster.core_raster_iterator import raster_to_array_iterator
from buteo.core_raster.core_raster_iterator import raster_to_array_iterator_random

@pytest.fixture
def sample_raster(tmp_path):
    """Create a sample raster for patch testing."""
    raster_path = tmp_path / "test_raster.tif"
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(str(raster_path), 100, 100, 3, gdal.GDT_Float32)
    ds.SetGeoTransform([0, 1, 0, 0, 0, -1])
    
    # Create distinct patterns for each band
    for i in range(3):
        band = ds.GetRasterBand(i + 1)
        data = np.full((100, 100), i + 1, dtype=np.float32)
        band.WriteArray(data)
        if i == 0:  # Set nodata for first band
            band.SetNoDataValue(-9999)
            data[0:10, 0:10] = -9999
            band.WriteArray(data)
    
    ds.FlushCache()
    ds = None
    return str(raster_path)

class TestRasterToArrayPatches:
    def test_basic_patches(self, sample_raster):
        """Test basic patch extraction functionality."""
        patches = raster_to_array_iterator(sample_raster, patches=4)
        
        # Check total patches
        assert len(patches) == 4
        
        # Iterate and verify patches
        for patch, offsets in patches:
            assert patch.shape == (3, 50, 50)
            assert len(offsets) == 4
            assert all(isinstance(x, int) for x in offsets)

    def test_fixed_patch_size(self, sample_raster):
        """Test extraction with fixed patch size."""
        patch_size = [20, 20]
        patches = raster_to_array_iterator(sample_raster, patch_size=patch_size)
        
        for patch, offsets in patches:
            assert patch.shape == (3, 20, 20)
            _x_start, _y_start, x_size, y_size = offsets
            assert x_size == 20
            assert y_size == 20

    def test_overlap(self, sample_raster):
        """Test patches with overlap."""
        overlap = 5
        patches = raster_to_array_iterator(
            sample_raster,
            patches=2,
            overlap=overlap
        )
        
        for i, (_patch, offsets) in enumerate(patches):
            if i == 0:
                assert offsets == (0, 0, 53, 100)
            elif i == 1:
                assert offsets == (47, 0, 53, 100)
            else:
                raise ValueError("Unexpected patch index.")

    def test_band_selection(self, sample_raster):
        """Test patch extraction with specific band selection."""
        # Test single band
        patches = raster_to_array_iterator(
            sample_raster,
            patches=2,
            bands=1
        )
        for patch, _ in patches:
            assert patch.shape[0] == 1
        
        # Test multiple bands
        patches = raster_to_array_iterator(
            sample_raster,
            patches=2,
            bands=[1, 3]
        )
        for patch, _ in patches:
            assert patch.shape[0] == 2

    def test_border_strategies(self, sample_raster):
        """Test different border strategies."""
        patch_size = [30, 30]  # Will create uneven patches with 100x100 raster
        
        # Strategy 1: Ignore border patches
        patches1 = raster_to_array_iterator(
            sample_raster,
            patch_size=patch_size,
            border_strategy=1
        )
        count1 = sum(1 for _ in patches1)
        
        # Strategy 2: Oversample border patches
        patches2 = raster_to_array_iterator(
            sample_raster,
            patch_size=patch_size,
            border_strategy=2
        )
        count2 = sum(1 for _ in patches2)
        
        # Strategy 3: Shrink border patches
        patches3 = raster_to_array_iterator(
            sample_raster,
            patch_size=patch_size,
            border_strategy=3
        )
        count3 = sum(1 for _ in patches3)

        assert count1 == 9  # 9 normal patches
        assert count2 == 16 # 9 normal patches + 7 border patches
        assert count3 == 16 # 9 normal patches + 7 border patches


class TestRasterToArrayIteratorRandom:
    def test_basic_random_patches(self, sample_raster):
        """Test basic random patch extraction functionality."""
        patch_size = (10, 10)
        max_iter = 5
        patches = raster_to_array_iterator_random(sample_raster, patch_size, max_iter=max_iter)
        
        # Check total patches
        assert len(patches) == max_iter
        
        # Iterate and verify patches
        for patch, offsets in patches:
            assert patch.shape == (3, 10, 10)
            assert len(offsets) == 4
            x_offset, y_offset, x_size, y_size = offsets
            assert 0 <= x_offset <= 90
            assert 0 <= y_offset <= 90
            assert x_size == 10
            assert y_size == 10

    def test_random_patch_limits(self, sample_raster):
        """Test that random patches do not exceed raster boundaries."""
        patch_size = (50, 50)
        max_iter = 100
        patches = raster_to_array_iterator_random(sample_raster, patch_size, max_iter=max_iter)
        
        for patch, offsets in patches:
            x_offset, y_offset, x_size, y_size = offsets
            assert x_offset + x_size <= 100
            assert y_offset + y_size <= 100

    def test_random_patch_shape(self, sample_raster):
        """Test random patches with different patch sizes."""
        patch_sizes = [(5, 5), (25, 25), (100, 100)]
        max_iter = 3
        for size in patch_sizes:
            patches = raster_to_array_iterator_random(sample_raster, size, max_iter=max_iter)
            for patch, offsets in patches:
                assert patch.shape == (3, size[0], size[1])

    def test_random_patch_band_selection(self, sample_raster):
        """Test random patch extraction with specific band selection."""
        patch_size = (20, 20)
        max_iter = 2
        
        # Single band
        patches = raster_to_array_iterator_random(
            sample_raster, patch_size, max_iter=max_iter, bands=2
        )
        for patch, _ in patches:
            assert patch.shape[0] == 1
        
        # Multiple bands
        patches = raster_to_array_iterator_random(
            sample_raster, patch_size, max_iter=max_iter, bands=[1, 3]
        )
        for patch, _ in patches:
            assert patch.shape[0] == 2

    def test_random_patch_filled(self, sample_raster):
        """Test random patch extraction with filled masked values."""
        patch_size = (10, 10)
        max_iter = 3
        patches = raster_to_array_iterator_random(
            sample_raster,
            patch_size,
            max_iter=max_iter,
            filled=True,
            fill_value=0
        )
        for patch, _ in patches:
            assert not np.isnan(patch).any()

    def test_random_patch_casting(self, sample_raster):
        """Test random patch extraction with casting to a different data type."""
        patch_size = (15, 15)
        max_iter = 2
        patches = raster_to_array_iterator_random(
            sample_raster,
            patch_size,
            max_iter=max_iter,
            cast='int16'
        )
        for patch, _ in patches:
            assert patch.dtype == np.int16

    def test_random_patch_invalid_patch_size(self, sample_raster):
        """Test that invalid patch sizes raise appropriate errors."""
        # Patch size larger than raster
        patch_size = (150, 150)
        with pytest.raises(ValueError):
            list(raster_to_array_iterator_random(sample_raster, patch_size, max_iter=1))
        
        # Negative patch size
        patch_size = (-10, 10)
        with pytest.raises(ValueError):
            list(raster_to_array_iterator_random(sample_raster, patch_size, max_iter=1))

    def test_random_patch_max_iter_zero(self, sample_raster):
        """Test that max_iter=0 results in no patches."""
        patch_size = (10, 10)
        max_iter = 0
        patches = raster_to_array_iterator_random(sample_raster, patch_size, max_iter=max_iter)
        assert len(patches) == 0
        with pytest.raises(StopIteration):
            next(iter(patches))
