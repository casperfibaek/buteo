# pylint: skip-file
# type: ignore

# Standard library
import sys; sys.path.append("../../")

import pytest
import numpy as np
from osgeo import gdal
from pathlib import Path

from buteo.core_raster.core_raster_chunks import raster_to_array_chunks

@pytest.fixture
def sample_chunked_raster(tmp_path):
    """Create a sample raster for chunk testing."""
    raster_path = tmp_path / "chunked_raster.tif"
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

class TestRasterToArrayChunks:
    def test_basic_chunking(self, sample_chunked_raster):
        """Test basic chunking functionality."""
        chunks = raster_to_array_chunks(sample_chunked_raster, chunks=4)
        
        # Check total chunks, should always be == chunks
        assert len(chunks) == 4
        
        # Iterate and verify chunks
        for chunk, offsets in chunks:
            assert chunk.shape == (3, 50, 50)
            assert len(offsets) == 4
            assert all(isinstance(x, int) for x in offsets)

    def test_fixed_chunk_size(self, sample_chunked_raster):
        """Test chunking with fixed chunk size."""
        chunk_size = [20, 20]
        chunks = raster_to_array_chunks(sample_chunked_raster, chunk_size=chunk_size)
        
        for chunk, offsets in chunks:
            assert chunk.shape == (3, 20, 20)
            x_start, y_start, x_size, y_size = offsets
            assert x_size == 20
            assert y_size == 20

    def test_overlap(self, sample_chunked_raster):
        """Test chunking with overlap."""
        overlap = 5
        chunks = raster_to_array_chunks(
            sample_chunked_raster,
            chunks=2,
            overlap=overlap
        )
        
        for i, (_chunk, offsets) in enumerate(chunks):
            if i == 0:
                assert offsets == (0, 0, 53, 100)
            elif i == 1:
                assert offsets == (47, 0, 53, 100)
            else:
                raise ValueError("Unexpected chunk index.")

    def test_band_selection(self, sample_chunked_raster):
        """Test chunking with specific band selection."""
        # Test single band
        chunks = raster_to_array_chunks(
            sample_chunked_raster,
            chunks=2,
            bands=1
        )
        for chunk, _ in chunks:
            assert chunk.shape[0] == 1
        
        # Test multiple bands
        chunks = raster_to_array_chunks(
            sample_chunked_raster,
            chunks=2,
            bands=[1, 3]
        )
        for chunk, _ in chunks:
            assert chunk.shape[0] == 2

    def test_nodata_handling(self, sample_chunked_raster):
        """Test chunking with nodata handling."""
        chunks = raster_to_array_chunks(
            sample_chunked_raster,
            chunks=2,
            filled=True,
            fill_value=0
        )
        
        for chunk, offsets in chunks:
            if offsets[0] == 0 and offsets[1] == 0:
                # First chunk should have nodata filled
                assert np.all(chunk[0, :10, :10] == 0)

    def test_dtype_casting(self, sample_chunked_raster):
        """Test chunking with dtype casting."""
        chunks = raster_to_array_chunks(
            sample_chunked_raster,
            chunks=2,
            cast=np.int32
        )
        
        for chunk, _ in chunks:
            assert chunk.dtype == np.int32

    def test_skip_chunks(self, sample_chunked_raster):
        """Test skipping chunks."""
        chunks = raster_to_array_chunks(
            sample_chunked_raster,
            chunks=4,
            skip=1
        )
        
        count = sum(1 for _ in chunks)
        assert count == 2

    def test_border_strategies(self, sample_chunked_raster):
        """Test different border strategies."""
        chunk_size = [30, 30]  # Will create uneven chunks with 100x100 raster
        
        # Strategy 1: Ignore border chunks
        chunks1 = raster_to_array_chunks(
            sample_chunked_raster,
            chunk_size=chunk_size,
            border_strategy=1
        )
        count1 = sum(1 for _ in chunks1)
        
        # Strategy 2: Oversample border chunks
        chunks2 = raster_to_array_chunks(
            sample_chunked_raster,
            chunk_size=chunk_size,
            border_strategy=2
        )
        count2 = sum(1 for _ in chunks2)
        
        # Strategy 3: Shrink border chunks
        chunks3 = raster_to_array_chunks(
            sample_chunked_raster,
            chunk_size=chunk_size,
            border_strategy=3
        )
        count3 = sum(1 for _ in chunks3)

        assert count1 == 9 # 9 normal chunks
        assert count2 == 9 + 7 # 9 normal chunks + 7 border chunks
        assert count3 == 9 + 7 # 9 normal chunks + 7 border chunks

        # all chunks should have the same size (strategy 1 & 2)
        for chunk, _ in chunks1:
            assert chunk.shape == (3, 30, 30)
        for chunk, _ in chunks2:
            assert chunk.shape == (3, 30, 30)

        # last chunk should have different size (strategy 3)
        last_chunk, _ = list(chunks3)[-1]
        assert last_chunk.shape == (3, 10, 10)
    

    def test_invalid_inputs(self, sample_chunked_raster):
        """Test invalid input handling."""
        # Invalid chunks number
        with pytest.raises(ValueError):
            raster_to_array_chunks(sample_chunked_raster, chunks=0)
        
        # Invalid chunk size
        with pytest.raises(ValueError):
            raster_to_array_chunks(sample_chunked_raster, chunk_size=[0, 10])
        
        # Invalid overlap
        with pytest.raises(ValueError):
            raster_to_array_chunks(sample_chunked_raster, chunks=2, overlap=-1)
        
        # Invalid border strategy
        with pytest.raises(ValueError):
            raster_to_array_chunks(
                sample_chunked_raster,
                chunk_size=[10, 10],
                border_strategy=4
            )

    def test_iterator_behavior(self, sample_chunked_raster):
        """Test iterator behavior of chunks."""
        chunks = raster_to_array_chunks(sample_chunked_raster, chunks=2)
        
        # Test iteration multiple times
        first_run = [(c.shape, o) for c, o in chunks]
        second_run = [(c.shape, o) for c, o in chunks]

        assert first_run == second_run
        assert len(first_run) == 2
        assert first_run[0][0] == (3, 100, 50)
        assert first_run[1][0] == (3, 100, 50)
        assert first_run[0][1] == (0, 0, 50, 100)
        assert first_run[1][1] == (50, 0, 50, 100)
