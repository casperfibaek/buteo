# pylint: skip-file
# type: ignore

# Standard library
import sys; sys.path.append("../../")

import pytest
from buteo.core_raster.core_raster_offsets import _find_optimal_chunk_factors, _get_chunk_offsets, _get_chunk_offsets_fixed_size, _compute_chunk_positions

# Fixtures
@pytest.fixture
def sample_image_shape():
    return (3, 100, 200)  # (channels, height, width)

@pytest.fixture
def small_image_shape():
    return (1, 10, 20)

# Test _find_optimal_chunk_factors
class TestFindOptimalChunkFactors:
    def test_basic_functionality(self):
        h_chunks, w_chunks = _find_optimal_chunk_factors(4, 100, 100)
        assert h_chunks * w_chunks == 4
        assert isinstance(h_chunks, int)
        assert isinstance(w_chunks, int)

    def test_single_chunk(self):
        result = _find_optimal_chunk_factors(1, 100, 100)
        assert result == (1, 1)

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            _find_optimal_chunk_factors(0, 100, 100)
        with pytest.raises(TypeError):
            _find_optimal_chunk_factors(1.5, 100, 100)

    def test_rectangular_image(self):
        result = _find_optimal_chunk_factors(4, 200, 100)
        assert result == (2, 2)

# Test _get_chunk_offsets
class TestGetChunkOffsets:
    def test_basic_functionality(self, sample_image_shape):
        offsets = _get_chunk_offsets(sample_image_shape, 4)
        assert len(offsets) == 4
        assert all(len(offset) == 4 for offset in offsets)

    def test_with_overlap(self, sample_image_shape):
        offsets = _get_chunk_offsets(sample_image_shape, 4, overlap=10)
        assert len(offsets) == 4
        for x_start, y_start, x_size, y_size in offsets:
            assert x_start >= 0
            assert y_start >= 0
            assert x_size > 0
            assert y_size > 0

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            _get_chunk_offsets((1, 100), 4)  # Invalid shape
        with pytest.raises(ValueError):
            _get_chunk_offsets((3, 100, 100), 4, overlap=-1)  # Negative overlap

# Test _compute_chunk_positions
class TestComputeChunkPositions:
    @pytest.mark.parametrize("border_strategy", [1, 2, 3])
    def test_basic_functionality(self, border_strategy):
        positions = _compute_chunk_positions(100, 30, 0, border_strategy)
        assert isinstance(positions, list)
        assert all(isinstance(pos, int) for pos in positions)
        assert all(pos >= 0 for pos in positions)

    def test_with_overlap(self):
        positions = _compute_chunk_positions(100, 30, 5, 1)
        assert len(positions) > 0
        assert positions[0] == 0

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            _compute_chunk_positions(100, 30, 30, 1)  # Overlap equals chunk size

# Test _get_chunk_offsets_fixed_size
class TestGetChunkOffsetsFixedSize:
    def test_basic_functionality(self, sample_image_shape):
        offsets = _get_chunk_offsets_fixed_size(sample_image_shape, 50, 50)
        assert isinstance(offsets, list)
        assert all(len(offset) == 4 for offset in offsets)

    @pytest.mark.parametrize("border_strategy", [1, 2, 3])
    def test_border_strategies(self, sample_image_shape, border_strategy):
        offsets = _get_chunk_offsets_fixed_size(
            sample_image_shape, 50, 50, border_strategy=border_strategy
        )
        assert len(offsets) > 0

    def test_with_overlap(self, sample_image_shape):
        offsets = _get_chunk_offsets_fixed_size(
            sample_image_shape, 50, 50, overlap=10
        )
        assert len(offsets) > 0

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            _get_chunk_offsets_fixed_size((3, 100, 100), 0, 50)  # Invalid chunk size
        with pytest.raises(ValueError):
            _get_chunk_offsets_fixed_size((3, 100, 100), 50, 50, border_strategy=4)  # Invalid strategy

    def test_chunk_consistency(self, sample_image_shape):
        """Test that chunk sizes are consistent when using border strategies 1 and 2"""
        chunk_size = 64
        offsets = _get_chunk_offsets_fixed_size(
            sample_image_shape, chunk_size, chunk_size, border_strategy=1
        )
        assert all(x_size == chunk_size and y_size == chunk_size 
                  for _, _, x_size, y_size in offsets)