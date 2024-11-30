# pylint: skip-file
# type: ignore

# Standard library
import sys; sys.path.append("../../")

import pytest
from buteo.core_raster.core_raster_offsets import _find_optimal_patch_factors, _get_patch_offsets, _get_patch_offsets_fixed_size, _compute_patch_positions

# Fixtures
@pytest.fixture
def sample_image_shape():
    return (3, 100, 200)  # (channels, height, width)

@pytest.fixture
def small_image_shape():
    return (1, 10, 20)

# Test _find_optimal_patch_factors
class TestFindOptimalPatchFactors:
    def test_basic_functionality(self):
        h_patches, w_patches = _find_optimal_patch_factors(4, 100, 100)
        assert h_patches * w_patches == 4
        assert isinstance(h_patches, int)
        assert isinstance(w_patches, int)

    def test_single_patch(self):
        result = _find_optimal_patch_factors(1, 100, 100)
        assert result == (1, 1)

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            _find_optimal_patch_factors(0, 100, 100)
        with pytest.raises(TypeError):
            _find_optimal_patch_factors(1.5, 100, 100)

    def test_rectangular_image(self):
        result = _find_optimal_patch_factors(4, 200, 100)
        assert result == (2, 2)

# Test _get_patch_offsets
class TestGetPatchOffsets:
    def test_basic_functionality(self, sample_image_shape):
        offsets = _get_patch_offsets(sample_image_shape, 4)
        assert len(offsets) == 4
        assert all(len(offset) == 4 for offset in offsets)

    def test_with_overlap(self, sample_image_shape):
        offsets = _get_patch_offsets(sample_image_shape, 4, overlap=10)
        assert len(offsets) == 4
        for x_start, y_start, x_size, y_size in offsets:
            assert x_start >= 0
            assert y_start >= 0
            assert x_size > 0
            assert y_size > 0

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            _get_patch_offsets((1, 100), 4)  # Invalid shape
        with pytest.raises(ValueError):
            _get_patch_offsets((3, 100, 100), 4, overlap=-1)  # Negative overlap

# Test _compute_patch_positions
class TestComputePatchPositions:
    @pytest.mark.parametrize("border_strategy", [1, 2, 3])
    def test_basic_functionality(self, border_strategy):
        positions = _compute_patch_positions(100, 30, 0, border_strategy)
        assert isinstance(positions, list)
        assert all(isinstance(pos, int) for pos in positions)
        assert all(pos >= 0 for pos in positions)

    def test_with_overlap(self):
        positions = _compute_patch_positions(100, 30, 5, 1)
        assert len(positions) > 0
        assert positions[0] == 0

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            _compute_patch_positions(100, 30, 30, 1)  # Overlap equals patch size

# Test _get_patch_offsets_fixed_size
class TestGetPatchOffsetsFixedSize:
    def test_basic_functionality(self, sample_image_shape):
        offsets = _get_patch_offsets_fixed_size(sample_image_shape, 50, 50)
        assert isinstance(offsets, list)
        assert all(len(offset) == 4 for offset in offsets)

    @pytest.mark.parametrize("border_strategy", [1, 2, 3])
    def test_border_strategies(self, sample_image_shape, border_strategy):
        offsets = _get_patch_offsets_fixed_size(
            sample_image_shape, 50, 50, border_strategy=border_strategy
        )
        assert len(offsets) > 0

    def test_with_overlap(self, sample_image_shape):
        offsets = _get_patch_offsets_fixed_size(
            sample_image_shape, 50, 50, overlap=10
        )
        assert len(offsets) > 0

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            _get_patch_offsets_fixed_size((3, 100, 100), 0, 50)  # Invalid patch size
        with pytest.raises(ValueError):
            _get_patch_offsets_fixed_size((3, 100, 100), 50, 50, border_strategy=4)  # Invalid strategy

    def test_patch_consistency(self, sample_image_shape):
        """Test that patch sizes are consistent when using border strategies 1 and 2"""
        patch_size = 64
        offsets = _get_patch_offsets_fixed_size(
            sample_image_shape, patch_size, patch_size, border_strategy=1
        )
        assert all(x_size == patch_size and y_size == patch_size 
                  for _, _, x_size, y_size in offsets)