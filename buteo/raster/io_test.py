""" Tests for io.py """

# Standard library
import os

# Dependencies
import numpy as np

# Local dependencies
from .io import raster_to_array


# Setup tests
folder = "geometry_and_rasters/"
s2_b04 = os.path.abspath(folder + "s2_b04.jp2")
s2_tci = os.path.abspath(folder + "s2_tci.jp2")


def test_image_paths():
    """Meta-test: Test if image paths are correct"""
    assert os.path.isfile(s2_b04)
    assert os.path.isfile(s2_tci)


# Start tests
def test_raster_to_array():
    b04_arr = raster_to_array(s2_b04)
    tci_arr = raster_to_array(s2_tci)

    assert isinstance(b04_arr, np.ndarray)
    assert isinstance(tci_arr, np.ndarray)

    assert b04_arr.shape == (1830, 1830, 1)
    assert tci_arr.shape == (1830, 1830, 3)

    assert b04_arr.dtype == np.uint16
    assert tci_arr.dtype == np.uint8

    assert b04_arr.mean() == 2538.740670966586
    assert tci_arr.mean() == 119.13041675375995
