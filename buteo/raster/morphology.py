"""
### Perform morphological operations on arrays and rasters.  ###
"""

# Standard library
import sys; sys.path.append("../../")

# Internal
from buteo.raster.core_raster import raster_to_array, array_to_raster
from buteo.raster.convolution import get_kernel, convolve_array


kernel, weights, offsets  = get_kernel(5, 3, spherical=True, normalise=False)
import pdb; pdb.set_trace()


def erode_array(arr, size, spherical=True):
    kernel, weights, offsets = get_kernel(size, arr.shape[-1], spherical=True, normalise=False)

# Erode: local minimum
# Dilate: local maximum
# Open: Erode -> Dilate
# Close: Dilate -> Close
# Hats
# Textures
