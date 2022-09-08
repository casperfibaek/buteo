"""
### Perform morphological operations on arrays and rasters.  ###
"""

# Standard library
import sys; sys.path.append("../../")

# Internal
from buteo.raster.convolution import convolve_array, get_kernel
from buteo.raster.edge_detection import edge_detection_sobel
from buteo.raster.morphology import morph_bothat, morph_tophat


def texture_local_variance(img, filter_size=5, spherical=False):
    """ Create a variance texture layer"""
    _kernel, weights, offsets = get_kernel(filter_size, distance_weight="linear", spherical=spherical)
    std = convolve_array(img, offsets, weights, "std")

    return std

def texture_local_median(img, filter_size=5, spherical=False):
    """ Create a variance texture layer"""
    _kernel, weights, offsets = get_kernel(filter_size, distance_weight="linear", spherical=spherical)
    median = convolve_array(img, offsets, weights, "median")

    return median

def texture_local_mean(img, filter_size=5, spherical=False):
    """ Create a variance texture layer"""
    _kernel, weights, offsets = get_kernel(filter_size, distance_weight="linear", spherical=spherical)
    summed = convolve_array(img, offsets, weights, "sum")

    return summed

def texture_local_madvariance(img, filter_size=5, spherical=False):
    """ Create a variance texture layer"""
    _kernel, weights, offsets = get_kernel(filter_size, distance_weight="linear", spherical=spherical)
    mad = convolve_array(img, offsets, weights, "mad")

    return mad

def texture_hole_dif(img, filter_size=5, spherical=False):
    """ Create a variance texture layer"""
    _kernel, weights, offsets = get_kernel(filter_size, distance_weight="linear", spherical=spherical, hole=True)
    hole = convolve_array(img, offsets, weights, "sum")

    hole_dif = img / hole

    return hole_dif


from buteo.raster.core_raster import raster_to_array
from matplotlib import pyplot as plt
img_path = "/home/casper/Desktop/buteo/geometry_and_rasters/s2_b04_baalbeck.tif"


size_filter = 5
sphere = True
read = raster_to_array(img_path)

variance = texture_local_variance(read, filter_size=size_filter, spherical=sphere)

sobel = edge_detection_sobel(read, filter_size=size_filter, spherical=sphere)
sobel_scaled = sobel * (texture_local_mean(variance, filter_size=size_filter, spherical=sphere) / texture_local_mean(sobel, filter_size=size_filter, spherical=sphere))
sobel_ratio = (variance - sobel_scaled) / (variance + sobel_scaled)

tex_1 = sobel_ratio
tex_2 = morph_bothat(read, filter_size=size_filter, spherical=sphere)
tex_3 = morph_tophat(read, filter_size=size_filter, spherical=sphere)
tex_4 = texture_hole_dif(read, filter_size=size_filter, spherical=sphere)
tex_5 = read

size = 16
ncols = 5
nrows = 1

aspect_ratio = read[:, :, 0].shape[0] / read[:, :, 0].shape[1]
figsize = (size * ncols * aspect_ratio, size * nrows)

fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
img1, img2, img3, img4, img5 = axes

for ax in axes:
    ax.set_axis_off()
    ax.margins(0,0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.axis('off')

img1.imshow(tex_1[:, :, 0])
img2.imshow(tex_2[:, :, 0])
img3.imshow(tex_3[:, :, 0])
img4.imshow(tex_4[:, :, 0])
img5.imshow(tex_5[:, :, 0])

fig.tight_layout()
plt.show()
# import pdb; pdb.set_trace()
