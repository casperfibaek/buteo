""" This is a debug script, used for ad-hoc testing. """
# disable all of pylint for this file only.
# pylint: disable-all

# Standard library
import sys; sys.path.append("../")
import os
import numpy as np
import buteo as beo

FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/test_data/"

udp = os.path.join(FOLDER, "T31UDP_20170119T110351_B01.jp2")
udq = os.path.join(FOLDER, "T31UDQ_20170119T110351_B01.jp2")

vrt_file = beo.raster_stack_vrt_list([udp, udq], separate=False)

# Define camera points in UTM-32631 or latlng (EPSG:4326)
points = [
    [430441, 5402109],
    [444808, 5409819],
    [451921, 5391856],
]

points_vector = beo.vector_from_points(points, vrt_file)

# Only necessary if points at in latlng
# points_vector = beo.vector_reproject(points_vector, vrt_file)

# Buffer points 10km
points_buffer = beo.vector_buffer(points_vector, 10000.0)
points_bbox = beo.vector_to_metadata(points_buffer)['bbox']
# Read the bounds and normalise to 0-1
arr = beo.raster_to_array(vrt_file, cast=np.float32, bbox=points_bbox) / 10000.0
# Save the image
beo.array_to_raster(arr, reference=vrt_file, out_path=os.path.join(FOLDER, "test.tif"), bbox=points_bbox)
