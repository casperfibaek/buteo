import os
import sys; sys.path.append("../")
import pandas as pd
import math
import tqdm
import requests
import numpy as np
import buteo as beo
from osgeo import gdal


image = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/projects/buteo/tests/features/normal_image.png"

# define projection
# beo.set_projection(image, 3857, pixel_size_y=10.0, pixel_size_x=10.0)

bob = beo.raster_dem_to_orientation(image, out_path="C:/Users/casper.fibaek/OneDrive - ESA/Desktop/projects/buteo/tests/features/normal_image_orientation.tif")
