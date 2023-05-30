# pylint: disable-all
import sys; sys.path.append("../")

import os
import numpy as np
import buteo as beo
from osgeo import gdal

FOLDER = "./features/"

path = os.path.join(FOLDER, "test_vector_roads.gpkg")

bob = beo.vector_rasterize(path, 0.0001, extent=path)

import pdb; pdb.set_trace()