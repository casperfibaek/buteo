""" Bug-fixing script """

# pylint: skip-file

# Standard library
import sys; sys.path.append("../")
import os
from glob import glob

# External
import numpy as np
from osgeo import gdal

# local
from buteo import align_rasters

FOLDER = r"C:/Users/casper.fibaek/OneDrive - ESA/Desktop/school_catchments/01_expectation_map/"

background_light = os.path.join(FOLDER, "background_light_MLP_12_MSE.tif")
reference = os.path.join(FOLDER, "500m_reference.tif")

align_rasters(background_light, master=reference, out_path=FOLDER)
