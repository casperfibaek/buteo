import os
import sys; sys.path.append("../")
import pandas as pd
import math
import tqdm
import requests
import numpy as np
import buteo as beo
from osgeo import gdal


FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/glimmer_test/"
OTB_PATH = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/projects/otb/"

beo.otb_fine_registration(
    reference_image=os.path.join(FOLDER, "master_HDR.tif"),
    secondary_image=os.path.join(FOLDER, "slave_HDR.tif"),
    warp_image=os.path.join(FOLDER, "slave_HDR.tif"),
    out_raster=os.path.join(FOLDER, "slave_HDR_out.tif"),
    out_raster_warp=os.path.join(FOLDER, "slave_HDR_warp.tif"),
    otb_path=OTB_PATH,
    erx=3,
    ery=3,
    mrx=3,
    mry=3,
)
