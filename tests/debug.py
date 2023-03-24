# Standard library
import sys; sys.path.append("../")

import os

import numpy as np
from buteo import (
    raster_to_array,
    vector_to_metadata,
    reproject_bbox,
)

FOLDER_MODELS = r"C:/Users/casper.fibaek/OneDrive - ESA/Desktop/school_catchments/01_nightlights/models/"
FOLDER_DST = r"C:/Users/casper.fibaek/OneDrive - ESA/Desktop/Unicef/"
MODEL_NAME = "NL_MSE_Tiny_01"
BATCH_SIZE = 10000  # Choose an appropriate batch size
BOUNDS = os.path.join(FOLDER_DST, "brazil_rough.gpkg")

# Load the data.
print("Loading data...")

bounds_meta = vector_to_metadata(BOUNDS)
bounds_proj = bounds_meta["projection_osr"]
bounds_bbox = bounds_meta["bbox"]

target_bbox = reproject_bbox(bounds_bbox, bounds_proj, "ESRI:54009")

coords = raster_to_array(os.path.join(FOLDER_DST, "encoded_coordinates_54009_500m.tif"), bbox=target_bbox)
if np.ma.isMaskedArray(coords):
    coords = coords.filled(0.0)

albedo = raster_to_array(os.path.join(FOLDER_DST, "albedo_2019_54009_500m.tif"), bbox=target_bbox)
if np.ma.isMaskedArray(albedo):
    albedo = albedo.filled(0.0)

encoded_time = np.arange(0, 8) / 7
time_raster = np.full((albedo.shape[0], albedo.shape[1], 1), encoded_time[-1], dtype=np.float32)

x_pred = np.concatenate((coords, albedo, time_raster), axis=2, dtype=np.float32)
x_pred = x_pred.reshape((x_pred.shape[0] * x_pred.shape[1], x_pred.shape[2]))
