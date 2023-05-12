# pylint: disable-all
import sys; sys.path.append("../")

import os
import numpy as np
import buteo as beo
from osgeo import gdal

FOLDER = "./features/"

path = os.path.join(FOLDER, "prediction_2_error.tif")
arr = beo.raster_to_array(path, filled=True, fill_value=0.0, cast=np.float32)

pred = beo.predict_array(
    arr,
    callback=lambda x: x[:, :, :, 0:1],
    tile_size=64,
    n_offsets=3,
    merge_method="mad",
    edge_weighted=True,
)

beo.array_to_raster(
    pred,
    reference=path,
    out_path=os.path.join(FOLDER, "prediction_2_error.tif"),
)

print(np.any(np.isnan(pred)))

# import pdb; pdb.set_trace()