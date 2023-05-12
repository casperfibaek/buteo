# pylint: disable-all
import sys; sys.path.append("../")

import os
import numpy as np
import buteo as beo
from osgeo import gdal

FOLDER = "./features/"
img = os.path.join(FOLDER, "prediction_5.tif")

arr = beo.raster_to_array(img, filled=True, fill_value=0.0, cast=np.float32)
arr = np.ones_like(arr)

pred = beo.predict_array(
    arr,
    callback=lambda x: x,
    tile_size=64,
    n_offsets=3,
    merge_method="median",
    edge_weighted=True,
)

print(np.any(np.isnan(pred)))

# import pdb; pdb.set_trace()