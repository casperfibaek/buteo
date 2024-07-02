import sys; sys.path.append("../")
import os
import buteo as beo
import numpy as np


FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/data/data/"
PATH_MASTER = os.path.join(FOLDER, "master_image.tif")
PATH_SLAVE = os.path.join(FOLDER, "slave_image.tif")

slave_aligned = beo.raster_align(
    beo.array_to_raster(beo.raster_to_array(PATH_SLAVE, cast="float32"), reference=PATH_MASTER),
    reference=PATH_MASTER,
    out_path=os.path.join(FOLDER, "coregistered_slave.tif"),
)[0]

slave_mask = (beo.raster_to_array(slave_aligned) != 0).astype(np.float32)

beo.array_to_raster(
    beo.raster_to_array(PATH_MASTER, cast="float32") * slave_mask,
    reference=slave_aligned,
    out_path=os.path.join(FOLDER, "coregistered_master.tif"),
)

gefolki = beo.coregister_images_gefolki(
    os.path.join(FOLDER, "coregistered_slave.tif"),
    os.path.join(FOLDER, "coregistered_master.tif"),
    out_path=os.path.join(FOLDER, "gefolki.tif"),
    fill_value=0,
)
