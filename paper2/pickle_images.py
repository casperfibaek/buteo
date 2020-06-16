import sys; sys.path.append('..');
from lib.raster_io import raster_to_array
from lib.utils_core import progress
import sqlite3
import pandas as pd
import numpy as np
from glob import glob

folder = "C:\\Users\\caspe\\Desktop\\Paper_2_StruturalDensity\\"
tiles_path = folder + "tiles_320m\\"

# tiles = glob(tiles_path + "*.tif")
labels_cnx = sqlite3.connect(folder + "analysis\\grid_320m.sqlite")
labels = pd.read_sql_query(f"SELECT fid, b_area, b_volume, ppl_ha FROM 'grid_320m';", labels_cnx)

label_area = []
label_volume = []
label_people = []
images = []

length = len(labels)
for index, row in labels.iterrows():
    fid = row["fid"].astype('uint32')
    b_area = row["b_area"]
    b_volume = row["b_volume"]
    ppl_ha = row["ppl_ha"]

    label_area.append(b_area)
    label_volume.append(b_volume)
    label_people.append(ppl_ha)

    image_path = glob(tiles_path + f"320m_{fid}.tif")[0]
    image = raster_to_array(image_path)

    images.append(image)

    progress(index, length, name="pickling")

images = np.stack(images)
labels = np.stack([label_area, label_volume, label_people])

np.save(folder + 'analysis\\320m_images.npy', images)
np.save(folder + 'analysis\\320m_labels.npy', labels)

import pdb; pdb.set_trace()
print(row)
