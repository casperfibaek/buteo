import os
import sys; sys.path.append("..")
from glob import glob
from tqdm import tqdm

import buteo as beo

FOLDER = "D:/data/s2_building_and_roads/images/"
VRT = "D:/data/esa_worldcover2021/esa_worldcover2021_wgs84.vrt"


images = glob(FOLDER + "*label_roads.tif")

for img in tqdm(images, total=len(images)):
    name = os.path.splitext(os.path.basename(img))[0]
    name = name.replace("label_roads", "label_lc.tif")

    warped = beo.raster_warp(
        VRT,
        out_path=os.path.join(FOLDER, name),
        src_projection=VRT,
        dst_projection=img,
        resampling_alg="mode",
        align_pixels=True,
        dst_extent=img,
        dst_extent_srs=img,
        dst_x_res=img,
        dst_y_res=img,
        clip_geom=beo.raster_get_footprints(img, latlng=False),
    )
