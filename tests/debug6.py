import os
import sys; sys.path.append("../")
from glob import glob
from tqdm import tqdm

import buteo as beo


PATHS = [
    { "path": "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/egypt/", "name": "EGY1" },
    { "path": "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/ghana/", "name": "GHA1" },
    { "path": "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/israel_gaza/", "name": "ISR1" },
    { "path": "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/tanzania_dar/", "name": "TZA1" },
    { "path": "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/tanzania_kigoma/", "name": "TZA2" },
    { "path": "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/tanzania_kilimanjaro/", "name": "TZA3" },
    { "path": "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/tanzania_mwanza_Q2/", "name": "TZA4" },
    { "path": "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/tanzania_mwanza_Q3/", "name": "TZA5" },
    { "path": "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/uganda/", "name": "UGA1" },
]

for idx, path in enumerate(tqdm(PATHS, total=len(PATHS))):
    FOLDER = path["path"]
    NAME = path["name"]

    # Then we resample the 20m images to 10m.
    # The 10m images are loaded and the nodata values are removed and replaced with 0.
    if not os.path.exists(os.path.join(FOLDER, "subset")):
        os.mkdir(os.path.join(FOLDER, "subset"))

    path_mask = os.path.join(FOLDER, "mask.gpkg"); beo.vector_reset_fids(path_mask)

    split_geoms = beo.vector_split_by_fid(path_mask, os.path.join(FOLDER, "subset"), prefix=f"{NAME}_geom_")

    for idx, geom in enumerate(sorted(split_geoms)):
        fid = idx

        beo.raster_clip(
            os.path.join(FOLDER, "mask.tif"),
            clip_geom=geom,
            out_path=os.path.join(FOLDER, "subset", f"{NAME}_mask_{fid}.tif"),
        )

        beo.raster_clip(
            os.path.join(FOLDER, "labels.tif"),
            clip_geom=geom,
            out_path=os.path.join(FOLDER, "subset", f"{NAME}_label_{fid}.tif"),
        )

        s02_clipped = beo.raster_clip(os.path.join(FOLDER, "B02.tif"), clip_geom=geom)

        s2_clipped = beo.raster_clip(
            [
                os.path.join(FOLDER, "B02.tif"),
                os.path.join(FOLDER, "B03.tif"),
                os.path.join(FOLDER, "B04.tif"),
                os.path.join(FOLDER, "B08.tif"),
                os.path.join(FOLDER, "B05.tif"),
                os.path.join(FOLDER, "B06.tif"),
                os.path.join(FOLDER, "B07.tif"),
                os.path.join(FOLDER, "B8A.tif"),
                os.path.join(FOLDER, "B11.tif"),
                os.path.join(FOLDER, "B12.tif"),
            ],
            clip_geom=geom,
        )
        beo.raster_stack_list(s2_clipped, out_path=os.path.join(FOLDER, "subset", f"{NAME}_s2_{fid}.tif"))
        beo.delete_dataset_if_in_memory(s2_clipped)

        s1_clipped = beo.raster_clip(
            [
                os.path.join(FOLDER, "VV.tif"),
                os.path.join(FOLDER, "VH.tif"),
            ],
            clip_geom=geom,
        )
        beo.raster_stack_list(s1_clipped, out_path=os.path.join(FOLDER, "subset", f"{NAME}_s1_{fid}.tif"))
        beo.delete_dataset_if_in_memory(s1_clipped)

    for file in glob(FOLDER + "/subset/*.gpkg"):
        os.remove(file)
