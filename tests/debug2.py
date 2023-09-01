import os
import sys; sys.path.append("../")
import buteo as beo
import numpy as np
from tqdm import tqdm
from glob import glob
from osgeo import gdal

gdal.PushErrorHandler('CPLQuietErrorHandler')


folder = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/datasets/s12_buildings/data_images/"
folder_out = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/datasets/s12_buildings/data_images_v2/"
world_cover_vrt = "D:/data/esa_worldcover2021/esa_worldcover2021_wgs84.vrt"

masks = glob(os.path.join(folder_out, "*_wc.tif"))

for f in masks:
    os.rename(f, f.replace("_wc.tif", "_lc.tif"))

exit()

masks = glob(os.path.join(folder, "*_mask.tif"))
for idx, mask in tqdm(enumerate(masks), total=len(masks)):
    basename = os.path.splitext(os.path.basename(mask))[0]
    split = basename.split("_")
    id_number = split[1]
    s2_path = os.path.join(folder, split[0] + "_" + split[1] + "_s2.tif")
    label_path = os.path.join(folder, split[0] + "_" + split[1] + "_label.tif")

    if split[0] == "DNK1":
        location = "denmark-1"
    elif split[0] == "DNK2":
        location = "denmark-2"
    elif split[0] == "EGY1":
        location = "egypt-1"
    elif split[0] == "GHA1":
        location = "ghana-1"
    elif split[0] == "ISR1":
        location = "isreal-1"
    elif split[0] == "ISR2":
        location = "isreal-2"
    elif split[0] == "TZA1":
        location = "tanzania-1"
    elif split[0] == "TZA2":
        location = "tanzania-2"
    elif split[0] == "TZA3":
        location = "tanzania-3"
    elif split[0] == "TZA4":
        location = "tanzania-4"
    elif split[0] == "TZA5":
        location = "tanzania-5"
    elif split[0] == "UGA1":
        location = "uganda-1"
    else:
        raise NotImplementedError

    out_arr = os.path.join(folder_out, f"{location}_{id_number}_0.tif")
    out_label_building = os.path.join(folder_out, f"{location}_{id_number}_0_label_building.tif")
    out_label_wc = os.path.join(folder_out, f"{location}_{id_number}_0_label_wc.tif")

    if os.path.exists(out_arr) and os.path.exists(out_label_building) and os.path.exists(out_label_wc):
        continue

    tile_wgs84 = beo.raster_get_footprints(mask, True, suffix=str(idx))
    tile_wgs84_buffer = beo.vector_buffer(tile_wgs84, 0.01, in_place=True)

    world_cover_wgs84 = beo.raster_clip(world_cover_vrt, tile_wgs84_buffer, to_extent=True, suffix=str(idx))
    world_cover_utm = beo.raster_align(world_cover_wgs84, reference=label_path, suffix=str(idx))
    arr_world_cover = beo.raster_to_array(world_cover_utm)

    beo.delete_dataset_if_in_memory(tile_wgs84)
    beo.delete_dataset_if_in_memory(world_cover_wgs84)
    beo.delete_dataset_if_in_memory(world_cover_utm)

    arr_label = beo.raster_to_array(label_path, cast=np.float32) / 100.0

    arr_mask = beo.raster_to_array(mask)
    arr_mask = np.where(arr_mask == 1, 7, 0)
    arr_mask = arr_mask.astype(np.uint16)
    arr_s2 = beo.raster_to_array(s2_path)

    arr = np.concatenate([arr_mask, arr_s2], axis=2)

    if np.isnan(arr).any():
        import pdb; pdb.set_trace()
    elif np.isnan(arr_label).any():
        import pdb; pdb.set_trace()
    elif np.isnan(arr_world_cover).any():
        import pdb; pdb.set_trace()
   
    beo.array_to_raster(
        arr,
        reference=label_path,
        out_path=os.path.join(folder_out, f"{location}_{id_number}_0.tif"),
    )

    beo.array_to_raster(
        arr_label,
        reference=label_path,
        out_path=os.path.join(folder_out, f"{location}_{id_number}_0_label_building.tif"),
    )

    beo.array_to_raster(
        arr_world_cover,
        reference=label_path,
        out_path=os.path.join(folder_out, f"{location}_{id_number}_0_label_wc.tif"),
    )
