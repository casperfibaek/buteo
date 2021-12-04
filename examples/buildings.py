import sys

sys.path.append("../")
sys.path.append("../../")
import numpy as np
from buteo.raster.io import raster_to_array, array_to_raster, stack_rasters_vrt, internal_raster_to_metadata
from buteo.vector.rasterize import rasterize_vector
from buteo.raster.resample import internal_resample_raster
from buteo.utils import progress
import os
from glob import glob


building_folder = "/home/cfi/Desktop/buildings_denmark/building_grid/"
hot_folder = "/media/cfi/lts/terrain/hot/"
dst_folder = "/home/cfi/Desktop/buildings_denmark/rasterized2/"
tmp_folder = "/media/cfi/lts/terrain/tmp/"

vrt_files = glob(hot_folder + "*.vrt")
building_tiles = glob(building_folder + "*.gpkg")

previous_errors = ['629_65', '618_68', '613_61', '612_65', '609_55', '626_56', '616_56', '626_47', '614_65', '622_57', '629_66', '613_63', '616_74', '628_66', '620_60', '619_71', '634_52', '617_74', '613_47', '608_68', '619_62', '619_58', '616_57', '609_60', '620_69', '610_64', '614_68', '622_51', '613_64', '612_62', '604_68', '633_61', '605_64', '625_44', '613_66', '616_58', '613_59', '619_46', '633_48', '631_45', '618_66', '634_63', '627_57']

error_files = []

processed = 0
for index, vrt_file in enumerate(vrt_files):
    progress(processed, len(vrt_files), "Generating")

    vrt_tile_name = "_".join(os.path.splitext(os.path.basename(vrt_file))[0].split("_")[1:])

    short_vrt_tile_name = "_".join(vrt_tile_name.split("_")[1:])

    if short_vrt_tile_name not in previous_errors:
        processed += 1
        continue
    
    found = False
    found_path = None
    for building_tile in building_tiles:
        if found:
            continue

        build_tile_name = "_".join(os.path.splitext(os.path.basename(building_tile))[0].split("_")[2:])

        if vrt_tile_name == build_tile_name:
            found_path = building_tile
            found = True

    if not found:
        area_vol_10m = internal_resample_raster(vrt_file, (10, 10), resample_alg='average', out_path=tmp_folder + f"buildings_volume_{vrt_tile_name}_10m_unscaled.tif")

        hot_arr = raster_to_array(area_vol_10m) * 0

        vol_10m_path = dst_folder + f"buildings_volume_{vrt_tile_name}_10m.tif"
        area_10m_path = dst_folder + f"buildings_area_{vrt_tile_name}_10m.tif"

        array_to_raster(hot_arr, reference=area_vol_10m, out_path=vol_10m_path)
        array_to_raster(hot_arr, reference=area_vol_10m, out_path=area_10m_path)

        processed += 1
        continue

    try:
        metadata = internal_raster_to_metadata(vrt_file)
    except:
        print(f"Error while processing tile: {vrt_file}")
        error_files.append(vrt_file)
        processed += 1
        continue

    width = metadata["pixel_width"]
    height = metadata["pixel_height"]
    out_name = tmp_folder + f"buildings_40cm_{vrt_tile_name}.tif"
    xmin, xmax, ymin, ymax = metadata["extent_ogr"]
    projection = metadata["projection"]

    try:
        cmd = f"""\
            gdal_rasterize\
            -burn 1.0\
            -tr {width} {height}\
            -init 0.0\
            -te {xmin} {ymin} {xmax} {ymax}\
            -ot Byte\
            -of GTiff\
            -co COMPRESS=PACKBITS\
            -q\
            {found_path}\
            {out_name}\
        """

        completed = os.system(cmd)

        if completed != 0:
            print(f"Error while rasterizing: {vrt_tile_name}")

        buildings_arr = raster_to_array(out_name, filled=True)
        hot_arr = raster_to_array(vrt_file, filled=True)

        volume = (np.maximum(buildings_arr * 2.5, hot_arr) * (buildings_arr > 0) * width * height).astype("float32")
        volume_path = tmp_folder + f"buildings_volume_{vrt_tile_name}.tif"
        array_to_raster(volume, reference=vrt_file, out_path=volume_path)
        
        hot_arr = None
        volume = None

        area = (buildings_arr * width * height).astype("float32")
        area_path = tmp_folder + f"buildings_area_{vrt_tile_name}.tif"
        array_to_raster(area, reference=vrt_file, out_path=area_path)

        area = None
        hot_arr = None

        target_size = (10, 10)
        value_scale_1 = (target_size[0] / width) * (target_size[1] * height)
        value_scale = value_scale_1 * (value_scale_1 / (target_size[0] * target_size[1]))

        vol_10m = internal_resample_raster(volume_path, (10, 10), resample_alg='average', out_path=tmp_folder + f"buildings_volume_{vrt_tile_name}_10m_unscaled.tif")
        area_10m = internal_resample_raster(area_path, (10, 10), resample_alg='average', out_path=tmp_folder + f"buildings_area_{vrt_tile_name}_10m_unscaled.tif")

        vol_10m_path = dst_folder + f"buildings_volume_{vrt_tile_name}_10m.tif"
        area_10m_path = dst_folder + f"buildings_area_{vrt_tile_name}_10m.tif"

        array_to_raster(raster_to_array(vol_10m) * value_scale, reference=vol_10m, out_path=vol_10m_path)
        array_to_raster(raster_to_array(area_10m) * value_scale, reference=area_10m, out_path=area_10m_path)

    except:
        print(f"Error while processing tile: {vrt_file}")
        error_files.append(vrt_file)
        processed += 1
        continue 

    for f in glob(tmp_folder + "*"):
        os.remove(f)

    processed += 1

    progress(processed, len(vrt_files), "Generating")

import pdb; pdb.set_trace()

# ['/media/cfi/lts/terrain/hot/HOT_10km_629_65.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_618_68.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_613_61.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_612_65.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_609_55.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_626_56.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_616_56.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_626_47.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_614_65.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_622_57.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_629_66.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_613_63.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_616_74.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_628_66.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_620_60.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_619_71.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_634_52.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_617_74.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_613_47.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_608_68.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_619_62.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_619_58.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_616_57.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_609_60.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_620_69.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_610_64.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_614_68.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_622_51.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_613_64.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_612_62.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_604_68.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_633_61.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_605_64.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_625_44.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_613_66.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_616_58.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_613_59.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_619_46.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_633_48.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_631_45.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_618_66.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_634_63.vrt', '/media/cfi/lts/terrain/hot/HOT_10km_627_57.vrt']
# ['629_65', '618_68', '613_61', '612_65', '609_55', '626_56', '616_56', '626_47', '614_65', '622_57', '629_66', '613_63', '616_74', '628_66', '620_60', '619_71', '634_52', '617_74', '613_47', '608_68', '619_62', '619_58', '616_57', '609_60', '620_69', '610_64', '614_68', '622_51', '613_64', '612_62', '604_68', '633_61', '605_64', '625_44', '613_66', '616_58', '613_59', '619_46', '633_48', '631_45', '618_66', '634_63', '627_57']
