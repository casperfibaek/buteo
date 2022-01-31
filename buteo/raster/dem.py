from doctest import master
import sys
from turtle import distance
import numpy as np
from osgeo import gdal


sys.path.append("../")
sys.path.append("../../")
np.set_printoptions(suppress=True)

from buteo.raster.resample import resample_raster
from buteo.raster.io import raster_to_array, array_to_raster
from buteo.filters.convolutions import filter_array
from buteo.raster.align import align_rasters
from buteo.raster.clip import clip_raster
from buteo.raster.reproject import reproject_raster

# TODO: remove geopandas
# TODO: make sense of all of this.l

# example
folder = "D:/training_data_buteo/denmark_2021_Q2/"

reference = resample_raster(folder + "B12_20m.tif", 40.0)
dem = folder + "DK_COP30.tif"

v1 = reproject_raster(dem, reference, postfix="")
v2 = resample_raster(v1, 40.0, postfix="", resample_alg="average")
v3 = align_rasters(v2, master=reference, postfix="")


smoothed = array_to_raster(
    filter_array(raster_to_array(v3), (5, 5), nodata_value=-9999, operation="median"),
    reference=reference,
    out_path=folder + "DK_COP30_20m_smoothed.tif",
)

slope_name = "/vsimem/v4_slope.tif"
gdal.DEMProcessing(slope_name, smoothed, 'slope', options=[], slopeFormat="percent")
array_to_raster((raster_to_array(slope_name) / 100.0).astype("float32"), reference=reference, out_path=folder+ "DK_COP30_20m_slope_smooth.tif")

aspect_name = "/vsimem/v4_aspect.tif"
gdal.DEMProcessing(aspect_name, smoothed, 'aspect', options=[])
array_to_raster(raster_to_array(aspect_name).astype("float32"), reference=reference, out_path=folder+ "DK_COP30_20m_aspect_smooth.tif")

import sys
import os
import time

import numpy as np
from zipfile import ZipFile
from glob import glob
from urllib import request

sys.path.append("../")

from buteo.raster.io import raster_to_array, array_to_raster, stack_rasters_vrt


start_time = time.time()



def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = (time.time() + 0.1) - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)

    sys.stdout.write(
        "\r...%d%%, %d MB, %d KB/s, %d seconds passed"
        % (percent, progress_size / (1024 * 1024), speed, duration)
    )
    sys.stdout.flush()


def get_file(url, filename):
    request.urlretrieve(url, filename, reporthook)


def find_tile_names(path_to_geom):
    import geopandas as gpd
    project_geom = gpd.read_file(path_to_geom)
    project_geom_wgs = project_geom.to_crs("EPSG:4326")

    grid = "../../geometry/Denmark_10km_grid_land.gpkg"
    grid_geom = gpd.read_file(grid)
    grid_dest = grid_geom.to_crs("EPSG:4326")

    data = []
    for _, geom in project_geom_wgs.iterrows():
        for _, tile in grid_dest.iterrows():
            if geom["geometry"].intersects(tile["geometry"]):
                if tile["KN10kmDK"] not in data:
                    data.append(tile["KN10kmDK"])

    return data


def download_dtm(tile_names, dst_folder, username, password):
    completed = 0
    if not os.path.isdir(dst_folder):
        raise Exception("Error: output directory does not exist.")

    for tile in tile_names:
        base_path = f"ftp://{username}:{password}@ftp.kortforsyningen.dk/dhm_danmarks_hoejdemodel/DTM/"
        file_name = f'DTM_{tile.split("_", 1)[1]}_TIF_UTM32-ETRS89.zip'
        if not os.path.exists(dst_folder + file_name):
            try:
                get_file(base_path + file_name, dst_folder + file_name)
                print(f"Completed: {completed}/{len(tile_names)} DTM tiles.")
            except:
                print(f"Error while trying to download: {base_path + file_name}")
        else:
            print(f"{file_name} Already exists.")
        completed += 1


def download_dsm(tile_names, dst_folder, username, password):
    completed = 0
    if not os.path.isdir(dst_folder):
        raise Exception("Error: output directory does not exist.")

    for tile in tile_names:
        base_path = f"ftp://{username}:{password}@ftp.kortforsyningen.dk/dhm_danmarks_hoejdemodel/DSM/"
        file_name = f'DSM_{tile.split("_", 1)[1]}_TIF_UTM32-ETRS89.zip'

        if not os.path.exists(dst_folder + file_name):
            try:
                get_file(base_path + file_name, dst_folder + file_name)
                print(f"Completed: {completed}/{len(tile_names)} DSM tiles.")
            except:
                print(f"Error while trying to download: {base_path + file_name}")
        else:
            print(f"{file_name} Already exists.")
        completed += 1


def download_orto(tile_names, dst_folder, username, password, year="2019"):
    completed = 0
    if not os.path.isdir(dst_folder):
        raise Exception("Error: output directory does not exist.")

    for tile in tile_names:
        base_path = f"ftp://{username}:{password}@ftp.kortforsyningen.dk/grundlaeggende_landkortdata/ortofoto/blokinddelt/GEODANMARK/"
        file_name = f'10km_{year}_{tile.split("_", 1)[1]}_ECW_UTM32-ETRS89.zip'

        if not os.path.exists(dst_folder + file_name):
            try:
                get_file(base_path + file_name, dst_folder + file_name)
                print(f"Completed: {completed + 1}/{len(tile_names)} orto tiles.")
            except:
                print(f"Error while trying to download: {base_path + file_name}")
        else:
            print(f"{file_name} Already exists.")
        completed += 1


def get_tile_from_zipped_url(path):
    basename = os.path.basename(path)
    split = basename.split("_")
    return split[1] + "_" + split[2]


def height_over_terrain(dsm_folder, dtm_folder, out_folder, tmp_folder):
    dsm_zipped = glob(dsm_folder + "*.zip")
    dtm_zipped = glob(dtm_folder + "*.zip")

    completed = 0
    for dsm_tile in dsm_zipped:
        s = get_tile_from_zipped_url(dsm_tile)

        for dtm_tile in dtm_zipped:
            t = get_tile_from_zipped_url(dtm_tile)

            if s == t:
                ZipFile(dsm_tile).extractall(tmp_folder)
                ZipFile(dtm_tile).extractall(tmp_folder)

                dsm_tiffs = glob(tmp_folder + "DSM_*.tif")
                dtm_tiffs = glob(tmp_folder + "DTM_*.tif")

                for s_tiff in dsm_tiffs:
                    s_tiff_tile_base = os.path.basename(s_tiff).split("_")[2:4]
                    s_tiff_tile = "_".join(s_tiff_tile_base).split(".")[0]

                    for t_tiff in dtm_tiffs:
                        t_tiff_tile_base = os.path.basename(t_tiff).split("_")[2:4]
                        t_tiff_tile = "_".join(t_tiff_tile_base).split(".")[0]

                        if s_tiff_tile == t_tiff_tile:
                            ss = raster_to_array(s_tiff)
                            tt = raster_to_array(t_tiff)

                            array_to_raster(
                                np.abs(np.subtract(ss, tt)),
                                out_path=out_folder + f"HOT_1km_{s_tiff_tile}.tif",
                                reference=s_tiff,
                            )

                for f in glob(tmp_folder + "/*"):
                    os.remove(f)

        completed += 1
        print(f"Completed: {completed}/{len(dsm_zipped)}")


def volume_over_terrain(
    tile_names,
    username,
    password,
    out_folder,
    tmp_folder,
):
    if not os.path.isdir(tmp_folder):
        raise Exception("Error: output directory does not exist.")

    error_tiles = []

    completed = 0

    for tile in tile_names:

        base_path_DSM = f"ftp://{username}:{password}@ftp.kortforsyningen.dk/dhm_danmarks_hoejdemodel/DSM/"
        file_name_DSM = f'DSM_{tile.split("_", 1)[1]}_TIF_UTM32-ETRS89.zip'

        base_path_DTM = f"ftp://{username}:{password}@ftp.kortforsyningen.dk/dhm_danmarks_hoejdemodel/DTM/"
        file_name_DTM = f'DTM_{tile.split("_", 1)[1]}_TIF_UTM32-ETRS89.zip'

        try:

            if not os.path.exists(tmp_folder + file_name_DSM):
                get_file(base_path_DSM + file_name_DSM, tmp_folder + file_name_DSM)
            else:
                print(f"{file_name_DSM} Already exists.")

            if not os.path.exists(tmp_folder + file_name_DTM):
                get_file(base_path_DTM + file_name_DTM, tmp_folder + file_name_DTM)
            else:
                print(f"{file_name_DTM} Already exists.")

            ZipFile(tmp_folder + file_name_DSM).extractall(tmp_folder)
            ZipFile(tmp_folder + file_name_DTM).extractall(tmp_folder)

            dsm_tiffs = glob(tmp_folder + "DSM_*.tif")
            dtm_tiffs = glob(tmp_folder + "DTM_*.tif")

            hot_files = []

            for s_tiff in dsm_tiffs:
                s_tiff_tile_base = os.path.basename(s_tiff).split("_")[2:4]
                s_tiff_tile = "_".join(s_tiff_tile_base).split(".")[0]

                for t_tiff in dtm_tiffs:
                    t_tiff_tile_base = os.path.basename(t_tiff).split("_")[2:4]
                    t_tiff_tile = "_".join(t_tiff_tile_base).split(".")[0]

                    if s_tiff_tile == t_tiff_tile:
                        ss = raster_to_array(s_tiff)
                        tt = raster_to_array(t_tiff)

                        hot_name = out_folder + f"HOT_1km_{s_tiff_tile}.tif"

                        subtracted = np.subtract(ss, tt)
                        hot_arr = np.abs((subtracted >= 0) * subtracted)
                        array_to_raster(
                            hot_arr,
                            out_path=hot_name,
                            reference=s_tiff,
                        )

                        hot_files.append(hot_name)

            km10_tilename = "_".join(file_name_DSM.split("_")[1:3])

            vrt_name = out_folder + f"HOT_10km_{km10_tilename}.vrt"

            stack_rasters_vrt(
                hot_files,
                seperate=False,
                out_path=vrt_name,
            )

        except:
            print("Error while processing tile: {tile}")
            error_tiles.append(tile)

        finally:
            for f in glob(tmp_folder + "/*"):
                os.remove(f)

            completed += 1
            print(f"Completed: {completed}/{len(tile_names)} - {tile}")
