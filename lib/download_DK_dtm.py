import sys

sys.path.append("..")
sys.path.append("../lib/")
import geopandas as gpd
import numpy
from .raster_io import *
from .stats_filters import truncate_array
import sys
import os
import time
from zipfile import ZipFile
from glob import glob
from urllib import request


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
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
    project_geom = gpd.read_file(path_to_geom)
    project_geom_wgs = project_geom.to_crs("EPSG:4326")

    grid = "../geometry/Denmark_10km_grid.gpkg"
    grid_geom = gpd.read_file(grid)
    grid_dest = grid_geom.to_crs("EPSG:4326")

    data = []
    for index_g, geom in project_geom_wgs.iterrows():
        for index_t, tile in grid_dest.iterrows():
            if geom["geometry"].intersects(tile["geometry"]):
                if tile["KN10kmDK"] not in data:
                    data.append(tile["KN10kmDK"])

    return data


def download_dtm(tile_names, dst_folder, username, password):
    completed = 0
    for tile in tile_names:
        base_path = f"ftp://{username}:{password}@ftp.kortforsyningen.dk/dhm_danmarks_hoejdemodel/DTM/"
        file_name = f'DTM_{tile.split("_", 1)[1]}_TIF_UTM32-ETRS89.zip'

        if not os.path.exists(dst_folder + file_name):
            try:
                get_file(base_path + file_name, dst_folder + file_name)
            except:
                print(f"Error while trying to download: {base_path + file_name}")
        completed += 1
        print(f"Completed: {completed}/{len(tile_names)} DTM tiles.")


def download_dsm(tile_names, dst_folder, username, password):
    completed = 0
    for tile in tile_names:
        base_path = f"ftp://{username}:{password}@ftp.kortforsyningen.dk/dhm_danmarks_hoejdemodel/DSM/"
        file_name = f'DSM_{tile.split("_", 1)[1]}_TIF_UTM32-ETRS89.zip'

        if not os.path.exists(dst_folder + file_name):
            try:
                get_file(base_path + file_name, dst_folder + file_name)
            except:
                print(f"Error while trying to download: {base_path + file_name}")

        completed += 1
        print(f"Completed: {completed}/{len(tile_names)} DSM tiles.")


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
                sz = ZipFile(dsm_tile).extractall(tmp_folder)
                tz = ZipFile(dtm_tile).extractall(tmp_folder)

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
                                truncate_array(numpy.subtract(ss, tt), min_value=0),
                                out_raster=hot_folder + f"HOT_1km_{s_tiff_tile}.tif",
                                reference_raster=s_tiff,
                            )

                for f in glob(tmp_folder + "/*"):
                    os.remove(f)

        completed += 1
        print(f"Completed: {completed}/{len(dsm_zipped)}")


if __name__ == "__main__":
    dsm_folder = "/home/cfi/data/dsm/"
    dtm_folder = "/home/cfi/data/dtm/"
    hot_folder = "/home/cfi/data/hot/"
    tmp_folder = "/home/cfi/data/tmp/"
    vol_folder = "/home/cfi/data/vol/"

    hot_files = glob(hot_folder + "/*.tif")
    meta = raster_to_metadata(hot_files[0])

    completed = 0
    for f in hot_files:
        tile_base = os.path.basename(f).split("_")[2:4]
        tile_name = "_".join(tile_base).split(".")[0]

        array_to_raster(
            raster_to_array(f)
            * (meta["pixel_width"] * (-meta["pixel_height"])),
            out_raster=vol_folder + f"VOL_1km_{tile_name}.tif",
            reference_raster=f,
        )

        completed += 1
        print(f"Completed: {completed}/{len(hot_files)}")

    download_dtm(
        find_tile_names("../geometry/studyAreaHull.gpkg"),
        "/home/cfi/data/dtm/",
        "casperfibaek",
        "Goldfish12",
    )
    download_dsm(
        find_tile_names("../geometry/studyAreaHull.gpkg"),
        "/home/cfi/data/dsm/",
        "casperfibaek",
        "Goldfish12",
    )

    height_over_terrain(dsm_folder, dtm_folder, hot_folder, tmp_folder)
