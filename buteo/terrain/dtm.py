import sys

sys.path.append("../")
sys.path.append("../../")
import geopandas as gpd
import numpy as np
from buteo.raster.io import *
import sys
import os
import time
from zipfile import ZipFile
from glob import glob
from urllib import request


def reporthook(count, block_size, total_size):
    # print('-------------------------', count)
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
    project_geom = gpd.read_file(path_to_geom)
    project_geom_wgs = project_geom.to_crs("EPSG:4326")

    grid = "../../geometry/Denmark_10km_grid.gpkg"
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
                                np.abs(np.subtract(ss, tt)),
                                out_path=out_folder + f"HOT_1km_{s_tiff_tile}.tif",
                                reference=s_tiff,
                            )

                for f in glob(tmp_folder + "/*"):
                    os.remove(f)

        completed += 1
        print(f"Completed: {completed}/{len(dsm_zipped)}")


if __name__ == "__main__":
    from glob import glob
    from buteo.raster.resample import resample_raster

    base = "C:/Users/caspe/Desktop/test/"
    orto_folder = base + "orto/"

    # download_orto(
    #     find_tile_names(base + "fjord.gpkg"),
    #     orto_folder,
    #     "ezratrotter",
    #     "Bigcloud8!!!",
    # )

    ortos = glob(orto_folder + "og/*.ecw")
    for index, orto in enumerate(ortos):
        ortos[index] = os.path.abspath(ortos[index])

    # stupid ecw format.. :/
    # gdal_translate -of GTiff -co "COMPRESS=DEFLATE" -co "TILED=YES" -co "BIGTIFF=YES" -co "NUM_THREADS=ALL_CPUS" 10km_6140_580_2019.ecw 10km_6140_580_2019.tif
    resample_raster(
        ortos, target_size=2.0, out_path=orto_folder + "2m/", resample_alg="average"
    )
    import pdb

    pdb.set_trace()

