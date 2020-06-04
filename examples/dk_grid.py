import geopandas as gpd
import sys
import time
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
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()


def get_file(url, filename):
    request.urlretrieve(url, filename, reporthook)


def find_tile_names(path_to_geom):
    project_geom = gpd.read_file(path_to_geom)
    project_geom_wgs = project_geom.to_crs('EPSG:4326')

    grid = '../geometry/Denmark_10km_grid.gpkg'
    grid_geom = gpd.read_file(grid)
    grid_dest = grid_geom.to_crs('EPSG:4326')

    data = []
    for index_g, geom in project_geom_wgs.iterrows():
        for index_t, tile in grid_dest.iterrows():
            if geom['geometry'].intersects(tile['geometry']):
                if tile['KN10kmDK'] not in data:
                    data.append(tile['KN10kmDK'])

    return data


def download_dtm(tile_names, dst_folder, username, password):
    for tile in tile_names:
        base_path = f'ftp://{username}:{password}@ftp.kortforsyningen.dk/dhm_danmarks_hoejdemodel/DTM/'
        file_name = f'DTM_{tile.split("_", 1)[1]}_TIF_UTM32-ETRS89.zip'
        get_file(base_path + file_name, dst_folder + file_name)


def download_dsm(tile_names, dst_folder, username, password):
    for tile in tile_names:
        base_path = f'ftp://{username}:{password}@ftp.kortforsyningen.dk/dhm_danmarks_hoejdemodel/DSM/'
        file_name = f'DSM_{tile.split("_", 1)[1]}_TIF_UTM32-ETRS89.zip'
        get_file(base_path + file_name, dst_folder + file_name)


if __name__ == "__main__":
    # download_dtm(find_tile_names('../geometry/studyAreaHull.gpkg'), '/home/cfi/data/dtm/', 'casperfibaek', 'Goldfish12')
    download_dsm(find_tile_names('../geometry/studyAreaHull.gpkg'), '/home/cfi/data/dsm/', 'casperfibaek', 'Goldfish12')