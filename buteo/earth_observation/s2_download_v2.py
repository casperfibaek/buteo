import multiprocessing
import sys
from glob import glob

from sen2mosaic.download import main, search
from lib.utils_core import get_size, divide_into_steps

# TODO: Incorporate the vector geometry.
# from intersects_sentinel2_tile import intersecting_tile


def size_of_tile(out_folder, tile):
    images_for_tile = glob(f"{out_folder}*{tile}*.SAFE")
    total_size = 0

    for image in images_for_tile:
        total_size += get_size(image, rough=True)

    return total_size


def get_data(block):
    tiles = block['tiles']
    user = block['user']['username']
    password = block['user']['password']
    out_folder = block['out_folder']

    target_tile_size = block['target_tile_size']

    for tile in tiles:
        current_cloud_percent = 3

        while size_of_tile(out_folder, tile) < target_tile_size:
            print(f'Current cloud level for tile ({tile}): {current_cloud_percent}%')

            main(user, password, tiles,
                 level='2A',
                 start=block['start'],
                 end=block['end'],
                 maxcloud=current_cloud_percent,
                 minsize=block['minsize'],
                 output_dir=out_folder)

            current_cloud_percent += 3

            if current_cloud_percent > 100:
                print(f'Not enough images available in timeframe to fit threshhold ({target_tile_size}mb)')
                break

        print(f'Enough images of {tile} were already present in folder. Increase target_tile_size to collect more.')


def download(tiles_or_geompath, start, end, out_folder, minsize=100, target_tile_size=3 * 1024):
    '''
        Downloads sentinel 2 imagery from either a geometry or a series of tiles.
        Searches for imagery in 3% cloud intervals.

        Args:
            tiles_or_geompath: path to geometry or array with tiles.
            start: start date to download from. eg. 20190301.
            end: start date to download from. eg. 20190615.
            out_folder: destination folder.
            minsize: minimum size of files to download. Full coverage is about 1024mb. default = 100mb
            target_tile_size: attempts to get at least that much data of each tile. default = 3 * 1024mb

        Example:
        start = '20190301'
        end = '20190615'
        out_folder = 'E:\\ghana\\wet_season_2019\\'
        tiles = ['30NYM', '31NBG', '31NBH', '30NYL', '30NYN', '30NZN', '30NZM']

        download(tiles, start, end, out_folder, target_tile_size=3 * 1024)
    '''

    user1 = credentials["user1"]
    user2 = credentials["user2"]
    user3 = credentials["user3"]
    users = [user1, user1, user2, user2, user3, user3]

    tiles_to_download = tiles_or_geompath

    # if isinstance(tiles_or_geompath, str):
    # tiles_to_download = intersecting_tile(tiles_or_geompath)

    tiles_steps = divide_into_steps(tiles_to_download, 6)

    blocks = []
    for i, step in enumerate(tiles_steps):
        blocks.append(
            {'tiles': step, 'out_folder': out_folder, 'start': start, 'end': end, 'user': users[i], 'minsize': minsize, 'target_tile_size': target_tile_size},
        )

    pool = multiprocessing.Pool(6, maxtasksperchild=1)
    pool.map(get_data, blocks, chunksize=1)

    pool.close()
    pool.join()

if __name__ == "__main__":
    # pip install sentinelsat
    # conda install -c conda-forge gdal

    download(['32VNH', '32VNJ', '32UNG'], '20190301', '20190901', 'C:\\Users\\caspe\\Desktop\\s2_downloads')
