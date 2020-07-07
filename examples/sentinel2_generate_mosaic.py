import sys; sys.path.append('..'); sys.path.append('../lib/')

import os
from time import time
import geopandas as gpd
from fnmatch import fnmatch
from glob import glob
import shutil
from pyproj import CRS
from sen2mosaic.download import decompress
from mosaic_tool import mosaic_tile


def create_mosaic(s2_files, dst_dir, dst_projection=None):
    assert os.path.isdir(dst_dir), "Output directory is invalid"

    if isinstance(s2_files, str):
        assert os.path.isdir(s2_files), "Input directory is invalid"

        input_images = glob(s2_files + '/*')
        assert len(input_images) > 0, "Input folder is empty"
    else:
        assert isinstance(s2_files, list)
        assert len(s2_files) > 0, "Input file list is empty"

        for f in s2_files:
            assert os.path.exists(f), "File referenced does not exist"
        
        input_images = s2_files

    # Test filename pattern
    for f in input_images:
        assert fnmatch(os.path.basename(f), "S2*_*_*_*_*_*"), "Input file does not match pattern S2*_*_*_*_*_*"
    
    # Seperate input_images into constituent tiles
    tiles = {}
    for f in input_images:
        tile_name = os.path.basename(f).split("_")[5][1:]
        if tile_name not in tiles:
            tiles[tile_name] = [f]
        else:
            tiles[tile_name].append(f)

    for tile, paths in tiles.items():

        before = time()

        # Test if files are zipped
        zipped = 0
        for f in paths:
            basename = os.path.basename(f)
            ext = basename.rsplit(".", 1)[1]
            
            if ext == "zip":
                zipped += 1
        
        assert zipped == 0 or zipped == len(paths), "Mix of zipped and unzipped files"

        # If files are zipped, unzip to temporary folder
        if zipped > 0:
            tmp_folder = os.path.join(dst_dir, "__tmp__")
            if not os.path.exists(tmp_folder):
                os.makedirs(tmp_folder)
            
            empty = glob(tmp_folder + "/*")
            for e in empty:
                os.remove(e)

            for f in paths:
                decompress(f, tmp_folder)

            paths = glob(tmp_folder + "/*")

        mosaic_tile(paths, dst_dir, out_name=tile, dst_projection=dst_projection)

        if zipped > 0:
            try:
                for f in glob(tmp_folder + "/*"):
                    shutil.rmtree(f)
                os.rmdir(tmp_folder)
            except:
                pass
        
        print(f"Finished processing {tile} in {round(time() - before, 1)} seconds")


if __name__ == "__main__":
    in_files = "/home/cfi/data/sentinel2_paper2/"
    out_folder = "/home/cfi/data/sentinel2_paper2_mosaic/"
    create_mosaic(in_files, out_folder)
