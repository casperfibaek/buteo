import os
import gzip
import shutil

import requests
import pandas as pd
from tqdm import tqdm
from osgeo import gdal; gdal.UseExceptions()


def process_path(path, OUT_FOLDER):
    path = path.replace("polygons", "points")
    tile = os.path.basename(path)

    input_csv = os.path.join(OUT_FOLDER, tile.replace('.gz', ''))
    output_gpkg = input_csv.replace(".csv", ".gpkg")

    if not os.path.exists(output_gpkg):
        # download and unzip
        r = requests.get(path, timeout=300)
        with open(os.path.join(OUT_FOLDER, tile), 'wb') as f:
            f.write(r.content)

        # Unzip gz file
        with gzip.open(os.path.join(OUT_FOLDER, tile), 'rb') as f_in:
            with open(os.path.join(OUT_FOLDER, tile.replace(".gz", "")), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # remove zip
        os.remove(os.path.join(OUT_FOLDER, tile))

        # call command line
        ogr_options = '-oo X_POSSIBLE_NAMES=longitude -oo Y_POSSIBLE_NAMES=latitude -a_srs "EPSG:4326"'
        ogr_call = f'ogr2ogr "{output_gpkg}" "{input_csv}" {ogr_options}'
        os.system(ogr_call)

        # remove csv
        os.remove(input_csv)

    return True


if __name__ == "__main__":
    import multiprocessing as mp

    FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/projects/unicef/buildings/"
    OUT_FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/projects/unicef/buildings/south-america/"
    PATHS = os.path.join(FOLDER, "south-america_buildings_url.csv")
    PROCESSES = 4

    paths = pd.read_csv(PATHS)["tile_url"]

    progress_bar = tqdm(total=len(paths))

    def update_progress_bar(_result):
        """ Update progress bar """
        progress_bar.update()

    with mp.Pool(PROCESSES) as pool:
        for path in paths:
            pool.apply_async(process_path, args=(path, OUT_FOLDER), callback=update_progress_bar)

        pool.close()
        pool.join()

    progress_bar.close()

    print("Done!")
