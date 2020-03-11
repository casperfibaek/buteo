from osgeo import gdal, osr
from glob import glob
import time
import multiprocessing
import sys
import os

from sen2mosaic.mosaic import main
from sen2mosaic.utilities import prepInfiles
from lib.orfeo_toolbox import merge_rasters


def create_tile_mosaic(block):
    preped = prepInfiles(block['tiles'], '2A')

    before = time.time()

    main(
        preped,
        block['target_extent'],
        block['target_epsg'],
        resolution=block['resolution'],
        cloud_buffer=block['cloud_buffer'],
        output_dir=block['output_dir'],
        output_name=block['output_name'],
    )

    after = time.time()
    dif = after - before
    hours = int(dif / 3600)
    minutes = int((dif % 3600) / 60)
    seconds = "{0:.2f}".format(dif % 60)
    print(f"Calculated tile: {block['output_name']} in {hours}h {minutes}m {seconds}s")


if __name__ == '__main__':
    base = 'E:\\sentinel_2_data\\ghana\\wet_season_2019\\'
    files = glob(f"{base}*MSIL2A*.SAFE")

    output_dir = 'E:\\sentinel_2_data\\ghana\\wet_season_2019_mosaic\\'
    epsg = 32630
    cloud_buffer = 100
    resolution = 10

    tiles = []

    for f in files:
        tile_name = f.rsplit('_', 2)[1]
        if tile_name not in tiles:
            tiles.append(tile_name)

    blocks = []

    for tile in tiles:
        all_tile_files = glob(f"{base}*MSIL2A*{tile}*.SAFE")

        # Read first tile and calculate extent in correct epsg
        first_tile = all_tile_files[0]
        first_tile_df = gdal.Open(glob(f"{first_tile}\\GRANULE\\*\\IMG_DATA\\R10m\\*B04*.jp2")[0])

        x_min, xres, xskew, y_max, yskew, yres = first_tile_df.GetGeoTransform()
        x_max = x_min + (first_tile_df.RasterXSize * xres)
        y_min = y_max + (first_tile_df.RasterYSize * yres)

        bounding_box = [x_min, y_min, x_max, y_max]

        src_proj = osr.SpatialReference(wkt=first_tile_df.GetProjection())
        src_proj.AutoIdentifyEPSG()
        src_epsg = int(src_proj.GetAttrValue('AUTHORITY', 1))

        if src_epsg is epsg:
            target_bounds = bounding_box
        else:
            target_proj = osr.SpatialReference()
            target_proj.ImportFromEPSG(epsg)

            transform = osr.CoordinateTransformation(src_proj, target_proj)
            reproj_min = transform.TransformPoint(x_min, y_min)
            reproj_max = transform.TransformPoint(x_max, y_max)

            target_bounds = [int(reproj_min[0] + 1), int(reproj_min[1] + 1), int(reproj_max[0] + 1), int(reproj_max[1] + 1)]

        blocks.append({
            'output_name': tile,
            'output_dir': output_dir,
            'tiles': all_tile_files,
            'cloud_buffer': cloud_buffer,
            'resolution': resolution,
            'level': '2A',
            'target_epsg': epsg,
            'target_extent': target_bounds
        })

    pool = multiprocessing.Pool(1, maxtasksperchild=1)
    pool.map(create_tile_mosaic, blocks, chunksize=1)

    pool.close()
    pool.join()

    generated_hash = {}
    generated_files = glob(output_dir + '*.tif')

    for f in generated_files:
        tile_id = os.path.basename(f).split('_')[0]
        if tile_id in generated_hash:
            generated_hash[tile_id].append(f)
        else:
            generated_hash[tile_id] = [f]

    # Terrible way of getting the length of the first hashed key
    band_len = len(generated_hash[list(generated_hash.keys())[0]])

    matches = [[] for i in range(band_len)]

    for f in generated_hash:
        generated_hash[f].sort()

    for f in generated_hash:
        for index, value in enumerate(generated_hash[f]):
            matches[index].append(value)

    destination_folder = os.path.abspath(os.path.dirname(matches[0][0]) + '\\' + 'merged')

    for t in matches:
        name = os.path.basename(t[0]).split('_')
        name[0] = 'mosaic_cloud_100'

        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        destination = os.path.abspath(destination_folder + '\\' + '_'.join(name))

        merge_rasters(t, destination, tmp=output_dir)

    delete_files = glob(output_dir + '\\' + '[!merged]*')
    for f in delete_files:
        os.remove(f)

    # Filthy way of ensuring all the tmp files are deleted.
    time.sleep(5)

    delete_files = glob(output_dir + '\\' + '*.tif')
    for f in delete_files:
        os.remove(f)
