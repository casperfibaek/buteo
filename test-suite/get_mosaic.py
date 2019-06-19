from osgeo import gdal, osr
from glob import glob
import time
import multiprocessing
import sys

sys.path.append('../lib')

from utilities import prepInfiles
from mosaic import main


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
    base = 'E:\\sentinel_2_data\\ghana\\wet_season_2019\\NZN\\'
    files = glob(f"{base}*MSIL2A*.SAFE")

    output_dir = 'E:\\sentinel_2_data\\ghana\\wet_season_2019\\NZN\\'
    epsg = 32630
    cloud_buffer = 200
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

    pool = multiprocessing.Pool(2, maxtasksperchild=1)
    pool.map(create_tile_mosaic, blocks, chunksize=1)

    pool.close()
    print('Closed pool.')

    pool.join()
    print('Joined pool.')
