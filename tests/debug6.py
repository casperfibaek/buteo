import os
import sys; sys.path.append("../")
import pandas as pd
import math
import tqdm
import requests
import numpy as np
import buteo as beo
from osgeo import gdal


def latlon_to_tilexy(lat: float, lon: float, z: int):
    """
    Convert lat/lon to tile x/y

    Parameters
    ----------
    lat : float
        Latitude
    lon : float
        Longitude
    z : int
        Zoom level

    Returns
    -------
    tuple(list, tuple)
        List of tiles, and tuple of x/y
    """
    n = 2.0 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    
    x0 = round(x)
    y0 = round(y)

    x1 = x0 - 1 if x < x0 else x0 - 1
    y1 = y0 - 1 if y < y0 else y0 - 1

    squares = [
        (min(x0, x1), min(y0, y1), z),
        (min(x0, x1), max(y0, y1), z),
        (max(x0, x1), max(y0, y1), z),
        (max(x0, x1), min(y0, y1), z)
    ]

    if round(x) == max(x0, x1):
        x = 0.5 + ((x - max(x0, x1)) / 2.0)
    else:
        x = (x - min(x0, x1)) / 2.0

    if round(y) == max(y0, y1):
        y = 0.5 + ((y - max(y0, y1)) / 2.0)
    else:
        y = (y - min(y0, y1)) / 2.0

    return squares, (x, y)

if __name__ == "__main__":
    from time import sleep
    FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/projects/unicef/"
    FOLDER_OUT = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/projects/unicef/scraped_schools_sudan/"
    MAX_IMAGES = 25000

    csv = pd.read_csv(os.path.join(FOLDER, "sudan_schools.csv"))

    np.random.seed(42)
    convertor = beo.GlobalMercator(tileSize=256)

    latlng = csv[["fid", "lon", "lat"]].to_numpy()
    latlng = latlng[np.random.permutation(len(latlng))]
    latlng = latlng[:MAX_IMAGES]

    def main():
        for i, c in tqdm.tqdm(enumerate(latlng), total=len(latlng)):
            for BING in [True, False]:
                for px_size in [0.5, 2.0, 5.0]:
                    z = convertor.ZoomForPixelSize(px_size)
                    fid, x, y = c

                    source = 0 if BING else 1

                    outname = f"i{int(fid)}_x{round(x, 6)}_y{round(y, 6)}_z{z}_s{source}.tif"

                    if os.path.isfile(os.path.join(FOLDER_OUT, outname)):
                        continue

                    squares, (og_x, og_y) = latlon_to_tilexy(y, x, z)

                    total_image = np.zeros((3, 512, 512), dtype=np.uint8)

                    pixel_sizes = []
                    x_min, y_min, x_max, y_max = None, None, None, None

                    skip = False
                    for i, s in enumerate(squares):
                        if skip:
                            continue
                        x_tile, y_tile, z_tile = s

                        tms_x_tile, tms_y_tile = convertor.GoogleToTMSTile(x_tile, y_tile, z_tile)
                        minx, miny, maxx, maxy = convertor.TileBounds(tms_x_tile, tms_y_tile, z_tile)

                        if x_min is None:
                            x_min, y_min, x_max, y_max = minx, miny, maxx, maxy
                        else:
                            x_min = min(x_min, minx)
                            y_min = min(y_min, miny)
                            x_max = max(x_max, maxx)
                            y_max = max(y_max, maxy)

                        pixel_sizes.append((maxx - minx) / 256)
                        pixel_sizes.append((maxy - miny) / 256)

                        if BING:
                            q = convertor.QuadTree(tms_x_tile, tms_y_tile, z_tile)
                            url = f"https://ecn.t3.tiles.virtualearth.net/tiles/a{q}.jpeg?g=1"
                            tmp_path = os.path.abspath(os.path.join("./tmp", f"{x_tile}_{y_tile}_{z_tile}.jpeg"))
                        else:
                            url = f"https://mt1.google.com/vt/lyrs=s&x={x_tile}&y={y_tile}&z={z_tile}"
                            tmp_path = os.path.abspath(os.path.join("./tmp", f"{x_tile}_{y_tile}_{z_tile}.png"))

                        response = requests.get(url)

                        # if respose is not ok, skip
                        if response.status_code != 200:
                            skip = True
                            continue

                        with open(f'{tmp_path}', 'wb') as file:
                            file.write(response.content)

                        try:
                            # Read the raster band as numpy array
                            array = gdal.Open(tmp_path).ReadAsArray()

                            if i == 0:
                                total_image[:, 0:256, 0:256] = array
                            elif i == 3:
                                total_image[:, 0:256, 256:512] = array
                            elif i == 2:
                                total_image[:, 256:512, 256:512] = array
                            elif i == 1:
                                total_image[:, 256:512, 0:256] = array
                            
                        except:
                            skip = True
                        finally:
                            os.remove(tmp_path)

                    if skip:
                        continue

                    total_image = total_image.transpose(1, 2, 0)

                    raster_tmp = beo.raster_create_from_array(
                        total_image,
                        out_path=None,
                        pixel_size=np.array(pixel_sizes).mean(),
                        x_min=x_min,
                        y_max=y_max,
                    )

                    label_y = max(0, round(og_y * 512) - 128)
                    label_x = max(0, round(og_x * 512) - 128)

                    beo.array_to_raster(
                        beo.raster_to_array(raster_tmp, pixel_offsets=[label_x, label_y, 256, 256]),
                        out_path=os.path.join(FOLDER_OUT, outname),
                        reference=raster_tmp,
                        pixel_offsets=[label_x, label_y, 256, 256],
                    )

                    beo.delete_dataset_if_in_memory(raster_tmp)

    try:
        main()
    except:
        sleep(60 * 5)
        main()
