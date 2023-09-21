import os
import pandas as pd
import math
import requests
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import tqdm

FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/projects/unicef/"
FOLDER_OUT = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/projects/unicef/scraped_schools/"

# schools_gpkg = beo.vector_open(os.path.join(FOLDER, "complete_all_schools.gpkg"))
# attributes = beo.vector_get_attribute_table(schools_gpkg)

csv = pd.read_csv(os.path.join(FOLDER, "schools_complete_csv_abisubset.csv"))

# set seeed
np.random.seed(42)

latlng = csv[["fid", "X.chosen", "Y.chosen"]].to_numpy()
latlng = latlng[np.random.permutation(len(latlng))]


def latlon_to_tilexy(lat, lon, z):
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

for i, c in tqdm.tqdm(enumerate(latlng), total=len(latlng)):
    fid, x, y = c
    z = 18

    outname = f"{int(fid)}_{round(x, 6)}_{round(y, 6)}.png"

    if os.path.isfile(os.path.join(FOLDER_OUT, outname)):
        continue

    squares, (og_x, og_y) = latlon_to_tilexy(y, x, z)

    total_image = np.zeros((3, 512, 512), dtype=np.uint8)

    skip = False
    for i, s in enumerate(squares):
        if skip:
            continue
        x_tile, y_tile, z_tile = s
        url = f"https://mt1.google.com/vt/lyrs=s&x={x_tile}&y={y_tile}&z={z_tile}"
        tmp_path = os.path.abspath(os.path.join("./tmp", f"{x_tile}_{y_tile}_{z_tile}.png"))

        response = requests.get(url)

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

    label_y = round(og_y * 512)
    label_x = round(og_x * 512)

    adj = 128
    lxmin = max(0, label_x - adj)
    lxmax = min(511, label_x + adj)
    lymin = max(0, label_y - adj)
    lymax = min(511, label_y + adj)

    # Ensure that the clip is always 256 x 256
    if lxmax - lxmin < 256:
        lxmin = max(0, lxmax - 256)
    if lymax - lymin < 256:
        lymin = max(0, lymax - 256)

    total_image = total_image[:, lymin:lymax, lxmin:lxmax]
    total_image = total_image.transpose(1, 2, 0)

    fig, ax = plt.subplots(figsize=plt.figaspect(total_image))
    fig.subplots_adjust(0,0,1,1)
    ax.imshow(total_image)

    plt.savefig(os.path.join(FOLDER_OUT, outname))
    plt.close()
