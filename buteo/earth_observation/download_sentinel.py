import sys
import os
from tkinter.constants import E
import requests
import numpy as np
from osgeo import ogr
from sentinelsat import SentinelAPI
from collections import OrderedDict
from requests.auth import HTTPBasicAuth

from tqdm import tqdm

sys.path.append("../../")

from buteo.vector.io import internal_vector_to_metadata, filter_vector
from buteo.raster.io import raster_to_metadata
from buteo.gdal_utils import is_raster, is_vector
from buteo.earth_observation.s2_utils import timeout
from buteo.utils import progress


# api_url = "https://apihub.copernicus.eu/apihub"
api_url = "http://apihub.copernicus.eu/apihub"


def str_to_mb(string):
    split = string.split(" ")
    val = split[0]
    typ = split[1]

    if typ == "MB":
        return float(val)
    elif typ == "GB":
        return float(val) * 1000
    elif typ == "KB":
        return 1.0
    else:
        raise ValueError("Not MB, GB, or KB")


def arr_str_to_mb(arr):
    ret_arr = np.empty(len(arr), dtype="float32")

    for idx in range(len(arr)):
        ret_arr[idx] = str_to_mb(arr[idx])

    return ret_arr


def download(url: str, out_path: str, auth=None, verbose=False):
    resp = requests.get(url, stream=True, auth=auth)
    total = int(resp.headers.get("content-length", 0))

    if verbose:
        with open(out_path, "wb") as file, tqdm(
            desc=".." + out_path[-40:],
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    else:
        with open(out_path, "wb") as file:
            for data in resp.iter_content(chunk_size=1024):
                size = file.write(data)


# Only Ascending imagery available over Africa.
def download_s1_tile(
    scihub_username,
    scihub_password,
    onda_username,
    onda_password,
    destination,
    footprint,
    date=("20200601", "20210101"),
    orbitdirection="ASCENDING",  # ASCENDING, DESCENDING
    min_overlap=0.50,
    producttype="GRD",
    sensoroperationalmode="IW",
    polarisationmode="VV VH",
):
    api = SentinelAPI(scihub_username, scihub_password, api_url, timeout=60)

    if is_vector:
        geom = internal_vector_to_metadata(footprint, create_geometry=True)
    elif is_raster:
        geom = raster_to_metadata(footprint, create_geometry=True)

    products = api.query(
        geom["extent_wkt_latlng"],
        date=date,
        platformname="Sentinel-1",
        orbitdirection=orbitdirection,
        producttype=producttype,
        sensoroperationalmode=sensoroperationalmode,
        polarisationmode=polarisationmode,
        timeout=60,
    )

    download_products = OrderedDict()
    download_ids = []

    geom_footprint = ogr.CreateGeometryFromWkt(geom["extent_wkt_latlng"])

    for product in products:
        dic = products[product]

        img_footprint = ogr.CreateGeometryFromWkt(dic["footprint"])
        img_area = img_footprint.GetArea()

        intersection = img_footprint.Intersection(geom_footprint)

        within = img_footprint.Intersection(intersection)
        within_area = within.GetArea()

        overlap_img = within_area / img_area
        overlap_geom = within_area / geom_footprint.GetArea()

        if max(overlap_img, overlap_geom) > min_overlap:
            download_products[product] = dic

            download_ids.append(product)

    if len(download_products) > 0:
        print(f"Downloading {len(download_products)} files.")

        downloaded = []
        for img_id in download_ids:
            out_path = destination + download_products[img_id]["filename"] + ".zip"

            if os.path.exists(out_path):
                print(f"Skipping {out_path}")
                continue

            download_url = f"https://catalogue.onda-dias.eu/dias-catalogue/Products({img_id})/$value"

            try:
                download(
                    download_url,
                    out_path,
                    auth=HTTPBasicAuth(onda_username, onda_password),
                )

                downloaded.append(out_path)
            except Exception as e:
                print(f"Error downloading {img_id}: {e}")

        return downloaded
    else:
        print("No images found")
        return []


def download_s2_tile(
    scihub_username,
    scihub_password,
    onda_username,
    onda_password,
    destination,
    aoi_vector,
    date_start="20200601",
    date_end="20210101",
    clouds=10,
    producttype="S2MSI2A",
    tile=None,
):
    print("Downloading Sentinel-2 tiles")
    try:
        api = SentinelAPI(scihub_username, scihub_password, api_url, timeout=60)
    except Exception as e:
        print(e)
        raise Exception("Error connecting to SciHub")

    if is_vector(aoi_vector):
        geom = internal_vector_to_metadata(aoi_vector, create_geometry=True)
    elif is_raster(aoi_vector):
        geom = raster_to_metadata(aoi_vector, create_geometry=True)

    geom_extent = geom["extent_wkt_latlng"]

    download_products = OrderedDict()
    download_ids = []

    date = (date_start, date_end)

    if tile is not None and tile != "":
        kw = {"raw": f"tileid:{tile} OR filename:*_T{tile}_*"}

        try:
            products = api.query(
                date=date,
                platformname="Sentinel-2",
                cloudcoverpercentage=(0, clouds),
                producttype="S2MSI2A",
                timeout=60,
                **kw,
            )
        except Exception as e:
            print(e)
            raise Exception("Error connecting to SciHub")
    else:
        try:
            products = api.query(
                geom_extent,
                date=date,
                platformname="Sentinel-2",
                cloudcoverpercentage=(0, clouds),
                producttype=producttype,
            )

        except Exception as e:
            print(e)
            raise Exception("Error connecting to SciHub")

    for product in products:
        dic = products[product]

        product_tile = dic["title"].split("_")[-2][1:]
        if (tile is not None and tile != "") and product_tile != tile:
            continue

        download_products[product] = dic
        download_ids.append(product)

    print(f"Downloading {len(download_products)} tiles")

    downloaded = []
    for img_id in download_ids:
        out_path = destination + download_products[img_id]["filename"] + ".zip"

        if os.path.exists(out_path):
            print(f"Skipping {out_path}")
            continue

        download_url = (
            f"https://catalogue.onda-dias.eu/dias-catalogue/Products({img_id})/$value"
        )

        try:
            print(f"Downloading: {img_id}")
            download(
                download_url,
                out_path,
                auth=HTTPBasicAuth(onda_username, onda_password),
                verbose=False,
            )
            print(f"Downloaded: {img_id}")

            downloaded.append(out_path)
        except Exception as e:
            print(f"Error downloading {img_id}: {e}")

    return downloaded
