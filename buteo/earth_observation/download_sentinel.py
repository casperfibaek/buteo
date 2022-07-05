"""
This module is used to download Sentinel data from the Onda DIAS catalogue and ESA SciHub.

TODO:
    - Improve documentation
"""

import sys; sys.path.append("../../") # Path: buteo/earth_observation/download_sentinel.py
import os
from time import sleep
from collections import OrderedDict

import requests
from osgeo import ogr
from sentinelsat import SentinelAPI
from requests.auth import HTTPBasicAuth
from tqdm import tqdm

from buteo.vector.io import internal_vector_to_metadata
from buteo.raster.io import raster_to_metadata
from buteo.gdal_utils import is_raster, is_vector


def get_content_size(url: str, auth=None):
    resp = requests.get(url, stream=True, auth=auth)
    total = int(resp.headers.get("content-length", 0))

    return total


def order(url, auth=None):
    resp = requests.post(url, stream=True, auth=auth)

    json = {}
    try:
        json = resp.json()
        status = json["Status"]
        status_message = json["StatusMessage"]
        estimated_time = json["EstimatedTime"]

        print(f"{status}: {status_message}")
        print(f"Estimated ready time: {estimated_time}")
    except Exception as e:
        print(f"Error while ordering, {e}")

    return json


def download(url: str, out_path: str, auth=None, verbose=False, skip_if_exists=False):
    resp = requests.get(url, stream=True, auth=auth)
    total = int(resp.headers.get("content-length", 0))

    if (
        skip_if_exists
        and (os.path.exists(out_path) and os.path.getsize(out_path) == total)
        or total == 0
    ):
        print(f"Skipping {out_path}")
        return

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
    date_start="20200601",
    date_end="20210101",
    orbitdirection="ASCENDING",  # ASCENDING, DESCENDING
    min_overlap=0.50,
    producttype="GRD",
    sensoroperationalmode="IW",
    polarisationmode="VV VH",
    api_url="https://apihub.copernicus.eu/apihub/",
):
    api = SentinelAPI(scihub_username, scihub_password, api_url, timeout=60)

    if is_vector(footprint):
        geom = internal_vector_to_metadata(footprint, create_geometry=True)
    elif is_raster(footprint):
        geom = raster_to_metadata(footprint, create_geometry=True)

    date = (date_start, date_end)

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
    zero_contents = 0

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
                content_size = get_content_size(
                    download_url, auth=(onda_username, onda_password)
                )
            except Exception as e:
                print(f"Failed to get content size for {img_id}")
                print(e)
                continue

            if content_size == 0:
                zero_contents += 1
                print(f"{img_id} requested from Archive but was not downloaded.")
                continue

            try:
                if content_size > 0:
                    download(
                        download_url,
                        out_path,
                        auth=(onda_username, onda_password),
                        verbose=True,
                    )
                    downloaded.append(img_id)
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
    retry_count=10,
    retry_wait_min=30,
    retry_current=0,
    retry_downloaded=[],
    api_url="http://apihub.copernicus.eu/apihub",
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

    downloaded = [] + retry_downloaded
    for img_id in download_ids:
        out_path = destination + download_products[img_id]["filename"] + ".zip"

        if out_path in downloaded:
            continue

        # /footprint url for.
        download_url = (
            f"https://catalogue.onda-dias.eu/dias-catalogue/Products({img_id})/$value"
        )

        try:
            content_size = get_content_size(
                download_url, auth=(onda_username, onda_password)
            )
        except Exception as e:
            print(f"Failed to get content size for {img_id}")
            print(e)
            continue

        try:
            if content_size > 0:

                if os.path.isfile(out_path) and content_size == os.path.getsize(
                    out_path
                ):
                    downloaded.append(out_path)
                    print(f"Skipping {img_id}")
                else:
                    print(f"Downloading: {img_id}")
                    download(
                        download_url,
                        out_path,
                        auth=HTTPBasicAuth(onda_username, onda_password),
                        verbose=False,
                        skip_if_exists=True,
                    )

                    downloaded.append(out_path)
            else:
                print("Requesting from archive. Not downloaded.")
                order_url = f"https://catalogue.onda-dias.eu/dias-catalogue/Products({img_id})/Ens.Order"
                order_response = order(order_url, auth=(onda_username, onda_password))

        except Exception as e:
            print(f"Error downloading {img_id}: {e}")

    if len(downloaded) >= len(download_ids):
        return downloaded
    elif retry_current < retry_count:
        print(
            f"Retrying {retry_current}/{retry_count}. Sleeping for {retry_wait_min} minutes."
        )
        sleep(retry_wait_min * 60)
        download_s2_tile(
            scihub_username,
            scihub_password,
            onda_username,
            onda_password,
            destination,
            aoi_vector,
            date_start=date_start,
            date_end=date_end,
            clouds=clouds,
            producttype=producttype,
            tile=tile,
            retry_count=retry_count,
            retry_wait_min=retry_wait_min,
            retry_current=retry_current + 1,
            retry_downloaded=retry_downloaded + downloaded,
        )
    else:
        return retry_downloaded + downloaded
