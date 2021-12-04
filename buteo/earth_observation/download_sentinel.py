import sys
import os
import requests
import numpy as np
from osgeo import ogr
from sentinelsat import SentinelAPI
from collections import OrderedDict
from requests.auth import HTTPBasicAuth
from tqdm import tqdm

sys.path.append("../../")

from buteo.vector.io import internal_vector_to_metadata, filter_vector


api_url = "https://apihub.copernicus.eu/apihub"


def str_to_mb(str):
    split = str.split(" ")
    val = split[0]
    typ = split[1]

    if typ == "MB":
        return float(val)
    elif typ == "GB":
        return float(val) * 1000
    elif typ == "KB":
        return 1.0
    else:
        raise ValueError("Not MB or GB")


def arr_str_to_mb(arr):
    ret_arr = np.empty(len(arr), dtype="float32")

    for idx in range(len(arr)):
        ret_arr[idx] = str_to_mb(arr[idx])

    return ret_arr


def download(url: str, out_path: str, auth=None):
    resp = requests.get(url, stream=True, auth=auth)
    total = int(resp.headers.get("content-length", 0))
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


def list_available_s1(
    username,
    password,
    footprint=None,
    date=("20201001", "20210101"),
    orbitdirection="Ascending",  # Ascending, Descending
    producttype="GRD",
):
    api = SentinelAPI(username, password, api_url)

    geom = internal_vector_to_metadata(footprint, create_geometry=True)

    products = api.query(
        geom["extent_wkt_latlng"],
        date=date,
        platformname="Sentinel-1",
        orbitdirection=orbitdirection,
        producttype=producttype,
        sensoroperationalmode="IW",
    )

    df = api.to_geodataframe(products)

    return df


def list_available_s2(
    username,
    password,
    footprint=None,
    tiles=[],
    date=("20200601", "20210101"),
    clouds=20,
    min_size=500,
):
    api = SentinelAPI(username, password, api_url)

    if footprint is None and len(tiles) == 0:
        raise ValueError("Either footprint or tilesnames must be supplied.")

    if len(tiles) > 0:
        products = OrderedDict()

        for tile in tiles:
            geom = filter_vector(
                "../../geometry/sentinel2_tiles_world.shp", filter_where=("Name", tile)
            )

            geom_meta = internal_vector_to_metadata(geom, create_geometry=True)

            tile_products = api.query(
                geom_meta["extent_wkt_latlng"],
                date=date,
                platformname="Sentinel-2",
                cloudcoverpercentage=(0, clouds),
                producttype="S2MSI2A",
            )
            products.update(tile_products)
    else:
        geom = internal_vector_to_metadata(footprint, create_geometry=True)

        products = api.query(
            geom["extent_wkt_latlng"],
            date=date,
            platformname="Sentinel-2",
            cloudcoverpercentage=(0, clouds),
            producttype="S2MSI2A",
        )

    df = api.to_geodataframe(products)

    # reduce df to above min_size
    dfs = df[arr_str_to_mb(df["size"].values) > min_size]

    return dfs


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
    api = SentinelAPI(scihub_username, scihub_password, api_url)

    geom = internal_vector_to_metadata(footprint, create_geometry=True)

    products = api.query(
        geom["extent_wkt_latlng"],
        date=date,
        platformname="Sentinel-1",
        orbitdirection=orbitdirection,
        producttype=producttype,
        sensoroperationalmode=sensoroperationalmode,
        polarisationmode=polarisationmode,
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
    tile,
    date=("20200601", "20210101"),
    clouds=10,
    min_size=100,
):
    api = SentinelAPI(scihub_username, scihub_password, api_url)

    geom = filter_vector(
        "../../geometry/sentinel2_tiles_world.shp", filter_where=("Name", tile)
    )

    geom_meta = internal_vector_to_metadata(geom, create_geometry=True)
    geom_extent = geom_meta["extent_wkt_latlng"]

    kw = {"raw": f"tileid:{tile} OR filename:*_T{tile}_*"}

    download_products = OrderedDict()
    download_ids = []

    products = api.query(
        geom_extent,
        date=date,
        platformname="Sentinel-2",
        cloudcoverpercentage=(0, clouds),
        producttype="S2MSI2A",
        **kw,
    )

    for product in products:
        dic = products[product]

        product_tile = dic["title"].split("_")[-2][1:]
        if product_tile != tile:
            continue

        size = str_to_mb(dic["size"])
        if size < min_size:
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
            download(
                download_url, out_path, auth=HTTPBasicAuth(onda_username, onda_password)
            )

            downloaded.append(out_path)
        except Exception as e:
            print(f"Error downloading {img_id}: {e}")

    return downloaded
