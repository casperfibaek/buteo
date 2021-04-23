import sys

sys.path.append("../../")
from osgeo import ogr
from sentinelsat import SentinelAPI
from collections import OrderedDict
from buteo.vector.io import internal_vector_to_metadata, filter_vector
import numpy as np


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


def list_available_s1(
    username,
    password,
    footprint=None,
    date=("20201001", "20210101"),
    orbitdirection="Ascending",  # Ascending, Descending
    producttype="GRD",
):
    api = SentinelAPI(username, password, "https://scihub.copernicus.eu/apihub")

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


# Only Ascending imagery available over Africa.
def download_s1(
    username,
    password,
    footprint,
    destination,
    date=("20200601", "20210101"),
    orbitdirection="ASCENDING",  # ASCENDING, DESCENDING
    min_overlap=0.25,
    producttype="GRD",
    sensoroperationalmode="IW",
    polarisationmode="VV VH",
):
    api = SentinelAPI(username, password, "https://scihub.copernicus.eu/apihub")

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

    geom_footprint = ogr.CreateGeometryFromWkt(geom["extent_wkt_latlng"])
    geom_area = geom_footprint.GetArea()

    for product in products:
        dic = products[product]

        img_footprint = ogr.CreateGeometryFromWkt(dic["footprint"])

        intersection = img_footprint.Intersection(geom_footprint)
        intersection_area = intersection.GetArea()

        overlap = intersection_area / geom_area

        if overlap > min_overlap:
            download_products[product] = dic

    print(f"Downloading {len(download_products)} files.")
    download = api.download_all(download_products, directory_path=destination)

    return download


def list_available_s2(
    username,
    password,
    footprint=None,
    tiles=[],
    date=("20200601", "20210101"),
    clouds=20,
    min_size=500,
):
    api = SentinelAPI(username, password, "https://scihub.copernicus.eu/apihub")

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


# Update to have at least x of tile
def download_s2(
    username,
    password,
    destination,
    footprint=None,
    tile=None,
    date=("20200601", "20210101"),
    only_specified_tile=True,
    clouds=20,
    min_size=50,
    min_update=0.01,
    min_overlap=0.33,
    min_images=5,
    iterate=False,
    _iteration=0,
    _coverage=0.0,
    _union=0.0,
    _added_images=0,
):
    if _iteration > 5 or clouds > 100:
        print("Ended due to iteration or cloud limit.")
        return None

    api = SentinelAPI(username, password, "https://scihub.copernicus.eu/apihub")

    if footprint is None and tile is None:
        raise ValueError("Either footprint or tilesnames must be supplied.")

    if tile is not None:
        if _iteration == 0:
            print(f"Processing tile: {tile}")

        geom = filter_vector(
            "../../geometry/sentinel2_tiles_world.shp", filter_where=("Name", tile)
        )

        geom_meta = internal_vector_to_metadata(geom, create_geometry=True)
        geom_extent = geom_meta["extent_wkt_latlng"]

        kw = {"raw": f"tileid:{tile} OR filename:*_T{tile}_*"}

        products = api.query(
            geom_extent,
            date=date,
            platformname="Sentinel-2",
            cloudcoverpercentage=(0, clouds),
            producttype="S2MSI2A",
            **kw,
        )
    else:
        geom_meta = internal_vector_to_metadata(footprint, create_geometry=True)
        geom_extent = geom_meta["extent_wkt_latlng"]

        products = api.query(
            geom_extent,
            date=date,
            platformname="Sentinel-2",
            cloudcoverpercentage=(0, clouds),
            producttype="S2MSI2A",
        )

    download_products = OrderedDict()

    union = _union
    coverage = _coverage
    geom_footprint = ogr.CreateGeometryFromWkt(geom_extent)
    geom_area = geom_footprint.GetArea()

    for product in products:
        dic = products[product]
        product_tile = dic["title"].split("_")[-2][1:]

        if tile is not None and only_specified_tile and product_tile != tile:
            continue

        img_footprint = ogr.CreateGeometryFromWkt(dic["footprint"])

        intersection = img_footprint.Intersection(geom_footprint)
        intersection_area = intersection.GetArea()

        overlap = intersection_area / geom_area

        if _iteration == 0 and overlap > min_overlap:
            download_products[product] = dic

            if tile is not None and product_tile != tile:
                _added_images += 1

        elif union == 0.0 and overlap > min_overlap:
            size = str_to_mb(dic["size"])

            if size > min_size:
                union = intersection
                coverage = overlap

                download_products[product] = dic

                if tile is not None and product_tile != tile:
                    _added_images += 1

        elif union != 0.0:
            comp = union.Union(intersection)
            cover = comp.GetArea() / geom_area

            if _iteration == 0 or (cover - coverage) > min_update:
                size = str_to_mb(dic["size"])

                if size > min_size:
                    union = comp
                    coverage = cover

                    download_products[product] = dic

                    if tile is not None and product_tile != tile:
                        _added_images += 1

    if tile is not None:
        if coverage > 95 and _added_images < min_images:
            for product in products:
                dic = products[product]
                product_tile = dic["title"].split("_")[-2][1:]

                img_footprint = ogr.CreateGeometryFromWkt(dic["footprint"])

                intersection = img_footprint.Intersection(geom_footprint)
                intersection_area = intersection.GetArea()

                overlap = intersection_area / geom_area

                if product_tile != tile:
                    continue

                if overlap > min_overlap:
                    union = union.Union(intersection)
                    coverage = comp.GetArea() / geom_area

                    download_products[product] = dic

                    _added_images += 1

    if len(download_products) == 0:
        print("Download list was empty")
        return download_products
    else:
        if _iteration == 0:
            print(f"Downloading {len(download_products)} tiles")

        download = api.download_all(download_products, directory_path=destination)

    if iterate and clouds <= 100:
        if coverage < 0.975 or _added_images < min_images:

            print(
                f"Completed. Iteration: {_iteration} - Cloud cover: {clouds}% - Coverage: {round(coverage * 100, 3)}%"
            )

            download_s2(
                username,
                password,
                destination,
                footprint=footprint,
                tile=tile,
                date=date,
                clouds=clouds + 10,
                min_size=min_size,
                min_overlap=min_overlap,
                iterate=iterate,
                _iteration=_iteration + 1,
                _union=union,
                _added_images=_added_images,
            )

    return download


if __name__ == "__main__":
    folder = "C:/Users/caspe/Desktop/paper_transfer_learning/data/"
    dst = folder + "sentinel1/raw_2020/"

    vector = folder + "denmark_polygon_border_region_removed.gpkg"

    # 2020 06 01 - 2020 08 01 (good dates: 0615-0701)
    # 2021 02 15 - 2021 04 15

    avai = download_s1(
        "casperfibaek",
        "Goldfish12",
        vector,
        dst,
        date=("20200601", "20200630"),
        min_overlap=0.1,
    )

    import pdb

    pdb.set_trace()
