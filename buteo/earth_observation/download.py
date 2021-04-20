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


def download_s1(
    footprint,
    username,
    password,
    destination,
    date=("20200601", "20210101"),
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

    download = api.download_all(products, directory_path=destination)

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


def download_s2(
    username,
    password,
    destination,
    footprint=None,
    tiles=[],
    date=("20200601", "20210101"),
    clouds=30,
    min_size=50,
    overlap=0.5,
    min_update=0.01,
    iterate=False,
    _iteration=0,
    _coverage=0.0,
    _union=0.0,
):
    if _iteration > 10 or clouds > 100:
        print("Ended due to iteration or cloud limit.")
        return None

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
            geom_extent = geom_meta["extent_wkt_latlng"]

            tile_products = api.query(
                geom_extent,
                date=date,
                platformname="Sentinel-2",
                cloudcoverpercentage=(0, clouds),
                producttype="S2MSI2A",
            )
            products.update(tile_products)
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

        img_footprint = ogr.CreateGeometryFromWkt(dic["footprint"])

        intersection = img_footprint.Intersection(geom_footprint)
        intersection_area = intersection.GetArea()

        if _iteration == 0 and (intersection_area / geom_area) > overlap:
            download_products[product] = dic

        if union == 0.0 and (intersection_area / geom_area) > overlap:
            size = str_to_mb(dic["size"])

            if size > min_size:
                union = intersection
                coverage = intersection_area / geom_area

                download_products[product] = dic
        elif union != 0.0:
            comp = union.Union(intersection)
            cover = comp.GetArea() / geom_area

            if _iteration == 0 or (cover - coverage) > min_update:
                size = str_to_mb(dic["size"])

                if size > min_size:
                    union = comp
                    coverage = cover

                    download_products[product] = dic

    if len(download_products) == 0:
        print("Download list was empty")
        return download_products
    else:
        download = api.download_all(download_products, directory_path=destination)

    if iterate and clouds <= 100:
        if coverage < 0.98:

            print(
                f"Completed. Iteration: {_iteration} - Cloud cover: {clouds}% - Coverage: {round(coverage * 100, 3)}%"
            )

            download_s2(
                username,
                password,
                destination,
                footprint=footprint,
                tiles=tiles,
                date=date,
                clouds=clouds + 10,
                min_size=min_size,
                overlap=overlap,
                iterate=iterate,
                _iteration=_iteration + 1,
                _union=union,
            )

    return download


# Only Ascending imagery available over Africa.
if __name__ == "__main__":
    from glob import glob

    folder = "C:/Users/caspe/Desktop/test/"
    vector = folder + "tema.gpkg"
    dst = folder + "s2_download/"

    # avai = download_s2(
    #     "casperfibaek2",
    #     "Goldfish12",
    #     dst,
    #     tiles=["32VNH"],
    #     date=("20210201", "20210420"),
    #     clouds=10,
    #     min_update=0.01,
    #     iterate=True,
    #     # orbitdirection="Ascending",
    # )

    s2_files = glob(dst + "*.safe")
    align
