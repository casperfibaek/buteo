import sys
import os
import signal
import xml.etree.ElementTree as ET
from datetime import datetime
from zipfile import ZipFile
from glob import glob

sys.path.append("../../")

from buteo.vector.io import filter_vector
from buteo.vector.intersect import intersect_vector
from buteo.vector.clip import clip_vector
from buteo.vector.attributes import vector_get_attribute_table
from buteo.vector.reproject import reproject_vector



class TimeoutError(Exception):
    def __init__(self, value = "Timed Out"):
        self.value = value
    def __str__(self):
        return repr(self.value)

def timeout(seconds_before_timeout):
    def decorate(f):
        def handler(signum, frame):
            raise TimeoutError()
        def new_f(*args, **kwargs):
            old = signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds_before_timeout)
            try:
                result = f(*args, **kwargs)
            finally:
                # reinstall the old signal handler
                signal.signal(signal.SIGALRM, old)
                # cancel the alarm
                # this line should be inside the "finally" block (per Sam Kortchmar)
                signal.alarm(0)
            return result
        new_f.func_name = f.func_name
        return new_f
    return decorate


def get_band_paths(safe_folder):
    bands = {
        "10m": {"B02": None, "B03": None, "B04": None, "B08": None, "AOT": None},
        "20m": {
            "B02": None,
            "B03": None,
            "B04": None,
            "B05": None,
            "B06": None,
            "B07": None,
            "B8A": None,
            "B11": None,
            "B12": None,
            "SCL": None,
            "AOT": None,
        },
        "60m": {
            "B01": None,
            "B02": None,
            "B03": None,
            "B04": None,
            "B05": None,
            "B06": None,
            "B07": None,
            "B8A": None,
            "B09": None,
            "B11": None,
            "B12": None,
            "SCL": None,
        },
        "QI": {
            "CLDPRB_20m": None,
            "CLDPRB_60m": None,
        },
    }

    assert os.path.isdir(safe_folder), f"Could not find folder: {safe_folder}"

    bands["QI"]["CLDPRB_20m"] = glob(
        f"{safe_folder}/GRANULE/*/QI_DATA/MSK_CLDPRB_20m.jp2"
    )[0]
    bands["QI"]["CLDPRB_60m"] = glob(
        f"{safe_folder}/GRANULE/*/QI_DATA/MSK_CLDPRB_60m.jp2"
    )[0]

    bands_10m = glob(f"{safe_folder}/GRANULE/*/IMG_DATA/R10m/*_???_*.jp2")
    for band in bands_10m:
        basename = os.path.basename(band)
        band_name = basename.split("_")[2]
        if band_name == "B02":
            bands["10m"]["B02"] = band
        if band_name == "B03":
            bands["10m"]["B03"] = band
        if band_name == "B04":
            bands["10m"]["B04"] = band
        if band_name == "B08":
            bands["10m"]["B08"] = band
        if band_name == "AOT":
            bands["10m"]["AOT"] = band

    bands_20m = glob(f"{safe_folder}/GRANULE/*/IMG_DATA/R20m/*.jp2")
    for band in bands_20m:
        basename = os.path.basename(band)
        band_name = basename.split("_")[2]
        if band_name == "B02":
            bands["20m"]["B02"] = band
        if band_name == "B03":
            bands["20m"]["B03"] = band
        if band_name == "B04":
            bands["20m"]["B04"] = band
        if band_name == "B05":
            bands["20m"]["B05"] = band
        if band_name == "B06":
            bands["20m"]["B06"] = band
        if band_name == "B07":
            bands["20m"]["B07"] = band
        if band_name == "B8A":
            bands["20m"]["B8A"] = band
        if band_name == "B09":
            bands["20m"]["B09"] = band
        if band_name == "B11":
            bands["20m"]["B11"] = band
        if band_name == "B12":
            bands["20m"]["B12"] = band
        if band_name == "SCL":
            bands["20m"]["SCL"] = band
        if band_name == "AOT":
            bands["20m"]["AOT"] = band

    bands_60m = glob(f"{safe_folder}/GRANULE/*/IMG_DATA/R60m/*_???_*.jp2")
    for band in bands_60m:
        basename = os.path.basename(band)
        band_name = basename.split("_")[2]
        if band_name == "B01":
            bands["60m"]["B01"] = band
        if band_name == "B02":
            bands["60m"]["B02"] = band
        if band_name == "B03":
            bands["60m"]["B03"] = band
        if band_name == "B04":
            bands["60m"]["B04"] = band
        if band_name == "B05":
            bands["60m"]["B05"] = band
        if band_name == "B06":
            bands["60m"]["B06"] = band
        if band_name == "B07":
            bands["60m"]["B07"] = band
        if band_name == "B8A":
            bands["60m"]["B8A"] = band
        if band_name == "B09":
            bands["60m"]["B09"] = band
        if band_name == "B11":
            bands["60m"]["B11"] = band
        if band_name == "B12":
            bands["60m"]["B12"] = band
        if band_name == "SCL":
            bands["60m"]["SCL"] = band
        if band_name == "AOT":
            bands["60m"]["AOT"] = band

    for outer_key in bands:
        for inner_key in bands[outer_key]:
            current_band = bands[outer_key][inner_key]
            assert (
                current_band != None
            ), f"{outer_key} - {inner_key} was not found. Verify the folders. Was the decompression interrupted?"

    return bands


def get_metadata(safe_folder):
    metadata = {
        "PRODUCT_START_TIME": None,
        "PRODUCT_STOP_TIME": None,
        "PRODUCT_URI": None,
        "PROCESSING_LEVEL": None,
        "PRODUCT_TYPE": None,
        "PROCESSING_BASELINE": None,
        "GENERATION_TIME": None,
        "SPACECRAFT_NAME": None,
        "DATATAKE_SENSING_START": None,
        "SENSING_ORBIT_NUMBER": None,
        "SENSING_ORBIT_DIRECTION": None,
        "EXT_POS_LIST": None,
        "Cloud_Coverage_Assessment": None,
        "NODATA_PIXEL_PERCENTAGE": None,
        "SATURATED_DEFECTIVE_PIXEL_PERCENTAGE": None,
        "DARK_FEATURES_PERCENTAGE": None,
        "CLOUD_SHADOW_PERCENTAGE": None,
        "VEGETATION_PERCENTAGE": None,
        "NOT_VEGETATED_PERCENTAGE": None,
        "WATER_PERCENTAGE": None,
        "UNCLASSIFIED_PERCENTAGE": None,
        "MEDIUM_PROBA_CLOUDS_PERCENTAGE": None,
        "HIGH_PROBA_CLOUDS_PERCENTAGE": None,
        "THIN_CIRRUS_PERCENTAGE": None,
        "SNOW_ICE_PERCENTAGE": None,
        "ZENITH_ANGLE": None,
        "AZIMUTH_ANGLE": None,
        "SUN_ELEVATION": None,
        "folder": safe_folder,
        "gains": {},
    }

    meta_xml = os.path.join(safe_folder, "MTD_MSIL2A.xml")
    meta_solar = glob(safe_folder + "/GRANULE/*/MTD_TL.xml")[0]

    assert os.path.isfile(
        meta_xml
    ), f"{safe_folder} did not contain a valid metadata file."
    assert os.path.isfile(
        meta_solar
    ), f"{meta_solar} did not contain a valid metadata file."

    # Parse the xml tree and add metadata
    root = ET.parse(meta_xml).getroot()
    for elem in root.iter():
        if elem.tag in metadata:
            try:
                metadata[elem.tag] = float(elem.text)  # Number?
            except:
                try:
                    metadata[elem.tag] = datetime.strptime(
                        elem.text, "%Y-%m-%dT%H:%M:%S.%f%z"
                    )  # Date?
                except:
                    metadata[elem.tag] = elem.text
        if elem.tag == "PHYSICAL_GAINS":
            if elem.attrib["bandId"] == "0":
                metadata["gains"]["B01"] = float(elem.text)
            if elem.attrib["bandId"] == "1":
                metadata["gains"]["B02"] = float(elem.text)
            if elem.attrib["bandId"] == "2":
                metadata["gains"]["B03"] = float(elem.text)
            if elem.attrib["bandId"] == "3":
                metadata["gains"]["B04"] = float(elem.text)
            if elem.attrib["bandId"] == "4":
                metadata["gains"]["B05"] = float(elem.text)
            if elem.attrib["bandId"] == "5":
                metadata["gains"]["B06"] = float(elem.text)
            if elem.attrib["bandId"] == "6":
                metadata["gains"]["B07"] = float(elem.text)
            if elem.attrib["bandId"] == "7":
                metadata["gains"]["B08"] = float(elem.text)
            if elem.attrib["bandId"] == "8":
                metadata["gains"]["B8A"] = float(elem.text)
            if elem.attrib["bandId"] == "9":
                metadata["gains"]["B09"] = float(elem.text)
            if elem.attrib["bandId"] == "10":
                metadata["gains"]["B10"] = float(elem.text)
            if elem.attrib["bandId"] == "11":
                metadata["gains"]["B11"] = float(elem.text)
            if elem.attrib["bandId"] == "12":
                metadata["gains"]["B12"] = float(elem.text)

    # Parse the xml tree and add metadata
    root = ET.parse(meta_solar).getroot()
    for elem in root.iter():
        if elem.tag == "Mean_Sun_Angle":
            metadata["ZENITH_ANGLE"] = float(elem.find("ZENITH_ANGLE").text)
            metadata["SUN_ELEVATION"] = 90 - metadata["ZENITH_ANGLE"]
            metadata["AZIMUTH_ANGLE"] = float(elem.find("AZIMUTH_ANGLE").text)

    # Did we get all the metadata?
    for name in metadata:
        assert (
            metadata[name] != None
        ), f"Input metatadata file invalid. {metadata[name]}"

    metadata["INVALID"] = (
        metadata["NODATA_PIXEL_PERCENTAGE"]
        + metadata["SATURATED_DEFECTIVE_PIXEL_PERCENTAGE"]
        + metadata["CLOUD_SHADOW_PERCENTAGE"]
        + metadata["MEDIUM_PROBA_CLOUDS_PERCENTAGE"]
        + metadata["HIGH_PROBA_CLOUDS_PERCENTAGE"]
        + metadata["THIN_CIRRUS_PERCENTAGE"]
        + metadata["SNOW_ICE_PERCENTAGE"]
        + metadata["DARK_FEATURES_PERCENTAGE"]
    )

    metadata["timestamp"] = float(metadata["DATATAKE_SENSING_START"].timestamp())

    metadata["paths"] = get_band_paths(safe_folder)

    return metadata


def get_tilename_from_safe(safe_file):
    return safe_file.split("_")[-2][1:]


def get_adjacent_tiles(tile_name):
    tiles_path = "../../geometry/sentinel2_tiles_world.shp"
    geom = filter_vector(tiles_path, filter_where=("Name", tile_name))
    intersected = intersect_vector(tiles_path, geom)

    attributes = vector_get_attribute_table(intersected)

    return attributes["Name"].values.tolist()


def get_date_from_safe(safe_file):
    datestr = os.path.basename(safe_file).split("_")[2]
    return datetime.strptime(datestr, "%Y%m%dt%H%M%S")


def get_tile_files_from_safe(safe_folder, tile_name):
    files = glob(safe_folder + "*.SAFE")

    tile_files = []
    for file in files:
        try:
            tile = get_tilename_from_safe(file)

            if tile == tile_name:
                tile_files.append(file)

        except:
            pass

    return tile_files


def get_all_tiles_in_folder(folder):
    files = glob(folder + "*.SAFE")

    tiles = []
    for file in files:
        try:
            tile = get_tilename_from_safe(file)
            tiles.append(tile)
        except:
            pass

    return list(set(tiles))


def get_tile_files_from_safe_zip(safe_folder, tile_name):
    files = glob(safe_folder + "*.zip")

    tile_files = []
    for file in files:
        try:
            tile = get_tilename_from_safe(file)

            if tile == tile_name:
                tile_files.append(file)

        except:
            pass

    return tile_files


def unzip_files_to_folder(files, dst_folder):
    unzipped = []
    for f in files:
        try:
            with ZipFile(f) as z:
                unzipped.append(z.extractall(dst_folder))
        except:
            print(f"Invalid file: {f}")
    return unzipped


def get_tiles_from_geom(geom, return_geom=False):
    tiles_path = "../../geometry/sentinel2_tiles_world.shp"

    intersected = clip_vector(tiles_path, reproject_vector(geom, tiles_path))

    attributes = vector_get_attribute_table(intersected)

    return attributes["Name"].values.tolist()


def get_tile_geom_from_name(tile_name):
    tiles_path = "../../geometry/sentinel2_tiles_world.shp"
    geom = filter_vector(tiles_path, filter_where=("Name", tile_name))

    return geom
