"""
This module contains utility functions to work with sentinel 2 data.
"""

# Standard library
import sys; sys.path.append("../../")
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from zipfile import ZipFile
from glob import glob

# Internal
from buteo.raster import raster_to_array


def s2_l2a_get_bands(
    zip_or_safe_folder: str,
    zipfile: bool = True,
) -> dict:
    """ Get the bands from a sentinel 2 L2A product.
    
    Parameters
    ----------
    zip_or_safe_folder : str
        Path to the zip file or SAFE folder.

    zipfile : bool, optional
        If True, the zip file is used. If False, the SAFE folder is used. Default: True.

    Returns
    -------
    dict
        Dictionary with the paths to the bands.
    """
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

    if zipfile:
        assert os.path.isfile(zip_or_safe_folder), f"Could not find file: {zip_or_safe_folder}"
    else:
        assert os.path.isdir(zip_or_safe_folder), f"Could not find folder: {zip_or_safe_folder}"

    prepend = ""
    if zipfile:
        archive = ZipFile(zip_or_safe_folder, "r")
        prepend = f"/vsizip/{zip_or_safe_folder}/"

        bands["QI"]["CLDPRB_20m"] = prepend + [name for name in archive.namelist() if name.endswith("MSK_CLDPRB_20m.jp2")][0]
        bands["QI"]["CLDPRB_60m"] = prepend + [name for name in archive.namelist() if name.endswith("MSK_CLDPRB_60m.jp2")][0]
    else:
        bands["QI"]["CLDPRB_20m"] = prepend + glob(f"{zip_or_safe_folder}/GRANULE/*/QI_DATA/MSK_CLDPRB_20m.jp2")[0]
        bands["QI"]["CLDPRB_60m"] = prepend + glob(f"{zip_or_safe_folder}/GRANULE/*/QI_DATA/MSK_CLDPRB_60m.jp2")[0]

    if zipfile:
        bands_10m = [name for name in archive.namelist() if name.endswith("_10m.jp2")]
    else:
        bands_10m = glob(f"{zip_or_safe_folder}/GRANULE/*/IMG_DATA/R10m/*_???_*.jp2")

    for band in bands_10m:
        basename = os.path.basename(band)
        band_name = basename.split("_")[2]
        if band_name == "B02":
            bands["10m"]["B02"] = prepend + band
        if band_name == "B03":
            bands["10m"]["B03"] = prepend + band
        if band_name == "B04":
            bands["10m"]["B04"] = prepend + band
        if band_name == "B08":
            bands["10m"]["B08"] = prepend + band
        if band_name == "AOT":
            bands["10m"]["AOT"] = prepend + band

    if zipfile:
        bands_20m = [name for name in archive.namelist() if name.endswith("_20m.jp2")]
    else:
        bands_20m = glob(f"{zip_or_safe_folder}/GRANULE/*/IMG_DATA/R20m/*.jp2")
    for band in bands_20m:
        basename = os.path.basename(band)
        band_name = basename.split("_")[2]
        if band_name == "B02":
            bands["20m"]["B02"] = prepend + band
        if band_name == "B03":
            bands["20m"]["B03"] = prepend + band
        if band_name == "B04":
            bands["20m"]["B04"] = prepend + band
        if band_name == "B05":
            bands["20m"]["B05"] = prepend + band
        if band_name == "B06":
            bands["20m"]["B06"] = prepend + band
        if band_name == "B07":
            bands["20m"]["B07"] = prepend + band
        if band_name == "B8A":
            bands["20m"]["B8A"] = prepend + band
        if band_name == "B09":
            bands["20m"]["B09"] = prepend + band
        if band_name == "B11":
            bands["20m"]["B11"] = prepend + band
        if band_name == "B12":
            bands["20m"]["B12"] = prepend + band
        if band_name == "SCL":
            bands["20m"]["SCL"] = prepend + band
        if band_name == "AOT":
            bands["20m"]["AOT"] = prepend + band

    if zipfile:
        bands_60m = [name for name in archive.namelist() if name.endswith("_60m.jp2")]
    else:
        bands_60m = glob(f"{zip_or_safe_folder}/GRANULE/*/IMG_DATA/R60m/*_???_*.jp2")
    for band in bands_60m:
        basename = os.path.basename(band)
        band_name = basename.split("_")[2]
        if band_name == "B01":
            bands["60m"]["B01"] = prepend + band
        if band_name == "B02":
            bands["60m"]["B02"] = prepend + band
        if band_name == "B03":
            bands["60m"]["B03"] = prepend + band
        if band_name == "B04":
            bands["60m"]["B04"] = prepend + band
        if band_name == "B05":
            bands["60m"]["B05"] = prepend + band
        if band_name == "B06":
            bands["60m"]["B06"] = prepend + band
        if band_name == "B07":
            bands["60m"]["B07"] = prepend + band
        if band_name == "B8A":
            bands["60m"]["B8A"] = prepend + band
        if band_name == "B09":
            bands["60m"]["B09"] = prepend + band
        if band_name == "B11":
            bands["60m"]["B11"] = prepend + band
        if band_name == "B12":
            bands["60m"]["B12"] = prepend + band
        if band_name == "SCL":
            bands["60m"]["SCL"] = prepend + band
        if band_name == "AOT":
            bands["60m"]["AOT"] = prepend + band

    for key_i, value_i in bands.items():
        for key_j, value_j in value_i.items():
            current_band = value_j
            assert (
                current_band is not None
            ), f"{key_i} - {key_j} was not found. Verify the folders. Was the decompression interrupted?"

    return bands


def s2_l2a_get_metadata(
    zip_or_safe_folder: str,
    zipfile: bool = True,
) -> dict:
    """
    Get metadata from the SAFE folder or the zip file.
    
    Parameters
    ----------
    zip_or_safe_folder : str
        Path to the zip file or the SAFE folder.

    zipfile : bool, optional
        If True, the zip file is used. If False, the SAFE folder is used. Default: True

    Returns
    -------
    dict
        Dictionary with the metadata.
    """
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
        "folder": zip_or_safe_folder,
        "gains": {},
    }

    if zipfile:
        archive = ZipFile(zip_or_safe_folder, "r")

        meta_xml_path = [name for name in archive.namelist() if name.endswith("MTD_MSIL2A.xml")][0]
        meta_solar_path = [name for name in archive.namelist() if name.endswith("MTD_TL.xml")][0]

        meta_xml = archive.open(meta_xml_path)
        meta_solar = archive.open(meta_solar_path)
    else:
        meta_xml = os.path.join(zip_or_safe_folder, "MTD_MSIL2A.xml")
        meta_solar = glob(zip_or_safe_folder + "/GRANULE/*/MTD_TL.xml")[0]

        assert os.path.isfile(meta_xml), f"{zip_or_safe_folder} did not contain a valid metadata file."
        assert os.path.isfile(meta_solar), f"{meta_solar} did not contain a valid metadata file."

    # Parse the xml tree and add metadata
    root = ET.parse(meta_xml).getroot()
    for elem in root.iter():
        if elem.tag in metadata:
            try:
                metadata[elem.tag] = float(elem.text)  # Number?
            except ValueError:
                try:
                    metadata[elem.tag] = datetime.strptime(
                        elem.text, "%Y-%m-%dT%H:%M:%S.%f%z"
                    )  # Date?
                except ValueError:
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
    for name, value in metadata.items():
        assert value is not None, f"Input metatadata file invalid. {value}"

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

    metadata["paths"] = s2_l2a_get_bands(zip_or_safe_folder, zipfile=zipfile)

    def get_bands(bands="10m"):
        if bands == "10m":
            paths = [metadata["paths"]["10m"][band] for band in ["B02", "B03", "B04", "B08"]]
        elif bands == "20m":
            paths = [metadata["paths"]["20m"][band] for band in ["B05", "B06", "B07", "B8A", "B11", "B12"]]
        elif bands == "60m":
            paths = [metadata["paths"]["60m"][band] for band in ["B01", "B09"]]
        elif bands == "scl_20m":
            paths = [metadata["paths"]["20m"]["SCL"]]
        elif bands == "cld_prb_20m":
            paths = [metadata["paths"]["QI"]["CLDPRB_20m"]]

        read_files = raster_to_array(paths)

        return read_files

    metadata["bands_10m"] = lambda : get_bands(bands="10m")
    metadata["bands_20m"] = lambda : get_bands(bands="20m")
    metadata["bands_60m"] = lambda : get_bands(bands="60m")
    metadata["bands_scl_20m"] = lambda : get_bands(bands="scl_20m")
    metadata["bands_cld_prb_20m"] = lambda : get_bands(bands="cld_prb_20m")

    return metadata
