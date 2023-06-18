"""
This module provides functions to ease the preprocessing of Sentinel 1 data
and finding the GPT tools.

TODO:
    - Improve documentation
"""

import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Union, List, Tuple, Dict, Any
import numpy as np


# TODO: wkt to ogr geometry
def _s1_kml_to_bbox(path_to_kml: str) -> str:
    """ Internal. """
    root = ET.parse(path_to_kml).getroot()
    for elem in root.iter():
        if elem.tag == "coordinates":
            coords = elem.text
            break

    coords = coords.split(",")
    coords[0] = coords[-1] + " " + coords[0]
    del coords[-1]
    coords.append(coords[0])

    min_x = 180
    max_x = -180
    min_y = 90
    max_y = -90

    for coord in coords:
        intermediate = coord.split(" ")
        intermediate.reverse()

        intermediate[0] = float(intermediate[0])
        intermediate[1] = float(intermediate[1])

        if intermediate[0] < min_x:
            min_x = intermediate[0]
        elif intermediate[0] > max_x:
            max_x = intermediate[0]

        if intermediate[1] < min_y:
            min_y = intermediate[1]
        elif intermediate[1] > max_y:
            max_y = intermediate[1]

    footprint = f"POLYGON (({min_x} {min_y}, {min_x} {max_y}, {max_x} {max_y}, {max_x} {min_y}, {min_x} {min_y}))"

    return footprint


# TODO: get_metadata_from_zip
def s1_get_metadata(
    image_paths: List[str],
) -> List[Dict[str, Any]]:
    """
    Get metadata from Sentinel 1 images.

    Parameters
    ----------
    image_paths : List[str]
        List of paths to Sentinel 1 images.

    Returns
    -------
    List[Dict[str, Any]]
        List of metadata dictionaries.
    """
    images_obj = []

    for img in image_paths:
        kml = f"{img}/preview/map-overlay.kml"

        timestr = str(img.rsplit(".")[0].split("/")[-1].split("_")[5])
        timestamp = datetime.strptime(timestr, "%Y%m%dT%H%M%S").timestamp()

        meta = {
            "path": img,
            "timestamp": timestamp,
            "footprint_wkt": _s1_kml_to_bbox(kml),
        }

        images_obj.append(meta)

    return images_obj

def s1_to_db(
    arr: np.ndarray,
) -> np.ndarray:
    """
    Convert intensity to dB
    10 * log10(arr)
    """
    epsilon = np.finfo(arr.dtype).eps
    return 10.0 * np.log10(np.where(arr == 0.0, epsilon, arr))


def s1_to_intensity(
    arr: np.ndarray,
) -> np.ndarray:
    """
    Convert dB to intensity
    10^(arr/10)
    """
    return np.power(10, arr / 10.0)
