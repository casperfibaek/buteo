"""
This module is used to super-sample S2 data to 10m resolution. Using pansharpening.
Requires the orfeo toolbox.

TODO:
    - Remove reliance on Orfeo toolbox
    - Add support a deep learning option.
"""

import sys; sys.path.append("../../") # Path: buteo/earth_observation/s2_pansharpen.py
import os

from buteo.raster.io import raster_to_array, array_to_raster
from buteo.raster.resample import resample_raster
from buteo.orfeo_toolbox_bindings import pansharpen


def super_sample_s2(
    B04_link,
    B08_link,
    B05_link=None,
    B06_link=None,
    B07_link=None,
    B8A_link=None,
    out_folder="../raster/",
    prefix="",
    suffix="",
):
    assert (
        isinstance(B05_link, str)
        or isinstance(B06_link, str)
        or isinstance(B07_link, str)
        or isinstance(B8A_link, str)
    )

    paths = {
        "B04": B04_link,
        "B05": B05_link,
        "B06": B06_link,
        "B07": B07_link,
        "B08": B08_link,
        "B8A": B8A_link,
    }

    bands = {
        "B04": raster_to_array(B04_link).astype("float32"),
        "B05": raster_to_array(B05_link).astype("float32")
        if B05_link is not None
        else False,
        "B06": raster_to_array(B06_link).astype("float32")
        if B06_link is not None
        else False,
        "B07": raster_to_array(B07_link).astype("float32")
        if B07_link is not None
        else False,
        "B08": raster_to_array(B08_link).astype("float32"),
        "B8A": raster_to_array(B8A_link).astype("float32")
        if B8A_link is not None
        else False,
    }

    bands_to_pansharpen = []
    if bands["B05"] is not False:
        bands_to_pansharpen.append("B05")
    if bands["B06"] is not False:
        bands_to_pansharpen.append("B06")
    if bands["B07"] is not False:
        bands_to_pansharpen.append("B07")
    if bands["B8A"] is not False:
        bands_to_pansharpen.append("B8A")

    for band_x in bands_to_pansharpen:
        if band_x == "B05":
            pseudo_band = "B04"
        else:
            pseudo_band = "B08"

        pseudo_path = os.path.join(out_folder, f"{prefix}{band_x}{suffix}_pseudo.tif")
        array_to_raster(
            bands[pseudo_band],
            reference_raster=paths[pseudo_band],
            out_raster=pseudo_path,
        )

        low_res_10m = raster_to_array(
            resample_raster(paths[band_x], reference_raster=paths[pseudo_band])
        ).astype("float32")
        resampled_path = os.path.join(
            out_folder, f"{prefix}{band_x}{suffix}_resampled.tif"
        )
        array_to_raster(
            low_res_10m, reference_raster=paths[pseudo_band], out_raster=resampled_path
        )

        low_res_10m = None

        pansharpened_path = os.path.join(
            out_folder, f"{prefix}{band_x}{suffix}_float.tif"
        )
        pansharpen(pseudo_path, resampled_path, pansharpened_path)

        os.remove(resampled_path)
        os.remove(pseudo_path)
