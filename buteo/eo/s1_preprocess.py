"""
This module preprocesses Sentinel-1 data. Using SNAP GPT.

TODO:
    - Enable support for processing only VV or VH.
    - Improve documentation
"""

from sys import platform
import os
import shutil
from glob import glob
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from buteo.raster import (
    raster_to_array,
    array_to_raster,
    raster_to_metadata,
)
from buteo.vector import vector_to_metadata
from buteo.utils.gdal_utils import is_raster, is_vector


def find_gpt(test_gpt_path):
    gpt = os.path.realpath(os.path.abspath(os.path.expanduser(test_gpt_path)))
    if not os.path.exists(gpt):
        possible_locations = [
            "~/esa_snap/bin/gpt",
            "~/snap/bin/gpt",
            "/opt/esa_snap/bin/gpt",
            "/opt/snap/bin/gpt",
            "C:/Program Files/snap/bin/gpt.exe",
            '"C:/Program Files/snap/bin/gpt.exe"',
        ]

        for loc in possible_locations:
            gpt = os.path.realpath(os.path.abspath(os.path.expanduser(loc)))
            if os.path.exists(gpt):
                return gpt

        assert os.path.exists(gpt), "Graph processing tool not found."

    else:
        return gpt


def backscatter_step1(
    zip_file,
    out_path,
    gpt_path="~/snap/bin/gpt",
    extent=None,
    tmp_folder=None,
):
    graph = "backscatter_step1.xml"

    # Get absolute location of graph processing tool
    gpt = find_gpt(gpt_path)

    out_path_ext = out_path + ".dim"
    if os.path.exists(out_path_ext):
        print(f"{out_path_ext} already processed")
        return out_path_ext

    xmlfile = os.path.join(os.path.dirname(__file__), f"./graphs/{graph}")

    snap_graph_step1 = open(xmlfile, "r")
    snap_graph_step1_str = snap_graph_step1.read()
    snap_graph_step1.close()

    if extent is not None:
        if is_vector(extent):
            metadata = vector_to_metadata(extent, create_geometry=True)
        elif is_raster(extent):
            metadata = raster_to_metadata(extent, create_geometry=True)
        elif isinstance(extent, str):
            metadata = raster_to_metadata(extent, create_geometry=True)
        else:
            raise ValueError("Extent must be a vector, raster or a path to a raster.")

        interest_area = metadata["extent_wkt_latlng"]

    else:
        interest_area = "POLYGON ((-180.0 -90.0, 180.0 -90.0, 180.0 90.0, -180.0 90.0, -180.0 -90.0))"

    snap_graph_step1_str = snap_graph_step1_str.replace("${extent}", interest_area)
    snap_graph_step1_str = snap_graph_step1_str.replace("${inputfile}", zip_file)
    snap_graph_step1_str = snap_graph_step1_str.replace("${outputfile}", out_path)

    xmlfile = tmp_folder + os.path.basename(out_path) + "_graph.xml"

    f = open(xmlfile, "w")
    f.write(snap_graph_step1_str)
    f.close()

    command = [
        gpt,
        os.path.abspath(xmlfile),
        f"-q {cpu_count()}",
    ]

    if platform == "linux" or platform == "linux2":
        cmd = " ".join(command)
    else:
        cmd = f'cmd /c {" ".join(command)}'

    os.system(cmd)

    return out_path_ext

def backscatter_step2(
    dim_file,
    out_path,
    speckle_filter=False,
    epsg=None,
    gpt_path="~/snap/bin/gpt",
    tmp_folder=None,
):
    graph = "backscatter_step2.xml"
    if speckle_filter:
        graph = "backscatter_step2_speckle.xml"

    # Get absolute location of graph processing tool
    gpt = find_gpt(gpt_path)

    out_path_ext = out_path + ".dim"
    if os.path.exists(out_path_ext):
        print(f"{out_path_ext} already processed")
        return out_path_ext

    xmlfile = os.path.join(os.path.dirname(__file__), f"./graphs/{graph}")

    snap_graph_step2 = open(xmlfile, "r")
    snap_graph_step2_str = snap_graph_step2.read()
    snap_graph_step2.close()

    if epsg is None:
        use_epsg = "AUTO:42001"
    else:
        use_epsg = f"EPSG:{epsg}"

    snap_graph_step2_str = snap_graph_step2_str.replace("${epsg}", use_epsg)
    snap_graph_step2_str = snap_graph_step2_str.replace("${inputfile}", dim_file)
    snap_graph_step2_str = snap_graph_step2_str.replace("${outputfile}", out_path)

    xmlfile = tmp_folder + os.path.basename(out_path) + "_graph.xml"

    f = open(xmlfile, "w")
    f.write(snap_graph_step2_str)
    f.close()

    command = [
        gpt,
        os.path.abspath(xmlfile),
        f"-q {cpu_count()}",
    ]

    if platform == "linux" or platform == "linux2":
        cmd = " ".join(command)
    else:
        cmd = f'cmd /c {" ".join(command)}'

    snap_graph_step2 = open(xmlfile, "r")
    snap_graph_step2_str = snap_graph_step2.read()
    snap_graph_step2.close()

    xmlfile = tmp_folder + os.path.basename(out_path) + "_graph.xml"

    f = open(xmlfile, "w")
    f.write(snap_graph_step2_str)
    f.close()

    os.system(cmd)

    return out_path_ext


def convert_to_tiff(
    dim_file,
    out_folder,
    decibel=False,
    use_nodata=True,
    nodata_value=-9999.0,
):
    data_folder = os.path.splitext(dim_file)[0] + ".data/"
    name = os.path.splitext(os.path.basename(dim_file))[0].replace("_step_2", "")

    vh_path = data_folder + "Gamma0_VH.img"
    vv_path = data_folder + "Gamma0_VV.img"

    out_paths = [
        out_folder + name + "_Gamma0_VH.tif",
        out_folder + name + "_Gamma0_VV.tif",
    ]

    if os.path.exists(out_paths[0]) and os.path.exists(out_paths[1]):
        print(f"{name} already processed")
        return out_paths

    vh = raster_to_array(vh_path)
    vv = raster_to_array(vv_path)

    if use_nodata:
        vh = np.ma.masked_equal(vh, 0.0, copy=False)
        vv = np.ma.masked_equal(vv, 0.0, copy=False)

        vh = np.nan_to_num(vh)
        vv = np.nan_to_num(vv)

        vh = np.ma.masked_equal(vh.filled(nodata_value), nodata_value)
        vv = np.ma.masked_equal(vv.filled(nodata_value), nodata_value)

    if decibel:
        with np.errstate(divide="ignore", invalid="ignore"):
            if use_nodata:

                vh = np.ma.multiply(np.ma.log10(np.ma.abs(vh)), 10)
                vv = np.ma.multiply(np.ma.log10(np.ma.abs(vv)), 10)
            else:
                vh = np.multiply(np.log10(np.abs(vh)), 10)
                vv = np.multiply(np.log10(np.abs(vv)), 10)

    array_to_raster(vh, vh_path, out_paths[0])
    array_to_raster(vv, vv_path, out_paths[1])

    return out_paths


def clear_tmp_folder(tmp_folder):
    try:
        tmp_files = glob(tmp_folder + "*.dim")
        for f in tmp_files:
            os.remove(f)
        tmp_files = glob(tmp_folder + "*.data")
        for f in tmp_files:
            shutil.rmtree(f)
    except Exception:
        print("Error while cleaning tmp files.")


def backscatter(
    zip_file,
    out_path,
    tmp_folder,
    extent=None,
    epsg=None,
    use_nodata=True,
    nodata_value=-9999.0,
    speckle_filter=False,
    decibel=False,
    clean_tmp=False,
    gpt_path="~/snap/bin/gpt",
):
    base = os.path.splitext(os.path.splitext(os.path.basename(zip_file))[0])[0]
    name1 = base + "_step_1"

    vh = out_path + base + "_Gamma0_VH.tif"
    vv = out_path + base + "_Gamma0_VV.tif"

    if os.path.exists(vh) and os.path.exists(vv):
        if clean_tmp:
            clear_tmp_folder(tmp_folder)
        print(f"{zip_file} already processed")
        return [vh, vv]

    def step1_helper(args=[]):
        return backscatter_step1(
            args[0], args[1], gpt_path=args[2], extent=args[3], tmp_folder=args[4]
        )

    try:
        p = ThreadPoolExecutor(1)

        step1 = p.submit(
            step1_helper,
            args=[zip_file, tmp_folder + name1, gpt_path, extent, tmp_folder],
        )
        step1 = step1.result(timeout=60 * 10)

    except Exception as e:
        if clean_tmp:
            clear_tmp_folder(tmp_folder)

        raise Exception(f"{zip_file} failed to complete step 1., {e}")

    name2 = base + "_step_2"

    def step2_helper(args=[]):
        return backscatter_step2(
            args[0],
            args[1],
            speckle_filter=args[2],
            epsg=epsg,
            gpt_path=args[3],
            tmp_folder=args[4],
        )

    try:
        p = ThreadPoolExecutor(1)

        step2 = p.submit(
            step2_helper,
            args=(step1, tmp_folder + name2, speckle_filter, gpt_path, tmp_folder),
        )
        step2 = step2.result(timeout=60 * 60)

    except Exception as e:
        if clean_tmp:
            clear_tmp_folder(tmp_folder)

        raise Exception(f"{zip_file} failed to complete step 2., {e}")

    print("Generating .tif files from step2.")
    converted = convert_to_tiff(
        step2, out_path, decibel, use_nodata=use_nodata, nodata_value=nodata_value
    )

    if clean_tmp:
        clear_tmp_folder(tmp_folder)

    return converted
