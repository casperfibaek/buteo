import sys

sys.path.append("..")
sys.path.append("../../")

from glob import glob
from sys import platform
import xml.etree.ElementTree as ET
from datetime import datetime
from multiprocessing import cpu_count
from buteo.raster.io import (
    raster_to_array,
    array_to_raster,
)
from buteo.vector.io import internal_vector_to_metadata
import os
import numpy as np
import shutil
from concurrent.futures import ThreadPoolExecutor


def find_gpt(test_gpt_path):
    gpt = os.path.realpath(os.path.abspath(os.path.expanduser(test_gpt_path)))
    if not os.path.exists(gpt):
        gpt = os.path.realpath(
            os.path.abspath(os.path.expanduser("~/esa_snap/bin/gpt"))
        )
        if not os.path.exists(gpt):
            gpt = os.path.realpath(
                os.path.abspath(os.path.expanduser("~/snap/bin/gpt"))
            )
            if not os.path.exists(gpt):
                gpt = os.path.realpath(
                    os.path.abspath(os.path.expanduser("/opt/esa_snap/bin/gpt"))
                )
                if not os.path.exists(gpt):
                    gpt = os.path.realpath(
                        os.path.abspath(os.path.expanduser("/opt/snap/bin/gpt"))
                    )
                    if not os.path.exists(gpt):
                        gpt = os.path.realpath(
                            os.path.abspath(
                                os.path.expanduser("C:/Program Files/snap/bin/gpt.exe")
                            )
                        )

                        if os.path.exists(gpt):
                            gpt = '"C:/Program Files/snap/bin/gpt.exe"'
                        else:
                            if not os.path.exists(gpt):
                                assert os.path.exists(gpt), "Graph processing tool not found."

    return gpt


def s1_kml_to_bbox(path_to_kml):
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

    for i in range(len(coords)):
        intermediate = coords[i].split(" ")
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


def get_metadata(image_paths):
    images_obj = []

    for img in image_paths:
        kml = f"{img}/preview/map-overlay.kml"

        timestr = str(img.rsplit(".")[0].split("/")[-1].split("_")[5])
        timestamp = datetime.strptime(timestr, "%Y%m%dT%H%M%S").timestamp()

        meta = {
            "path": img,
            "timestamp": timestamp,
            "footprint_wkt": s1_kml_to_bbox(kml),
        }

        images_obj.append(meta)

    return images_obj


def backscatter_step1(
    zip_file, out_path, gpt_path="~/snap/bin/gpt",
):
    graph = "backscatter_step1.xml"

    # Get absolute location of graph processing tool
    gpt = find_gpt(gpt_path)

    out_path_ext = out_path + ".dim"
    if os.path.exists(out_path_ext):
        print(f"{out_path_ext} already processed")
        return out_path_ext

    xmlfile = os.path.join(os.path.dirname(__file__), f"./graphs/{graph}")

    command = [
        gpt,
        os.path.abspath(xmlfile),
        f"-Pinputfile={zip_file}",
        f"-Poutputfile={out_path}",
        f"-q {cpu_count()}",
        # "-c 16978M",
        # "-J-Xmx16G -J-Xms1G",
    ]

    if platform == "linux" or platform == "linux2":
        cmd = " ".join(command)
    else:
        cmd = f'cmd /c {" ".join(command)}'

    os.system(cmd)

    return out_path_ext


def backscatter_step2(
    dim_file, out_path, speckle_filter=False, extent=None, gpt_path="~/snap/bin/gpt",
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

    command = [
        gpt,
        os.path.abspath(xmlfile),
        f"-Pinputfile={dim_file}",
        f"-Poutputfile={out_path}",
        f"-q {cpu_count()}",
        # "-c 16978M",
        # "-J-Xmx16G -J-Xms1G",
    ]

    if extent is not None:
        metadata = internal_vector_to_metadata(extent)
        interest_area = metadata["extent_wkt_latlng"]

        command.append(f"-Pextent='{interest_area}'")
    else:
        command.append(
            f"-Pextent='POLYGON ((-180.0 -90.0, 180.0 -90.0, 180.0 90.0, -180.0 90.0, -180.0 -90.0))'"
        )

    if platform == "linux" or platform == "linux2":
        cmd = " ".join(command)
    else:
        cmd = f'cmd /c {" ".join(command)}'
    os.system(cmd)

    return out_path_ext


def convert_to_tiff(
    dim_file, out_folder, decibel=False, use_nodata=True, nodata_value=0.0
):
    data_folder = dim_file.split(".")[0] + ".data/"
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
        vh = np.ma.masked_equal(vh, nodata_value, copy=False)
        vv = np.ma.masked_equal(vv, nodata_value, copy=False)

    if decibel:
        with np.errstate(divide="ignore", invalid="ignore"):
            if use_nodata:
                vh = np.ma.multiply(np.ma.log10(vh), 10)
                vv = np.ma.multiply(np.ma.log10(vv), 10)
            else:
                vh = np.multiply(np.log10(vh), 10)
                vv = np.multiply(np.log10(vv), 10)

    out_paths = [
        out_folder + name + "_Gamma0_VH.tif",
        out_folder + name + "_Gamma0_VV.tif",
    ]

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
    except:
        print("Error while cleaning tmp files.")


def backscatter(
    zip_file,
    out_path,
    tmp_folder,
    extent=None,
    speckle_filter=False,
    decibel=False,
    gpt_path="~/snap/bin/gpt",
):
    name = os.path.splitext(os.path.basename(zip_file))[0] + "_step_1"
    
    vh = out_path + os.path.splitext(os.path.basename(zip_file))[0] + "_Gamma0_VH.tif"
    vv = out_path + os.path.splitext(os.path.basename(zip_file))[0] + "_Gamma0_VV.tif"

    if os.path.exists(vh) and os.path.exists(vv):
        clear_tmp_folder(tmp_folder)
        print(f"{zip_file} already processed")
        return [vh, vv]

    def step1_helper(args=[]):
        return backscatter_step1(args[0], args[1], gpt_path=args[2])

    try:
        p = ThreadPoolExecutor(1)

        step1 = p.submit(step1_helper, args=[zip_file, tmp_folder + name, gpt_path])
        step1 = step1.result(timeout=60*10)

    except:
        clear_tmp_folder(tmp_folder)
        raise Exception(f"{zip_file} filed to complete step 1.")

    name = os.path.splitext(os.path.basename(zip_file))[0] + "_step_2"

    def step2_helper(args=[]):
        return backscatter_step2(args[0], args[1], speckle_filter=args[2], extent=args[3], gpt_path=args[4])

    try:
        p = ThreadPoolExecutor(1)

        step2 = p.submit(step2_helper, args=(step1, tmp_folder + name, speckle_filter, extent, gpt_path))
        step2 = step2.result(timeout=60*60)
        
    except:
        clear_tmp_folder(tmp_folder)
        raise Exception(f"{zip_file} filed to complete step 2.")

    converted = convert_to_tiff(step2, out_path, decibel)

    clear_tmp_folder(tmp_folder)

    return converted


if __name__ == "__main__":
    folder = "C:/Users/caspe/Desktop/paper_transfer_learning/data/sentinel1/"
    # folder = "/media/cfi/lts/ghana/sentinel1/"
    # folder = "/home/cfi/Desktop/sentinel1/"
    tmp = folder + "tmp/"
    dst = folder + "mosaic_2021/"
    raw = folder + "raw_2021/"

    images = glob(raw + "*.zip")
    interest_area = folder + "denmark_polygon_1280m_buffer.gpkg"
    gpt_path = "/opt/esa_snap/bin/gpt"

    error_images = []
    for idx, image in enumerate(images):
        try:
            paths = backscatter(image, dst, tmp, extent=interest_area, gpt_path=gpt_path)
        except:
            print(f"Error with image: {image}")
            error_images.append(image)

        print(f"Completed {idx+1}/{len(images)}")
    
    import pdb; pdb.set_trace()
