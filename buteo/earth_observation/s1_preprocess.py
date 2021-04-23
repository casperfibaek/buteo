import sys

sys.path.append("..")
sys.path.append("../../")

from glob import glob
import xml.etree.ElementTree as ET
from datetime import datetime
from multiprocessing import cpu_count
from buteo.vector.io import internal_vector_to_metadata
import os


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
    zip_file,
    out_path,
    graph="backscatter_step1.xml",
    speckle_filter=False,
    extent=None,
    output_units="decibels",
    gpt="~/snap/bin/gpt",
    verbose=True,
):
    # Get absolute location of graph processing tool
    gpt = os.path.realpath(os.path.abspath(os.path.expanduser(gpt)))
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
                    os.path.abspath(
                        os.path.expanduser("C:/Program Files/snap/bin/gpt.exe")
                    )
                )

                if os.path.exists(gpt):
                    gpt = '"C:/Program Files/snap/bin/gpt.exe"'
                else:
                    if not os.path.exists(gpt):
                        assert os.path.exists(gpt), "Graph processing tool not found."

    if os.path.exists(out_path):
        print("File already processed")
        return 1

    xmlfile = os.path.join(os.path.dirname(__file__), f"./graphs/{graph}")

    command = [
        gpt,
        os.path.abspath(xmlfile),
        f"-Pinputfile={zip_file}",
        f"-Poutputfile={out_path}",
        f"-q {cpu_count()}",
        "-c 31978M",
        "-J-Xmx45G -J-Xms2G",
    ]

    os.system(f'cmd /c {" ".join(command)}')

    return out_path


def backscatter_step2(
    zip_file,
    out_path,
    graph="backscatter_step2.xml",
    interest_area=None,
    speckle_filter=False,
    extent=None,
    output_units="decibels",
    gpt="~/snap/bin/gpt",
    verbose=True,
):
    # Get absolute location of graph processing tool
    gpt = os.path.realpath(os.path.abspath(os.path.expanduser(gpt)))
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
                    os.path.abspath(
                        os.path.expanduser("C:/Program Files/snap/bin/gpt.exe")
                    )
                )

                if os.path.exists(gpt):
                    gpt = '"C:/Program Files/snap/bin/gpt.exe"'
                else:
                    if not os.path.exists(gpt):
                        assert os.path.exists(gpt), "Graph processing tool not found."

    if os.path.exists(out_path):
        print("File already processed")
        return 1

    xmlfile = os.path.join(os.path.dirname(__file__), f"./graphs/{graph}")

    command = [
        gpt,
        os.path.abspath(xmlfile),
        f"-Pinputfile={zip_file}",
        f"-Poutputfile={out_path}",
        f"-q {cpu_count()}",
        "-c 31978M",
        "-J-Xmx45G -J-Xms2G",
    ]

    if interest_area is not None:
        extent = internal_vector_to_metadata(interest_area)["extent_latlng"]
        command.append(f"-Pextent={' '.join(extent)}")

    print(command)

    os.system(f'cmd /c {" ".join(command)}')

    return out_path


if __name__ == "__main__":
    folder = "C:/Users/caspe/Desktop/paper_transfer_learning/data/sentinel1/"
    images = glob(folder + "*.dim")
    interest_area = folder + "denmark_polygon.gpkg"

    for image in images:
        out_name = os.path.splitext(os.path.basename(image))[0] + "bob"
        backscatter_step2(image, folder + out_name, interest_area=interest_area)

        import pdb

        pdb.set_trace()

