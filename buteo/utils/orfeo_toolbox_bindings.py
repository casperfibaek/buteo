"""
Bindings to interact with the orfeo-toolbox in a more pythonic way.

TODO:
    - Documentation
"""

import sys; sys.path.append("../../") # Path: buteo/orfeo_toolbox_bindings.py
import os
import subprocess
from time import time

from buteo.raster.io import raster_to_array
from buteo.utils.core import progress


def execute_cli_function(command, name, quiet=False):
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    try:
        before = time.time()
        for line in iter(process.stdout.readline, ""):
            if "FATAL" in line:
                raise RuntimeError(line)
            elif "CRITICAL" in line:
                raise RuntimeError(line)
            elif "WARNING" in line:
                continue
            elif quiet is False:
                if "INFO" in line:
                    continue
            try:
                strip = line.strip()
                if len(strip) != 0:
                    part = strip.rsplit(":", 1)[1]
                    percent = int(part.split("%")[0])
                    progress(percent, 100, name)
            except Exception:
                if len(line.strip()) != 0:
                    raise RuntimeError(line) from None

    except Exception:
        print("Critical failure while performing Orfeo-Toolbox action.")

    print(f"{name} completed in {round(time.time() - before, 2)}s.")


def otb_pansharpen(in_pan, in_xs, out_raster, options=None, out_datatype=None):
    """Pansharpen an image using the attributes
    of another image. Beware that the two images
    should be of the same size and position."""

    cli = "otbcli_Pansharpening"

    methods = ["rcs", "lmvm", "bayes"]

    if options is None:
        options = {
            "method": "lmvm",
            "method.lmvm.radiusx": 3,
            "method.lmvm.radiusy": 3,
        }

    if options["method"] not in methods:
        raise AttributeError("Selected method is not available.")

    if options["method"] == "lmvm":
        if "method.lmvm.radiusx" not in options:
            options["method.lmvm.radiusx"] = 3
        if "method.lmvm.radiusy" not in options:
            options["method.lmvm.radiusy"] = 3
    if options["method"] == "bayes":
        if "method.bayes.lamda" not in options:
            options["method.bayes.lamda"] = 0.9999
        if "method.bayes.s" not in options:
            options["method.bayes.s"] = 1

    if out_datatype is None:
        out_datatype = ""

    cli_args = [
        cli,
        "-inp",
        os.path.abspath(in_pan),
        "-inxs",
        os.path.abspath(in_xs),
        "-out",
        f'"{os.path.abspath(out_raster)}?&gdal:co:COMPRESS=DEFLATE&gdal:co:NUM_THREADS=ALL_CPUS&gdal:co:BIGTIFF=YES"',
        out_datatype,
    ]

    for key, value in options.items():
        cli_args.append("-" + str(key))
        cli_args.append(str(value))

    cli_string = " ".join(cli_args)

    execute_cli_function(cli_string, name="Pansharpening")

    return os.path.abspath(out_raster)


def otb_local_stats(in_raster, out_raster, options=None, band=None):
    """Computes local statistical moments on every pixel
    in the selected channel of the input image"""

    cli = "otbcli_LocalStatisticExtraction"

    if options is None:
        options = {
            "channel": 1,
            "radius": 2,
        }

    if band is not None:
        band = f"&bands={band}"
    else:
        band = ""

    cli_args = [
        cli,
        "-in",
        os.path.abspath(in_raster),
        "-out",
        f'"{os.path.abspath(out_raster)}?{band}&gdal:co:COMPRESS=DEFLATE&gdal:co:PREDICTOR=3&gdal:co:NUM_THREADS=ALL_CPUS&gdal:co:BIGTIFF=YES" float',
    ]

    for key, value in options.items():
        cli_args.append("-" + str(key))
        cli_args.append(str(value))

    cli_string = " ".join(cli_args)

    execute_cli_function(cli_string, name="Local statistics")

    return os.path.abspath(out_raster)


def otb_haralick(in_raster, out_raster, options=None, out_datatype="float", band=None):
    """Performs haralick texture extraction"""

    cli = "otbcli_HaralickTextureExtraction"

    stats_raster = raster_to_array(in_raster)
    stats = {"min": stats_raster.min(), "max": stats_raster.max()}
    stats_raster = None

    if options is None:
        options = {
            "texture": "simple",
            "channel": 1,
            "parameters.nbbin": 64,
            "parameters.xrad": 3,
            "parameters.yrad": 3,
            "parameters.min": stats["min"],
            "parameters.max": stats["max"],
        }

    if out_datatype is None:
        out_datatype = "float"

    if band is not None:
        band = f"&bands={band}"
    else:
        band = ""

    cli_args = [
        cli,
        "-in",
        os.path.abspath(in_raster),
        "-out",
        f'"{os.path.abspath(out_raster)}?{band}&gdal:co:COMPRESS=DEFLATE&gdal:co:NUM_THREADS=ALL_CPUS&gdal:co:BIGTIFF=YES"',
        out_datatype,
    ]

    for key, value in options.items():
        cli_args.append("-" + str(key))
        cli_args.append(str(value))

    cli_string = " ".join(cli_args)

    execute_cli_function(cli_string, name="Texture extraction")

    return os.path.abspath(out_raster)


def otb_dimension_reduction(in_raster, out_raster, options=None, out_datatype=None):
    """Performs dimensionality reduction on input image.
    PCA,NA-PCA,MAF,ICA methods are available.
    It is also possible to compute the inverse transform
    to reconstruct the image. It is also possible to
    optionally export the transformation matrix
    to a text file."""

    cli = "otbcli_DimensionalityReduction"

    if options is None:
        options = {
            "method": "pca",
            "rescale.outmin": 0,
            "rescale.outmax": 1,
            "nbcomp": 1,
            "normalize": "YES",
        }

    if out_datatype is None:
        out_datatype = ""

    cli_args = [
        cli,
        "-in",
        os.path.abspath(in_raster),
        "-out",
        f'"{os.path.abspath(out_raster)}?&gdal:co:COMPRESS=DEFLATE&gdal:co:NUM_THREADS=ALL_CPUS&gdal:co:BIGTIFF=YES"',
        out_datatype,
    ]

    for key, value in options.items():
        cli_args.append("-" + str(key))
        cli_args.append(str(value))

    cli_string = " ".join(cli_args)

    execute_cli_function(cli_string, name="Dimension reduction")

    return os.path.abspath(out_raster)


def otb_concatenate_images(in_rasters, out_raster, ram=None, out_datatype=None):
    """This application performs images channels concatenation.
    It reads the input image list (single or multi-channel) and
    generates a single multi-channel image. The channel order
    is the same as the list."""

    cli = "otbcli_ConcatenateImages"

    paths = []
    for raster in in_rasters:
        paths.append(os.path.abspath(raster))
    paths = " ".join(paths)

    if out_datatype is None:
        out_datatype = ""

    cli_string = " ".join(
        [
            cli,
            "-il",
            os.path.abspath(paths),
            "-out",
            f'"{os.path.abspath(out_raster)}?&gdal:co:COMPRESS=DEFLATE&gdal:co:NUM_THREADS=ALL_CPUS&gdal:co:BIGTIFF=YES"',
            out_datatype,
            "-ram",
            str(ram),
        ]
    )

    execute_cli_function(cli_string, name="concatenate images")

    return os.path.abspath(out_raster)


def otb_split_images(in_raster, out_rasters, ram=None, out_datatype=None):
    """This application splits a N-bands image into N mono-band images.
    The output images filename will be generated from the output parameter.
    Thus, if the input image has 2 channels, and the user has set as
    output parameter, outimage.tif, the generated images will be
    outimage_0.tif and outimage_1.tif."""

    cli = "otbcli_SplitImage"

    if out_datatype is None:
        out_datatype = ""

    cli_string = " ".join(
        [
            cli,
            "-in",
            os.path.abspath(in_raster),
            "-out",
            f'"{os.path.abspath(out_rasters)}?&gdal:co:COMPRESS=DEFLATE&gdal:co:NUM_THREADS=ALL_CPUS&gdal:co:BIGTIFF=YES"',
            out_datatype,
            "-ram",
            str(ram),
        ]
    )

    execute_cli_function(cli_string, name="splitting images")

    return out_rasters


def otb_rescale(in_raster, out_raster, options=None, out_datatype="float"):
    """This application scales the given image pixel intensity between two given values.
    By default min (resp. max) value is set to 0 (resp. 1).
    Input minimum and maximum values is automatically computed for all image bands."""

    cli = "otbcli_Rescale"

    if options is None:
        options = {
            "outmin": 0,
            "outmax": 1,
        }

    if out_datatype == "float":
        predictor = "gdal:co:PREDICTOR=3&"
    elif out_datatype == "uint16":
        predictor = "gdal:co:PREDICTOR=2&"
    else:
        predictor = ""

    cli_args = [
        cli,
        "-in",
        os.path.abspath(in_raster),
        "-out",
        f'"{os.path.abspath(out_raster)}?&gdal:co:COMPRESS=DEFLATE&{predictor}gdal:co:NUM_THREADS=ALL_CPUS&gdal:co:BIGTIFF=YES"',
        out_datatype,
    ]

    for key, value in options.items():
        cli_args.append("-" + str(key))
        cli_args.append(str(value))

    cli_string = " ".join(cli_args)

    execute_cli_function(cli_string, name="rescale image")

    return os.path.abspath(out_raster)


def otb_merge_rasters(
    in_rasters,
    out_raster,
    options=None,
    band=None,
    out_datatype="uint16",
    tmp=".",
    ram=42000,
    obt_path="C:/Program Files/OTB-7.4.0-Win64/",
    harmonisation=False,
    nodata_value=0,
    pixel_width=None,
    pixel_height=None,
):
    """Creates a mosaic out of a series of images. Must be of the same projection"""

    cli = "otbcli_Mosaic"

    if options is None:
        options = {
            "comp.feather": "slim",
            "comp.feather.slim.length": 1000,
            "harmo.method": "band",
            "harmo.cost": "rmse",
            "interpolator": "linear",
            "tmpdir": tmp,
            "nodata": nodata_value,
        }

        if pixel_width is not None and pixel_height is not None:
            options["output.spacingx"] = pixel_width
            options["output.spacingy"] = pixel_height

        if harmonisation:
            options["harmo.method"] = "band"
        else:
            options["harmo.method"] = "none"

    if band is not None:
        band = f"&bands={band}"
    else:
        band = ""

    cli_args = [
        cli,
        "-il",
        " ".join(in_rasters),
        "-out",
        f'"{os.path.abspath(out_raster)}?{band}&gdal:co:COMPRESS=DEFLATE&gdal:co:NUM_THREADS=ALL_CPUS&gdal:co:BIGTIFF=YES&gdal:co:TILED=YES"',
        out_datatype,
        "-ram",
        str(ram),
    ]

    for key, value in options.items():
        cli_args.append("-" + str(key))
        cli_args.append(str(value))

    cli_string = " ".join(cli_args)

    os.environ["PATH"] += os.pathsep + f"{obt_path}bin"
    os.environ["GDAL_DATA"] = os.pathsep + f"{obt_path}share/data"
    os.environ["PROJ_LIB"] = os.pathsep + f"{obt_path}share/proj"
    os.environ["OTB_APPLICATION_PATH"] = os.pathsep + f"{obt_path}lib/otb/applications"
    os.environ["PYTHONPATH"] = os.pathsep + f"{obt_path}lib/python"
    os.system(cli_string)

    execute_cli_function(cli_string, name="merge rasters")

    return os.path.abspath(out_raster)


def obt_bandmath(
    in_rasters,
    expression,
    out_raster,
    band=None,
    ram=42000,
    out_datatype="float",
    obt_path="C:/Program Files/OTB-7.4.0-Win64/",
):
    """Creates a mosaic out of a series of images. Must be of the same projection"""

    cli = "otbcli_BandMath"

    if band is not None:
        band = f"&bands={band}"
    else:
        band = ""

    cli_args = [
        cli,
        "-il",
        " ".join(in_rasters),
        "-out",
        f'"{os.path.abspath(out_raster)}?{band}&gdal:co:COMPRESS=LZW&gdal:co:NUM_THREADS=ALL_CPUS&gdal:co:BIGTIFF=YES&gdal:co:TILED=YES"',
        out_datatype,
        "-ram",
        str(ram),
        f'-exp "{expression}"',
    ]

    cli_string = " ".join(cli_args)

    os.environ["PATH"] += os.pathsep + f"{obt_path}bin"
    os.environ["GDAL_DATA"] = os.pathsep + f"{obt_path}share/data"
    os.environ["PROJ_LIB"] = os.pathsep + f"{obt_path}share/proj"
    os.environ["OTB_APPLICATION_PATH"] = os.pathsep + f"{obt_path}lib/otb/applications"
    os.environ["PYTHONPATH"] = os.pathsep + f"{obt_path}lib/python"
    os.system(cli_string)

    execute_cli_function(cli_string, name="band_math")

    return os.path.abspath(out_raster)


def otb_meanshift_segmentation(
    in_raster,
    out_geom,
    spatialr=5,
    ranger=15,
    thres=0.1,
    maxiter=100,
    minsize=100,
    mask=False,
    stitch=True,
    neighbor=True,
    vector_minsize=1,
    tilesize=0,
):
    """Computes local statistical moments on every pixel
    in the selected channel of the input image"""

    cli = "otbcli_Segmentation"

    options = {
        "mode": "vector",
        "mode.vector.out": out_geom,
        "mode.vector.neighbor": "false" if neighbor is False else "true",
        "mode.vector.stitch": "false" if stitch is False else "true",
        "mode.vector.tilesize": tilesize,
        "mode.vector.minsize": vector_minsize,
        "mode.vector.outmode": "ovw",
        "filter": "meanshift",
        "filter.meanshift.spatialr": spatialr,
        "filter.meanshift.ranger": ranger,
        "filter.meanshift.thres": thres,
        "filter.meanshift.maxiter": maxiter,
        "filter.meanshift.minsize": minsize,
    }

    if mask is not False:
        options["mode.vector.inmask"] = mask

    cli_args = [cli, "-in", os.path.abspath(in_raster)]

    for key, value in options.items():
        cli_args.append("-" + str(key))
        cli_args.append(str(value))

    cli_string = " ".join(cli_args)

    execute_cli_function(cli_string, name="meanshifting")
