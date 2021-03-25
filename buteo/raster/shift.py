import sys; sys.path.append('../../')
from typing import Union
from osgeo import gdal
from buteo.utils import remove_if_overwrite, is_number
from buteo.gdal_utils import (
    path_to_driver,
    raster_to_reference,
    default_options,
)
from buteo.raster.io import (
    default_options,
    raster_to_metadata,
)


def shift_raster(
    raster: Union[str, gdal.Dataset],
    shift: Union[tuple, list],
    out_path: Union[str, None]=None,
    overwrite: bool=True,
    creation_options: list=[],
) -> Union[gdal.Dataset, str]:
    """ Reprojects a raster given a target projection.

    Args:
        raster (path | raster): The raster to reproject.
        
        target_size (str | int | vector | raster): The target resolution of the
        raster. In the same unit as the projection of the raster. Beware if your
        input is in latitude and longitude, you'll need to specify degrees as well!
        It's better to reproject to a projected coordinate system for resampling.

    **kwargs:
        out_path (path | None): The destination to save to. If None then
        the output is an in-memory raster.

        overwite (bool): Is it possible to overwrite the out_path if it exists.

        creation_options (list): A list of options for the GDAL creation. Only
        used if an outpath is specified. Defaults are:
            "TILED=YES"
            "NUM_THREADS=ALL_CPUS"
            "BIGG_TIF=YES"
            "COMPRESS=LZW"

    Returns:
        An in-memory raster. If an out_path is given the output is a string containing
        the path to the newly created raster.
    """
    ref = raster_to_reference(raster)
    metadata = raster_to_metadata(ref)

    x_shift = None
    y_shift = None
    if isinstance(shift, tuple) or isinstance(shift, list):
        if len(shift) == 1:
            if is_number(shift[0]):
                x_shift = float(shift[0])
                y_shift = float(shift[0])
            else:
                raise ValueError("shift is not a number or a list/tuple of numbers.")
        elif len(shift) == 2:
            if is_number(shift[0]) and is_number(shift[1]):
                x_shift = float(shift[0])
                y_shift = float(shift[1])
        else:
            raise ValueError("shift is either empty or larger than 2.")
    elif is_number(shift):
        x_shift = float(shift)
        y_shift = float(shift)
    else:
        raise ValueError("shift is invalid.")

    out_name = None
    out_format = None
    out_creation_options = []
    if out_path is None:
        out_name = metadata["name"]
        out_format = "MEM"
    else:
        out_creation_options = default_options(creation_options)
        out_name = out_path
        out_format = path_to_driver(out_path)

    remove_if_overwrite(out_path, overwrite)

    driver = gdal.GetDriverByName(out_format)

    shifted = driver.Create(
        out_name,  # Location of the saved raster, ignored if driver is memory.
        metadata["width"],  # Dataframe width in pixels (e.g. 1920px).
        metadata["height"],  # Dataframe height in pixels (e.g. 1280px).
        metadata["bands"],  # The number of bands required.
        metadata["dtype_gdal_raw"],  # Datatype of the destination.
        out_creation_options,
    )

    new_transform = list(metadata["transform"])
    new_transform[0] += x_shift
    new_transform[3] += y_shift

    shifted.SetGeoTransform(new_transform)
    shifted.SetProjection(metadata["projection"])

    src_nodata = metadata["nodata_value"]

    for band in range(metadata["bands"]):
        origin_raster_band = ref.GetRasterBand(band + 1)
        target_raster_band = shifted.GetRasterBand(band + 1)

        target_raster_band.WriteArray(origin_raster_band.ReadAsArray())
        target_raster_band.SetNoDataValue(src_nodata)

    if out_path is not None:
        shifted = None
        return out_path
    else:
        return shifted
