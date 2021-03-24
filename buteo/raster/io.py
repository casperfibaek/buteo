import sys; sys.path.append('../../')
from osgeo import gdal, ogr, osr
from typing import Union
import numpy as np
import os, json

from buteo.utils import (
    remove_if_overwrite,
    path_to_name,
)
from buteo.gdal_utils import (
    is_raster,
    raster_to_reference,
    path_to_driver,
    numpy_to_gdal_datatype,
    gdal_to_numpy_datatype,
    default_options,
)


# TODO: Create empty or filled raster.
# TODO: Array to raster (advanced)


def raster_to_memory(raster: Union[str, gdal.Dataset]) -> gdal.Dataset:
    """ Takes a file path or a gdal raster dataset and copies
    it to memory. 

    Args:
        file_path (path | Dataset): A path to a raster or a GDAL dataframe.

    Returns:
        A gdal.Dataset copied into memory.
    """
    ref = raster_to_reference(raster)
    driver = gdal.GetDriverByName("MEM")

    name = path_to_name(ref.GetDescription())

    copy = driver.CreateCopy(name, ref)

    return copy


def raster_to_disk(
    raster: Union[str, gdal.Dataset],
    out_path: str,
    overwrite: bool=True,
    creation_options: list=[],
) -> str:
    """ Saves or copies a raster to disc. Input is either a 
    filepath to a raster or a GDAL.Dataset. The driver is infered
    from the file extension.

    Args:
        raster (path | Dataset): The raster to save to disc.
        out_path (path): The destination to save to.

    **kwargs:
        overwrite (bool): If the file exists, should it be overwritten?
        creation_options (list): GDAL creation options. Defaults are:
            "TILED=YES"
            "NUM_THREADS=ALL_CPUS"
            "BIGG_TIF=YES"
            "COMPRESS=LZW"

    Returns:
        The filepath for the newly created raster.
    """
    ref = raster_to_reference(raster)

    remove_if_overwrite(out_path, overwrite)
    
    driver = gdal.GetDriverByName(path_to_driver(out_path))
    if driver is None:
        raise ValueError(f"Unable to parse filetype from extension: {out_path}")

    if not isinstance(creation_options, list):
        raise ValueError("Options must be a list. ['BIGTIFF=YES', ...]")

    copy_created = driver.CreateCopy(out_path, ref, options=default_options(creation_options))

    if copy_created is None:
        raise Exception("Error while creating copy.")

    return out_path


def raster_to_metadata(raster: Union[str, gdal.Dataset], latlng_and_footprint: bool=True) -> dict:
    """ Reads a raster from a string or a dataset and returns metadata.

    Args:
        raster (path | Dataset): The raster to save to disc.

    **kwargs:
        latlng_and_footprint (bool): Should the metadata include a
            footprint of the raster in wgs84. Requires a reprojection
            check do not use it if not required and performance is important.

    Returns:
        A dictionary containing metadata about the raster.
    """
    metadata = {}
    metadata["dataframe"] = raster_to_reference(raster)
    metadata["name"] = metadata["dataframe"].GetDescription()

    metadata["transform"] = metadata["dataframe"].GetGeoTransform()
    metadata["projection"] = metadata["dataframe"].GetProjection()

    original_projection = osr.SpatialReference()
    original_projection.ImportFromWkt(metadata["projection"])
    metadata["projection_osr"] = original_projection

    band0 = metadata["dataframe"].GetRasterBand(1)

    if latlng_and_footprint:
        wgs84 = osr.SpatialReference()
        wgs84.ImportFromEPSG(4326)
        tx = osr.CoordinateTransformation(original_projection, wgs84)

    metadata["width"] = metadata["dataframe"].RasterXSize
    metadata["height"] = metadata["dataframe"].RasterYSize
    metadata["size"] = [metadata["dataframe"].RasterXSize, metadata["dataframe"].RasterYSize]
    metadata["shape"] = (metadata["width"], metadata["height"])
    metadata["pixel_width"] = metadata["transform"][1]
    metadata["pixel_height"] = metadata["transform"][5]
    metadata["minx"] = metadata["transform"][0]
    metadata["bands"] = metadata["dataframe"].RasterCount
    metadata["miny"] = (
        metadata["transform"][3]
        + metadata["width"] * metadata["transform"][4]
        + metadata["height"] * metadata["transform"][5]
    )
    metadata["maxx"] = (
        metadata["transform"][0]
        + metadata["width"] * metadata["transform"][1]
        + metadata["height"] * metadata["transform"][2]
    )
    metadata["maxy"] = metadata["transform"][3]

    metadata["dtype_gdal_raw"] = band0.DataType
    metadata["datatype_gdal_raw"] = metadata["dtype_gdal_raw"]

    metadata["dtype_gdal"] = gdal.GetDataTypeName(metadata["dtype_gdal_raw"])
    metadata["datatype_gdal"] = metadata["dtype_gdal"]

    metadata["dtype"] = gdal_to_numpy_datatype(band0.DataType)
    metadata["datatype"] = metadata["dtype"]

    metadata["nodata_value"] = band0.GetNoDataValue()

    # ulx, uly, lrx, lry = -180, 90, 180, -90
    metadata["extent"] = [
        metadata["minx"],
        metadata["maxy"],
        metadata["maxx"],
        metadata["miny"],
    ]

    if latlng_and_footprint:
        bottom_left = tx.TransformPoint(metadata["minx"], metadata["miny"])
        top_left = tx.TransformPoint(metadata["minx"], metadata["maxy"])
        top_right = tx.TransformPoint(metadata["maxx"], metadata["maxy"])
        bottom_right = tx.TransformPoint(metadata["maxx"], metadata["miny"])

        # ulx, uly, lrx, lry = -180, 90, 180, -90
        metadata["extent_wgs84"] = [
            top_left[0],
            top_left[1],
            bottom_right[0],
            bottom_right[1],
        ]

        coord_array = [
            [bottom_left[1], bottom_left[0]],
            [top_left[1], top_left[0]],
            [top_right[1], top_right[0]],
            [bottom_right[1], bottom_right[0]],
            [bottom_left[1], bottom_left[0]],
        ]

        wkt_coords = ""
        for coord in coord_array:
            wkt_coords += f"{coord[1]} {coord[0]}, "
        wkt_coords = wkt_coords[:-2]

        metadata["extent_wkt"] = f"POLYGON (({wkt_coords}))"

        # Create an OGR Datasource in memory with the extent
        extent_name = str(metadata["name"]) + "_extent"

        driver = ogr.GetDriverByName("Memory")
        extent_ogr = driver.CreateDataSource(extent_name)
        layer = extent_ogr.CreateLayer(extent_name + "_layer", wgs84, ogr.wkbPolygon)

        feature = ogr.Feature(layer.GetLayerDefn())
        extent_geom = ogr.CreateGeometryFromWkt(metadata["extent_wkt"], wgs84)
        feature.SetGeometry(extent_geom)
        layer.CreateFeature(feature)
        feature = None

        metadata["extent_ogr"] = extent_ogr
        metadata["extent_ogr_geom"] = extent_geom

        metadata["extent_geojson_dict"] = {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [coord_array],
            },
        }
        metadata["extent_geojson"] = json.dumps(metadata["extent_geojson_dict"])

    return metadata


def array_to_raster(
    array: np.ndarray,
    reference: Union[str, gdal.Dataset],
    out_path: Union[str, None]=None,
    overwrite: bool=True,
    creation_options: list=[],
) -> Union[str, gdal.Dataset]:
    """ Turns a numpy array into a GDAL dataset or exported
        as a raster using a reference raster.

    Args:
        array (np.ndarray): The numpy array to convert

        reference (path or Dataset): A reference on which to base
        the geographical extent and projection of the raster.

    **kwargs:
        out_path (path): The location to save the raster. If None is
        supplied an in memory raster is returned. filetype is infered
        from the extension.

        overwrite (bool): Specifies if the file already exists, should
        it be overwritten?

        creation_options (list): A list of options for the GDAL creation. Only
        used if an outpath is specified. Defaults are:
            "TILED=YES"
            "NUM_THREADS=ALL_CPUS"
            "BIGG_TIF=YES"
            "COMPRESS=LZW"

    Returns:
        If an out_path has been specified, it returns the path to the 
        newly created raster file. If 
    """

    # Verify inputs
    if not isinstance(reference, gdal.Dataset) and not isinstance(reference, str):
        raise ValueError("A valid reference raster must be supplied.")

    if isinstance(reference, str):
        if not is_raster(reference):
            raise ValueError("A valid reference raster must be supplied.")
    
    if not isinstance(out_path, str) and out_path != None:
        raise TypeError("out_path must be None or a path")

    # Verify the numpy array
    if not isinstance(array, np.ndarray) or array.size == 0 or array.ndim < 2 or array.ndim > 3:
        raise ValueError(f"Input array is invalid {array}")

    if not isinstance(overwrite, bool) and not isinstance(overwrite, int):
        if isinstance(overwrite, int):
            if overwrite != 0 and overwrite != 1:
                raise ValueError("overwrite parameter must be a boolean, 0, or 1.")
    else:
        raise TypeError("overwrite parameter must be a boolean, 0, or 1.")
    
    if not isinstance(creation_options, list):
        raise TypeError("Options must be a list of valid GDAL options or empty list.")

    # Parse the driver
    driver_name = "MEM" if out_path is None else path_to_driver(out_path)
    if driver_name is None:
        raise ValueError(f"Unable to parse filetype from path: {out_path}")
    
    driver = gdal.GetDriverByName(driver_name)
    if driver is None:
        raise ValueError(f"Error while creating driver from extension: {out_path}")

    # How many bands?
    bands = 1
    if array.ndim == 3:
        bands = array.shape[2]

    remove_if_overwrite(out_path, overwrite)

    metadata = raster_to_metadata(reference, latlng_and_footprint=False)

    # Handle nodata
    input_nodata = None
    if np.ma.is_masked(array) is True:
        input_nodata = array.get_fill_value()

    destination_dtype = numpy_to_gdal_datatype(array.dtype)

    # Weird double issue with GDAL and numpy. Cast to float or int
    if input_nodata is not None:
        input_nodata = float(input_nodata)
        if (input_nodata).is_integer() is True:
            input_nodata = int(input_nodata)

    # If the output is not memory, set compression options.
    creation_options = []
    if driver_name != "MEM":
        creation_options = default_options(creation_options)

    output_name = metadata["name"] if out_path is None else out_path

    if metadata["width"] != array.shape[1] or metadata["height"] != array.shape[0]:
        raise ValueError("The numpy array and the reference are not of the same size. Use the array_to_raster_adv function instead")

    destination = driver.Create(
        output_name,
        array.shape[1],
        array.shape[0],
        bands,
        destination_dtype,
        creation_options,
    )

    destination.SetProjection(metadata["projection"])
    destination.SetGeoTransform(metadata["transform"])

    for band_idx in range(bands):
        band = destination.GetRasterBand(band_idx + 1)
        if bands > 1 or array.ndim == 3:
            band.WriteArray(array[:, :, band_idx])
        else:
            band.WriteArray(array)

        if metadata["nodata_value"] == None and input_nodata is not None:
            band.SetNodataValue(input_nodata)

    # Return the destination raster
    if driver != "MEM":
        return os.path.abspath(out_path)
    else:
        return destination["dataframe"]


def raster_to_array(
    raster: Union[str, gdal.Dataset],
    bands: Union[str, int, list]="all",
    filled: bool=False,
    first_band: bool=False,
) -> np.ndarray:
    """ Turns a path to a raster or a GDAL.Dataset into a numpy
        array.

    Args:
        raster (path | Dataset): The raster to convert.

    **kwargs:
        bands (str | int | list): The bands from the raster to turn
        into a numpy array. Can be "all", "ALL", a list of ints or a
        single int.

        filled (bool): If the array contains nodata values. Should the
        resulting array be a filled numpy array or a masked array?

        first_band (bool): If True, returns only the first band in a 2D
        fashion eg. (1920x1080) instead of the default channel-last format
        (1920x1080x1)

    Returns:
        A numpy array in the channel-last format unless first_band is
        specified.
    """
    ref = raster_to_reference(raster)
    metadata = raster_to_metadata(ref, latlng_and_footprint=False)
    band_count = metadata["bands"]

    internal_bands = []

    if band_count == 0:
        raise ValueError("The input raster does not have any valid bands.")

    # Verify inputs
    if isinstance(bands, str):
        if bands != "all" and bands != "ALL":
            raise ValueError(f"Unable to parse bands. Passed: {bands}")
        else:
            for val in range(band_count):
                internal_bands.append(val)

    elif isinstance(bands, list):
        if len(bands) == 0:
            raise ValueError("Provided list of bands is empty.")
        for val in bands:
            try:
                band_int = int(val)
            except:
                raise ValueError(f"List of bands contained non-valid band number: {val}")

            if band_int > band_count + 1:
                raise ValueError("Requested a higher band that is available in raster.")
            else:
                internal_bands.append(band_int)

    elif isinstance(bands, int):
        if bands > band_count + 1:
            raise ValueError("Requested a higher band that is available in raster.")
        else:
            internal_bands.append(bands)

    if not isinstance(filled, bool) and not isinstance(filled, int):
        if isinstance(filled, int):
            if filled != 0 and filled != 1:
                raise ValueError("Filled parameter must be a boolean, 0, or 1.")
    else:
        raise TypeError("Filled parameter must be a boolean, 0, or 1.")

    if not isinstance(first_band, bool) and not isinstance(first_band, int):
        if isinstance(first_band, int):
            if first_band != 0 and first_band != 1:
                raise ValueError("first_band parameter must be a boolean, 0, or 1.")
    else:
        raise TypeError("first_band parameter must be a boolean, 0, or 1.")

    band_stack = []
    for band in internal_bands:
        band_ref = ref.GetRasterBand(band + 1)
        band_nodata_value = band_ref.GetNoDataValue()

        arr = band_ref.ReadAsArray()

        if band_nodata_value is None and filled is None:
            arr = np.ma.masked_invalid(arr)
        elif filled is None:
            arr = np.ma.masked_equal(arr, band_nodata_value)

        if first_band:
            return arr

        band_stack.append(arr)

    return np.dstack(band_stack)


def empty_raster(filetype: str="MEM", path: Union[str, None]=None) -> Union[gdal.Dataset, str]:
    wgs84 = osr.SpatialReference()
    wgs84.ImportFromEPSG(4326) # WGS84, latlng

    driver = gdal.GetDriverByName(filetype)
    destination = driver.Create(path, 0, 0, 1, 1, default_options([]))
    destination.SetProjection(wgs84.ExportToWkt())

    if filetype == "MEM":
        return destination
    else:
        return path
