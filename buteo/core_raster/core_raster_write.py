"""### Basic IO functions for working with Rasters. ###"""

# Standard library
from typing import List, Optional, Union

# External
import numpy as np
from osgeo import gdal, ogr, osr

# Internal
from buteo.utils import (
    utils_base,
    utils_gdal,
    utils_path,
    utils_io,
    utils_translate,
    utils_projection,
)
from buteo.core_raster.core_raster_read import _open_raster



def save_dataset_to_disk(
    dataset: Union[
        gdal.Dataset, ogr.DataSource, str, List[Union[gdal.Dataset, ogr.DataSource, str]]
    ],
    out_path: Union[str, List[str]],
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    add_timestamp: bool = False,
    overwrite: bool = True,
    creation_options: Optional[List[str]] = None,
) -> Union[str, List[str]]:
    """Writes a dataset to disk. Can be a raster or a vector. Can be a single dataset or a list of datasets.

    Parameters
    ----------
    dataset : Union[gdal.Dataset, ogr.DataSource, str, List[Union[gdal.Dataset, ogr.DataSource, str]]]
        The dataset to write to disk.
    out_path : Union[str, List[str]]
        The output path or list of output paths.
    prefix : str, optional
        A prefix to add to the output path. Default: "".
    suffix : str, optional
        A suffix to add to the output path. Default: "".
    add_uuid : bool, optional
        If True, a UUID will be added to the output path. Default: False.
    add_timestamp : bool, optional
        If True, a timestamp will be added to the output path. Default: False.
    overwrite : bool, optional
        If True, the output will be overwritten if it already exists. Default: True.
    creation_options : Optional[List[str]], optional
        A list of creation options. Default: None.

    Returns
    -------
    Union[str, List[str]]
        The output path or list of output paths.

    Raises
    ------
    ValueError
        If the dataset type is invalid.
    RuntimeError
        If the driver for the output dataset could not be obtained.
    """
    input_is_list = isinstance(dataset, list)

    in_paths = utils_io._get_input_paths(dataset, "mixed")
    out_paths = utils_io._get_output_paths(
        in_paths,  # type: ignore
        out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        add_timestamp=add_timestamp,
    )

    utils_io._check_overwrite_policy(out_paths, overwrite)
    utils_io._delete_if_required_list(out_paths, overwrite)

    creation_options = utils_gdal._get_default_creation_options(creation_options)

    for idx, ds in enumerate(in_paths):
        driver = None

        if utils_gdal._check_is_raster(ds):
            driver_name = utils_gdal._get_driver_name_from_path(ds)
            driver = gdal.GetDriverByName(driver_name)
            if driver is None:
                raise RuntimeError(f"Could not get GDAL driver for raster: {driver_name}")
            src_ds = gdal.Open(ds)
            if src_ds is None:
                raise ValueError(f"Unable to open raster dataset: {ds}")
            utils_io._delete_if_required(out_paths[idx], overwrite)
            driver.CreateCopy(out_paths[idx], src_ds, options=creation_options)
            src_ds = None

        elif utils_gdal._check_is_vector(ds):
            driver_name = utils_gdal._get_driver_name_from_path(ds)
            driver = ogr.GetDriverByName(driver_name)
            if driver is None:
                raise RuntimeError(f"Could not get OGR driver for vector: {driver_name}")
            src_ds = ogr.Open(ds)
            if src_ds is None:
                raise ValueError(f"Unable to open vector dataset: {ds}")
            utils_io._delete_if_required(out_paths[idx], overwrite)
            driver.CopyDataSource(src_ds, out_paths[idx])
            src_ds = None

        else:
            raise ValueError(f"Invalid dataset type: {ds}")

    return out_paths if input_is_list else out_paths[0]


def raster_create_empty(
    out_path: Optional[str] = None,
    *,
    width: int = 100,
    height: int = 100,
    pixel_size: Union[float, int, List[Union[float, int]]] = 10.0,
    bands: int = 1,
    dtype: Union[str, int, np.dtype] = "uint8",
    x_min: Union[float, int] = 0.0,
    y_max: Union[float, int] = 0.0,
    nodata_value: Optional[Union[float, int]] = None,
    fill_value: Optional[Union[float, int]] = 0.0,
    projection: Union[int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference] = "EPSG:3857",
    creation_options: Optional[List[str]] = None,
    overwrite: bool = True,
) -> str:
    """Create an empty raster.

    Parameters
    ----------
    out_path : str, optional
        The output path. If None, a temporary file will be created.
    width : int, optional
        The width of the raster in pixels. Default: 100.
    height : int, optional
        The height of the raster in pixels. Default: 100.
    pixel_size : int or float or list or tuple, optional
        The pixel size in units of the projection. Default: 10.0.
    bands : int, optional
        The number of bands in the raster. Default: 1.
    dtype : str, int, np.dtype, optional
        The data type of the raster. Default: "uint8".
    x_min : int or float, optional
        The x coordinate of the top left corner of the raster. Default: 0.0.
    y_max : int or float, optional
        The y coordinate of the top left corner of the raster. Default: 0.0.
    nodata_value : int or float or None, optional
        The nodata value of the raster. Default: None.
    fill_value : int or float or None, optional
        The fill value of the raster. Default: 0.0.
    projection : int or str or gdal.Dataset or ogr.DataSource or osr.SpatialReference, optional
        The projection of the raster. Default: "EPSG:3857".
    creation_options : list or None, optional
        A list of creation options. Default: None.
    overwrite : bool, optional
        If True, overwrite the output file if it exists. Default: True.

    Returns
    -------
    str
        The path to the output raster.

    Raises
    ------
    ValueError
        If any of the input parameters are invalid.
    RuntimeError
        If the GDAL driver cannot be obtained or raster creation fails.
    """
    utils_base._type_check(out_path, [str, type(None)], "out_path")
    utils_base._type_check(width, [int], "width")
    utils_base._type_check(height, [int], "height")
    utils_base._type_check(pixel_size, [int, float, list, tuple], "pixel_size")
    utils_base._type_check(bands, [int], "bands")
    utils_base._type_check(dtype, [str, int, np.dtype], "dtype")
    utils_base._type_check(x_min, [int, float], "x_min")
    utils_base._type_check(y_max, [int, float], "y_max")
    utils_base._type_check(nodata_value, [int, float, type(None)], "nodata_value")
    utils_base._type_check(fill_value, [int, float, type(None)], "fill_value")
    utils_base._type_check(projection, [int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference], "projection")
    utils_base._type_check(creation_options, [list, type(None)], "creation_options")
    utils_base._type_check(overwrite, [bool], "overwrite")

    if out_path and not utils_path._check_is_valid_output_filepath(out_path, overwrite=overwrite):
        raise ValueError(
            f"Output path {out_path} is not valid or already exists. "
            "Set overwrite to True to overwrite existing files."
        )

    if out_path is None:
        out_path = utils_path._get_temp_filepath(name="raster_create_empty", ext="tif", add_timestamp=True)

    driver_name = utils_gdal._get_driver_name_from_path(out_path)
    driver = gdal.GetDriverByName(driver_name)
    if driver is None:
        raise RuntimeError(f"GDAL driver '{driver_name}' not found.")

    utils_io._check_overwrite_policy([out_path], overwrite)
    utils_io._delete_if_required(out_path, overwrite)

    try:
        dtype_gdal = utils_translate._translate_dtype_numpy_to_gdal(utils_translate._parse_dtype(dtype))
    except Exception as e:
        raise ValueError(f"Invalid dtype '{dtype}': {e}") from e

    destination = driver.Create(
        out_path,
        width,
        height,
        bands,
        dtype_gdal,
        utils_gdal._get_default_creation_options(creation_options),
    )
    if destination is None:
        raise RuntimeError("Failed to create raster destination.")

    parsed_projection = utils_projection.parse_projection_wkt(projection)
    destination.SetProjection(parsed_projection)

    pixel_width = pixel_size if isinstance(pixel_size, (int, float)) else pixel_size[0]
    pixel_height = pixel_size if isinstance(pixel_size, (int, float)) else pixel_size[1]

    transform = [x_min, pixel_width, 0, y_max, 0, -pixel_height]  # negative for north-up
    destination.SetGeoTransform(transform)

    if nodata_value is not None or fill_value is not None:
        for band in range(1, bands + 1):
            band_obj = destination.GetRasterBand(band)
            if nodata_value is not None:
                band_obj.SetNoDataValue(nodata_value)
            if fill_value is not None:
                band_obj.Fill(fill_value)

    destination.FlushCache()
    destination = None

    return out_path


def raster_create_from_array(
    arr: np.ndarray,
    out_path: Optional[str] = None,
    pixel_size: Union[Union[float, int], List[Union[float, int]]] = 1.0,
    x_min: Union[float, int] = 0.0,
    y_max: Union[float, int] = 0.0,
    projection: Union[int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference] = "EPSG:3857",
    channel_last: bool = True,
    overwrite: bool = True,
    creation_options: Union[List[str], None] = None,
) -> str:
    """Create a raster from a numpy array.

        Parameters
        ----------
        arr : np.ndarray
            The array to convert to a raster.
        out_path : str, optional
            The output path. If None, a temporary file will be created.
        pixel_size : int or float or list or tuple, optional
            The pixel size of the output raster. Default: 10.0.
        x_min : int or float, optional
            The x coordinate of the top left corner of the output raster. Default: 0.0.
        y_max : int or float, optional
            The y coordinate of the top left corner of the output raster. Default: 0.0.
        projection : int or str or gdal.Dataset or ogr.DataSource or osr.SpatialReference, optional
            The projection of the output raster. Default: "EPSG:3857".
        creation_options : list or None, optional
            The creation options for the output raster. Default: None.
        overwrite : bool, optional
            If True, the output raster will be overwritten if it already exists. Default: True.

        Returns
        -------
        str
       The path to the output raster.
    """
    utils_base._type_check(arr, [np.ndarray, np.ma.MaskedArray], "arr")
    utils_base._type_check(out_path, [str, None], "out_path")
    utils_base._type_check(pixel_size, [int, float, [int, float], tuple], "pixel_size")
    utils_base._type_check(x_min, [int, float], "x_min")
    utils_base._type_check(y_max, [int, float], "y_max")
    utils_base._type_check(projection, [int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference], "projection")
    utils_base._type_check(creation_options, [[str], None], "creation_options")
    utils_base._type_check(overwrite, [bool], "overwrite")

    if arr.ndim not in [2, 3]:
        raise ValueError("Array must be 2 or 3 dimensional (3rd dimension considered bands.)")

    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    if not channel_last:
        arr = np.transpose(arr, (2, 0, 1))

    if out_path is not None and not utils_path._check_is_valid_output_filepath(out_path, overwrite=overwrite):
        raise ValueError(
            f"Output path {out_path} is not valid or already exists. "
            "Set overwrite to True to overwrite existing files."
        )

    if out_path is None:
        out_path = utils_path._get_temp_filepath(name="raster_create_from_array", ext="tif", add_timestamp=True)

    driver_name = utils_gdal._get_driver_name_from_path(out_path)
    driver = gdal.GetDriverByName(driver_name)

    utils_io._check_overwrite_policy([out_path], overwrite)
    utils_io._delete_if_required(out_path, overwrite)

    height, width, bands = arr.shape

    destination = driver.Create(
        out_path,
        width,
        height,
        bands,
        utils_translate._translate_dtype_numpy_to_gdal(arr.dtype),
        utils_gdal._get_default_creation_options(creation_options),
    )

    parsed_projection = utils_projection.parse_projection_wkt(projection)

    destination.SetProjection(parsed_projection)

    pixel_width = pixel_size if isinstance(pixel_size, (int,  float)) else pixel_size[0]
    pixel_height = pixel_size if isinstance(pixel_size, (int,  float)) else pixel_size[1]

    transform = [x_min, pixel_width, 0, y_max, 0, -pixel_height] # negative for north-up

    destination.SetGeoTransform(transform)

    nodata = None
    nodata_value = 0.0

    if isinstance(arr, np.ma.MaskedArray):
        nodata = True
        nodata_value = arr.fill_value
        arr = np.ma.getdata(arr.filled(nodata_value))

    for idx in range(0, bands):
        dst_band = destination.GetRasterBand(idx + 1)
        dst_band.WriteArray(arr[:, :, idx])

        if nodata:
            dst_band.SetNoDataValue(nodata_value)

    return out_path


def raster_create_copy(
    raster: Union[str, gdal.Dataset],
    out_path: Optional[str] = None,
    overwrite: bool = True,
) -> str:
    """Create a copy of a raster.

    Parameters
    ----------
    raster : str or gdal.Dataset
        The raster to copy.
    out_path : str, optional
        The output path. If None, a temporary file will be created.

    Returns
    -------
    str
        The path to the output raster.
    """
    assert utils_gdal._check_is_raster(raster), "Raster is not valid."

    if out_path is None:
        out_path = utils_path._get_temp_filepath(
            utils_gdal._get_path_from_dataset(raster),
            ext="tif",
        )
    else:
        assert utils_path._check_is_valid_output_filepath(out_path, overwrite=overwrite), (
            f"Output path {out_path} is not valid or already exists. "
        )

    utils_io._check_overwrite_policy([out_path], overwrite)
    utils_io._delete_if_required(out_path, overwrite)

    driver_name = utils_gdal._get_driver_name_from_path(out_path)
    driver = gdal.GetDriverByName(driver_name)

    src_ds = _open_raster(raster)
    dst_ds = driver.CreateCopy(out_path, src_ds) # pylint: disable=unused-variable
    dst_ds = None
    src_ds = None

    return out_path


def raster_set_band_descriptions(raster, bands, descriptions):
    """Update the band descriptions of a raster.

    Parameters
    ----------
    raster : str
        The raster to update.
    bands : list
        The bands to update. e.g. [1, 2, 3] for rgb. Starts at 1.
    descriptions : list
        The descriptions to set. e.g. ["Red", "Green", "Blue"].

    Returns
    -------
    str
        The path to the updated raster. (Same as input)
    """
    assert len(bands) == len(descriptions), "Bands and descriptions must be the same length."
    assert all([isinstance(band, int) for band in bands]), "Bands must be a list of integers."
    assert all([isinstance(description, str) for description in descriptions]), "Descriptions must be a list of strings."
    assert utils_gdal._check_is_raster(raster), "Raster is not valid."

    ds = gdal.Open(raster, gdal.GA_Update)

    for idx, band in enumerate(bands):
        rb = ds.GetRasterBand(band)
        rb.SetDescription(descriptions[idx])

    ds.FlushCache()
    ds = None

    return raster
