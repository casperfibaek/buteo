import sys; sys.path.append('../../')
import os
import numpy as np
from typing import Union, Sequence
from osgeo import gdal
from buteo.utils import overwrite_required, remove_if_overwrite
from buteo.gdal_utils import is_raster, gdal_nodata_value_from_type
from buteo.raster.io import raster_to_array, raster_to_disk, raster_to_memory, raster_to_metadata, array_to_raster


# TODO: raster_to_mask
# TODO: raster_invert_nodata


def raster_has_nodata_value(
    raster: Union[gdal.Dataset, str, list],
) -> Union[bool, Sequence[bool]]:
    """ Check if a raster or a list of rasters contain nodata values

    Args:
        raster (path | raster | list): The raster(s) to check for nodata values.

    Returns:
        True if input raster has nodata values. If a list is the input, the output
        is a list of booleans indicating if the input raster has nodata values.
    """
    rasters = []
    nodata_values = []
    if isinstance(raster, (gdal.Dataset, str)):
        rasters.append(raster)
    elif isinstance(raster, list):
        rasters = raster
    else:
        raise ValueError(f"Input raster is invalid: {raster}")
    
    for internal_raster in rasters:
        if not is_raster(internal_raster):
            raise ValueError(f"Input raster is invalid: {internal_raster}")
        
        raster_metadata = raster_to_metadata(internal_raster)
        raster_nodata = raster_metadata["nodata_value"]

        if raster_nodata is not None:
            nodata_values.append(True)
        else:
            nodata_values.append(False)

    if isinstance(raster, list):
        return nodata_values
    else:
        return nodata_values[0]


def raster_get_nodata_value(
   raster: Union[gdal.Dataset, str, list],
) -> Union[float, None]:
    """ Get the nodata value of a raster or a from a list of rasters.

    Args:
        raster (path | raster | list): The raster(s) to retrieve nodata values from.

    Returns:
        Returns the nodata value from a raster or a list of rasters
    """
    rasters = []
    nodata_values = []
    if isinstance(raster, (gdal.Dataset, str)):
        rasters.append(raster)
    elif isinstance(raster, list):
        rasters = raster
    else:
        raise ValueError(f"Input raster is invalid: {raster}")
    
    for internal_raster in rasters:
        if not is_raster(internal_raster):
            raise ValueError(f"Input raster is invalid: {internal_raster}")
        
        raster_metadata = raster_to_metadata(internal_raster)
        raster_nodata = raster_metadata["nodata_value"]

        nodata_values.append(raster_nodata)

    if isinstance(raster, list):
        return nodata_values
    else:
        return nodata_values[0]


def raster_set_nodata(
    raster: Union[gdal.Dataset, str, list],
    dst_nodata: Union[float, int, str, list, None],
    out_path: Union[str, Sequence[str], None]=None,
    in_place: bool=False,
    overwrite: bool=True,
    prefix: str="",
    postfix: str="_nodata_set",
) -> Union[gdal.Dataset, Sequence[gdal.Dataset], str, Sequence[str]]:
    """ Sets all the nodata from a raster or a list of rasters.

    Args:
        raster (path | raster | list): The raster(s) to retrieve nodata values from.

        dst_nodata (float, int, str, None): The target nodata value. If 'infer' the nodata
        value is set based on the input datatype. A list of nodata values can be based matching
        the amount of input rasters. If multiple nodata values should be set, use raster_mask_values.
    
    **kwargs:
        out_path (path | list | None): The destination of the changed rasters. If out_paths
        are specified, in_place is automatically set to False. The path can be a folder.

        in_place (bool): Should the rasters be changed in_place or copied?

        prefix (str): Prefix to add the the output if a folder is specified in out_path.

        postfix (str): Postfix to add the the output if a folder is specified in out_path.

    Returns:
        Returns the rasters with nodata set. If in_place is True a reference to the 
        changed orignal is returned, otherwise a copied memory raster or the path to the
        generated raster is outputted.
    """
    rasters = []
    output_rasters = []
    rasters_metadata = []
    internal_in_place = in_place if out_path is None else False
    internal_dst_nodata = None

    if not isinstance(dst_nodata, (float, int, str, list)) and dst_nodata is not None:
        raise ValueError(f"Invalid dst_nodata value. {dst_nodata}")
    
    if isinstance(dst_nodata, str) and dst_nodata != "infer":
        raise ValueError(f"Invalid dst_nodata value. {dst_nodata}")
    
    if isinstance(dst_nodata, list):
        if not isinstance(raster, list) or len(dst_nodata) != len(raster):
            raise ValueError("If dst_nodata is a list, raster must also be a list of equal length.")
        
        for value in dst_nodata:
            if isinstance(value, (float, int, str, None)):
                raise ValueError("Invalid type in dst_nodata list.")
            
            if isinstance(value, str) and value != "infer":
                raise ValueError("If dst_nodata is a string it must be 'infer'")

    if isinstance(raster, (gdal.Dataset, str)):
        rasters.append(raster)
    elif isinstance(raster, list):
        rasters = raster
    else:
        raise ValueError(f"Input raster is invalid: {raster}")

    if isinstance(out_path, list):
        if not isinstance(raster, list):
            raise ValueError("If out_path is a list, the input raster must also be a list.")
        
        if len(out_path) != len(raster):
            raise ValueError("list of out_path and list of input rasters must match in length.")
        
        for path in out_path:
            overwrite_required(path, overwrite)
            output_rasters.append(path)
    elif isinstance(out_path, str):
        if os.path.isdir(out_path):
            for internal_raster in rasters:
                raster_metadata = raster_to_metadata(internal_raster)
                rasters_metadata.append(raster_metadata)

                raster_basename = raster_metadata["basename"]
                raster_ext = raster_metadata["ext"]
                path = f"{out_path}{prefix}{raster_basename}{postfix}{raster_ext}"
                
                # Check if file exists and overwrite is required.
                overwrite_required(path, overwrite)
                output_rasters.append(path)
        else:
            raise ValueError(f"Unable to parse out_path: {out_path}")

    elif out_path != None:
        raise ValueError(f"out_path is invalid: {raster}")

    for index, internal_raster in enumerate(rasters):
        if not is_raster(internal_raster):
            raise ValueError(f"Input raster is invalid: {internal_raster}")
        
        raster_metadata = None
        if len(rasters_metadata) == 0:
            raster_metadata = raster_to_metadata(internal_raster)
            rasters_metadata.append(raster_metadata)
        else:
            raster_metadata = rasters_metadata[index]
        
        if dst_nodata == "infer":
            internal_dst_nodata = gdal_nodata_value_from_type(raster_metadata["dtype_gdal_raw"])
        elif isinstance(dst_nodata, list):
            internal_dst_nodata = dst_nodata[index]
        else:
            internal_dst_nodata = dst_nodata

        if internal_in_place:
            for band in range(raster_metadata["bands"]):
                raster_band = internal_raster.GetRasterBand(band + 1)
                raster_band.SetNodataValue(internal_dst_nodata)
                raster_band = None
        else:
            raster_mem = raster_to_memory(internal_raster)

            for band in range(raster_metadata["bands"]):
                raster_band = raster_mem.GetRasterBand(band + 1)
                raster_band.SetNodataValue(internal_dst_nodata)
            
            if out_path is None:
                output_rasters.append(raster_mem)
            else:
                path = out_path[index]
                remove_if_overwrite(path, overwrite)

                raster_to_disk(raster_mem, path, overwrite=overwrite)

    if internal_in_place:
        return raster

    if isinstance(raster, list):
        return output_rasters
    
    return output_rasters[0]


def raster_remove_nodata(
    raster: Union[gdal.Dataset, str, list],
    out_path: Union[str, Sequence[str], None]=None,
    in_place: bool=False,
    overwrite: bool=True,
    prefix: str="",
    postfix: str="_nodata_removed",
) -> Union[gdal.Dataset, Sequence[gdal.Dataset], str, Sequence[str]]:
    """ Removes all the nodata from a raster or a list of rasters.

    Args:
        raster (path | raster | list): The raster(s) to retrieve nodata values from.
    
    **kwargs:
        out_path (path | list | None): The destination of the changed rasters. If out_paths
        are specified, in_place is automatically set to False. The path can be a folder.

        in_place (bool): Should the rasters be changed in_place or copied?

        prefix (str): Prefix to add the the output if a folder is specified in out_path.

        postfix (str): Postfix to add the the output if a folder is specified in out_path.

    Returns:
        Returns the rasters with nodata removed. If in_place is True a reference to the 
        changed orignal is returned, otherwise a copied memory raster or the path to the
        generated raster is outputted.
    """

    return raster_set_nodata(
        raster,
        dst_nodata=None,
        out_path=out_path,
        in_place=in_place,
        overwrite=overwrite,
        prefix=prefix,
        postfix=postfix,
    )


def raster_mask_values(
    raster: Union[gdal.Dataset, str, list],
    values_to_mask: list,
    include_original_nodata: bool=True,
    dst_nodata: Union[float, int, str, list, None]="infer",
    out_path: Union[str, Sequence[str], None]=None,
    in_place: bool=False,
    overwrite: bool=True,
    prefix: str="",
    postfix: str="_nodata_masked",
) -> Union[gdal.Dataset, Sequence[gdal.Dataset], str, Sequence[str]]:
    """ Mask a raster with a list of values.

    Args:
        raster (path | raster | list): The raster(s) to retrieve nodata values from.

        values_to_mask (list): The list of values to mask in the raster(s)
    
    **kwargs:
        include_original_nodata: (bool): If True, the nodata value of the raster(s) will be
        included in the values to mask.

        dst_nodata (float, int, str, None): The target nodata value. If 'infer' the nodata
        value is set based on the input datatype. A list of nodata values can be based matching
        the amount of input rasters. If multiple nodata values should be set, use raster_mask_values.

        out_path (path | list | None): The destination of the changed rasters. If out_paths
        are specified, in_place is automatically set to False. The path can be a folder.

        in_place (bool): Should the rasters be changed in_place or copied?

        prefix (str): Prefix to add the the output if a folder is specified in out_path.

        postfix (str): Postfix to add the the output if a folder is specified in out_path.

    Returns:
        Returns the rasters with nodata removed. If in_place is True a reference to the 
        changed orignal is returned, otherwise a copied memory raster or the path to the
        generated raster is outputted.
    """
    rasters = []
    output_rasters = []
    rasters_metadata = []
    internal_in_place = in_place if out_path is None else False
    internal_dst_nodata = None

    if not isinstance(values_to_mask, list):
        raise ValueError(f"values_to_mask must be a list.")
    
    for value in values_to_mask:
        if not isinstance(value, (int, float)):
            raise ValueError("Values in values_to_mask must be ints or floats")

    if not isinstance(dst_nodata, (float, int, str, list)) and dst_nodata is not None:
        raise ValueError(f"Invalid dst_nodata value. {dst_nodata}")
    
    if isinstance(dst_nodata, str) and dst_nodata != "infer":
        raise ValueError(f"Invalid dst_nodata value. {dst_nodata}")
    
    if isinstance(dst_nodata, list):
        if not isinstance(raster, list) or len(dst_nodata) != len(raster):
            raise ValueError("If dst_nodata is a list, raster must also be a list of equal length.")
        
        for value in dst_nodata:
            if isinstance(value, (float, int, str, None)):
                raise ValueError("Invalid type in dst_nodata list.")
            
            if isinstance(value, str) and value != "infer":
                raise ValueError("If dst_nodata is a string it must be 'infer'")

    if isinstance(raster, (gdal.Dataset, str)):
        rasters.append(raster)
    elif isinstance(raster, list):
        rasters = raster
    else:
        raise ValueError(f"Input raster is invalid: {raster}")

    if isinstance(out_path, list):
        if not isinstance(raster, list):
            raise ValueError("If out_path is a list, the input raster must also be a list.")
        
        if len(out_path) != len(raster):
            raise ValueError("list of out_path and list of input rasters must match in length.")
        
        for path in out_path:
            overwrite_required(path, overwrite)
            output_rasters.append(path)
    elif isinstance(out_path, str):
        if os.path.isdir(out_path):
            for internal_raster in rasters:
                raster_metadata = raster_to_metadata(internal_raster)
                rasters_metadata.append(raster_metadata)

                raster_basename = raster_metadata["basename"]
                raster_ext = raster_metadata["ext"]
                path = f"{out_path}{prefix}{raster_basename}{postfix}{raster_ext}"
                
                # Check if file exists and overwrite is required.
                overwrite_required(path, overwrite)
                output_rasters.append(path)
        else:
            raise ValueError(f"Unable to parse out_path: {out_path}")

    elif out_path != None:
        raise ValueError(f"out_path is invalid: {raster}")

    for index, internal_raster in enumerate(rasters):
        if not is_raster(internal_raster):
            raise ValueError(f"Input raster is invalid: {internal_raster}")
        
        raster_metadata = None
        if len(rasters_metadata) == 0:
            raster_metadata = raster_to_metadata(internal_raster)
            rasters_metadata.append(raster_metadata)
        else:
            raster_metadata = rasters_metadata[index]
        
        if dst_nodata == "infer":
            internal_dst_nodata = gdal_nodata_value_from_type(raster_metadata["dtype_gdal_raw"])
        elif isinstance(dst_nodata, list):
            internal_dst_nodata = dst_nodata[index]
        else:
            internal_dst_nodata = dst_nodata

        mask_values = list(values_to_mask)
        if include_original_nodata:
            if raster_metadata["nodata_value"] is not None:
                mask_values.append(raster_metadata["nodata_value"])

        arr = raster_to_array(internal_raster, filled=True)

        mask = None
        for index, mask_value in enumerate(mask_values):
            if index == 0:
                mask = arr == mask_value
            else:
                mask = mask | arr == mask_value
        
        arr = np.ma.masked_array(arr, mask=mask, fill_value=internal_dst_nodata)

        if internal_in_place:
            for band in range(raster_metadata["bands"]):
                raster_band = internal_raster.GetRasterBand(band + 1)
                raster_band.WriteArray(arr[:, :, band])
                raster_band = None
        else:

            if out_path is None:
                raster_mem = array_to_raster(arr, internal_raster)
                output_rasters.append(raster_mem)
            else:
                path = out_path[index]
                remove_if_overwrite(path, overwrite)

                array_to_raster(arr, internal_raster, out_path=path)
                output_rasters.append(path)

    if internal_in_place:
        return raster

    if isinstance(raster, list):
        return output_rasters
    
    return output_rasters[0]
