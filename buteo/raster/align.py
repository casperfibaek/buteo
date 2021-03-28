import sys; sys.path.append('../../')
from typing import Union, List
from osgeo import gdal, ogr, osr
import numpy as np
from buteo.raster.io import (
    raster_to_metadata,
)
from buteo.raster.reproject import reproject_raster
from buteo.utils import folder_exists, overwrite_required, remove_if_overwrite
from buteo.gdal_utils import (
    parse_projection,
    raster_size_from_list,
    is_raster,
    path_to_driver,
    default_options,
    gdal_nodata_value_from_type,
    translate_resample_method,
)


def is_aligned(
    rasters: list,
    same_extent: bool=False,
    same_dtype: bool=False,
) -> bool:
    if len(rasters) == 1:
        return True

    metas = []

    for raster in rasters:
        metas.append(raster_to_metadata(raster))

    base = {}

    for index, meta in enumerate(metas):
        if index == 0:
            base["projection"] = meta["projection"]
            base["pixel_width"] = meta["pixel_width"]
            base["pixel_height"] = meta["pixel_height"]
            
            base["transform"] = meta["transform"]
            base["height"] = meta["height"]
            base["width"] = meta["width"]
            base["dtype"] = meta["nodata_value"]
        else:
            if meta["projection"] != base["projection"]:
                return False
            if meta["pixel_width"] != base["pixel_width"]:
                return False
            if meta["pixel_height"] != base["pixel_height"]:
                return False
            
            if same_extent:
                if meta["transform"] != base["transform"]:
                    return False
                if meta["height"] != base["height"]:
                    return False
                if meta["width"] != base["width"]:
                    return False
            
            if same_dtype:
                if meta["dtype"] != base["dtype"]:
                    return False

    return True


def align_rasters(
    rasters: list,
    output: Union[list, str, None]=None,
    master: Union[gdal.Dataset, str, None]=None,
    bounding_box: Union[str, gdal.Dataset, list, tuple]="intersection",
    resample_alg: str='nearest',
    target_size: Union[tuple, int, float, str, gdal.Dataset, None]=None,
    target_in_pixels: bool=False,
    projection: Union[int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference, None]=None,
    overwrite: bool=True,
    creation_options: list=[],
    src_nodata: Union[str, int, float]="infer",
    dst_nodata: Union[str, int, float]="infer",
    postfix: str="_aligned",
    prefix: str="",
) -> list:
    if isinstance(output, list):
        if len(output) != len(rasters):
            raise ValueError("If output is a list of paths, it must have the same length as rasters")

    metadata: List[dict] = []
    master_metadata = None

    target_projection = None
    target_bounds = None
    x_res = None
    y_res = None
    x_pixels = None
    y_pixels = None

    reprojected_rasters = []

    output_names = []
    if isinstance(output, list):
        output_names = output
    elif isinstance(output, str):
        if not folder_exists(output):
            raise ValueError("output folder does not exists.")

    used_projections = []
    for index, raster in enumerate(rasters):
        raster_metadata = raster_to_metadata(raster)
        used_projections.append(raster_metadata["projection"])

        if isinstance(output, str):
            basename = raster_metadata["basename"]
            ext = raster_metadata["ext"]
            output_names.append(f"{output}{prefix}{basename}{postfix}{ext}")

        metadata.append(raster_metadata)
    
    # throws an error if the file exists and overwrite is False.
    for output_name in output_names:
        overwrite_required(output_name, overwrite)

    if master is not None:
        master_metadata = raster_to_metadata(master)

        target_projection = master_metadata["projection_osr"]
        x_min, y_max, x_max, y_min = master_metadata["extent"]
        target_bounds = (x_min, y_min, x_max, y_max)
        x_res = master_metadata["width"]
        y_res = master_metadata["height"]
        target_in_pixels = False

    if projection is not None:
        target_projection = parse_projection(projection)
    elif target_projection is None:
        projection_counter = {}
        for proj in used_projections:
            if proj in projection_counter:
                projection_counter[proj] += 1
            else:
                projection_counter[proj] = 1
        most_common_projection = sorted(projection_counter, key=projection_counter.get, reverse=True)
        target_projection = parse_projection(most_common_projection[0])
        
    if target_size is not None:
        if isinstance(target_size, gdal.Dataset) or isinstance(target_size, str):
            reprojected_target_size = reproject_raster(target_size, target_projection)
            target_size_raster = raster_to_metadata(reprojected_target_size)
            x_res = target_size_raster["width"]
            y_res = target_size_raster["height"]
        else:
            x_res, y_res, x_pixels, y_pixels = raster_size_from_list(target_size, target_in_pixels)
    elif x_res is None and y_res is None and x_pixels is None and y_pixels is None:
        x_res_arr = np.empty(len(rasters), dtype="float32")
        y_res_arr = np.empty(len(rasters), dtype="float32")
        for index, raster in enumerate(rasters):
            reprojected = reproject_raster(raster, target_projection)
            target_size_raster = raster_to_metadata(reprojected)
            x_res_arr[index] = target_size_raster["pixel_width"]
            y_res_arr[index] = target_size_raster["pixel_height"]
            reprojected_rasters.append(reprojected)

        x_res = np.median(x_res_arr)
        y_res = np.median(y_res_arr)

    if target_bounds is None:
        if isinstance(bounding_box, list) or isinstance(bounding_box, tuple):
            if len(bounding_box) != 4:
                raise ValueError("bounding_box as a list/tuple must have 4 values.")
            target_bounds = bounding_box
        elif is_raster(bounding_box):
            reprojected_bbox = raster_to_metadata(reproject_raster(bounding_box, target_projection))
            x_min, y_max, x_max, y_min = reprojected_bbox["extent"]
            target_bounds = (x_min, y_min, x_max, y_max)
        elif isinstance(bounding_box, str):
            if bounding_box == "intersection" or bounding_box == "union":
                extents = []

                if len(reprojected_rasters) != len(rasters):
                    reprojected_rasters = []

                    for index, raster in enumerate(rasters):
                        raster_metadata = metadata[index]
                        if raster_metadata["projection_osr"].IsSame(target_projection):
                            reprojected_rasters.append(raster)
                        else:
                            reprojected = reproject_raster(raster, target_projection)
                            reprojected_rasters.append(reprojected)

                for reprojected_raster in reprojected_rasters:
                    reprojected_raster_metadata = raster_to_metadata(reprojected_raster)
                    extents.append(reprojected_raster_metadata["extent"])

                x_min, y_max, x_max, y_min = extents[0]

                for index, extent in enumerate(extents):
                    if index == 0:
                        continue

                    if bounding_box == "intersection":
                        if extent[0] > x_min: x_min = extent[0]
                        if extent[1] < y_max: y_max = extent[1]
                        if extent[2] < x_max: x_max = extent[2]
                        if extent[3] > y_min: y_min = extent[3]
                    
                    elif bounding_box == "union":
                        if extent[0] < x_min: x_min = extent[0]
                        if extent[1] > y_max: y_max = extent[1]
                        if extent[2] > x_max: x_max = extent[2]
                        if extent[3] < y_min: y_min = extent[3]

                target_bounds = (x_min, y_min, x_max, y_max)

            else:
                raise ValueError(f"Unable to parse or infer target_bounds: {target_bounds}")
        else:
            raise ValueError(f"Unable to parse or infer target_bounds: {target_bounds}")

    if len(reprojected_rasters) != len(rasters):
        reprojected_rasters = []

        for index, raster in enumerate(rasters):
            raster_metadata = metadata[index]
            if raster_metadata["projection_osr"].IsSame(target_projection):
                reprojected_rasters.append(raster)
            else:
                reprojected = reproject_raster(raster, target_projection)
                reprojected_rasters.append(reprojected)

    if target_projection is None or target_bounds is None:
        raise Exception("Error while preparing the target projection or bounds.")
    
    if x_res is None and y_res is None and x_pixels is None and y_pixels is None:
        raise Exception("Error while preparing the target pixel size.")

    return_list = []
    for index, raster in enumerate(reprojected_rasters):
        raster_metadata = raster_to_metadata(raster)
        out_name = None
        out_format = None
        out_creation_options = None
        if output is None:
            out_name = raster_metadata["name"]
            out_format = "MEM"
            out_creation_options = []
        else:
            out_name = output_names[index]
            out_format = path_to_driver(out_name)
            out_creation_options = default_options(creation_options)
    
        # nodata
        out_src_nodata = None
        out_dst_nodata = None
        if src_nodata == "infer":
            out_src_nodata = raster_metadata["nodata_value"]

            if out_src_nodata is None:
                out_src_nodata = gdal_nodata_value_from_type(raster_metadata["dtype_gdal_raw"])

        elif src_nodata == None:
            out_src_nodata = None
        else:
            out_src_nodata = src_nodata

        if dst_nodata == "infer":
            out_dst_nodata = out_src_nodata
        elif src_nodata == None:
            out_dst_nodata = None
        else:
            out_dst_nodata = dst_nodata


        # Removes file if it exists and overwrite is True.
        remove_if_overwrite(out_name, overwrite)

        warped = gdal.Warp(
            out_name,
            raster,
            xRes=x_res,
            yRes=y_res,
            width=x_pixels,
            height=y_pixels,
            outputBounds=target_bounds,
            format=out_format,
            resampleAlg=translate_resample_method(resample_alg),
            creationOptions=out_creation_options,
            srcNodata=out_src_nodata,
            dstNodata=out_dst_nodata,
            targetAlignedPixels=False,
            cropToCutline=False,
            multithread=True,
        )

        if output is not None:
            warped = None
            return_list.append(out_name)
        else:
            return_list.append(warped)
    
    return return_list
