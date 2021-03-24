import sys; sys.path.append('../../')
import os
from collections import Counter
from numpy.core.numeric import Infinity
# from pygeoprocessing import geoprocessing
from buteo.raster.io import raster_to_metadata


def is_aligned(input_rasters, same_extent=False, same_dtype=False):
    if len(input_rasters) == 1:
        print("WARNING: Only one input raster")
        return True

    metas = []

    for raster in input_rasters:
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


def align(
    input_rasters, output_folder, postfix="_aligned", prefix="",
    bounding_box_mode="intersection",
    resample_method_list=None,
    target_pixel_size=None,
    base_vector_path_list=None,
    raster_align_index=None,
    base_projection_wkt_list=None,
    target_projection_wkt=None,
    vector_mask_options=None,
    gdal_warp_options=None,
    raster_driver_creation_tuple=('GTIFF', ('TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256')),
):
    if resample_method_list is None:
        resample_method_list = ["near"] * len(input_rasters)
    
    if base_projection_wkt_list is None:
        base_projection_wkt_list = []
        for in_raster in input_rasters:
            in_rast = raster_to_metadata(in_raster)
            base_projection_wkt_list.append(in_rast["projection"])

    if target_projection_wkt is None:
        counts = None
        target_projection_wkt = None

        if base_projection_wkt_list is not None:
            counts = Counter(base_projection_wkt_list)
            target_projection_wkt = counts.most_common(1)[0][0]
        else:
            projections = []
            for in_raster in input_rasters:
                in_rast = raster_to_metadata(in_raster)
                projections.append(in_rast["projection"])

            counts = Counter(base_projection_wkt_list)
            target_projection_wkt = counts.most_common(1)[0][0]

    if target_pixel_size is None:
        target_pixel_size = Infinity

        for in_raster in input_rasters:
            meta = raster_to_metadata(in_raster)
            pixel_size = min(meta["pixel_width"], meta["pixel_height"])
            if pixel_size < target_pixel_size:
                target_pixel_size = (pixel_size, pixel_size)
    
    output_rasters = []
    for raster in input_rasters:
        base = os.path.basename(raster)
        basename = os.path.splitext(base)[0]
        extension = os.path.splitext(base)[1]
        out_path = os.path.join(output_folder + f"{prefix}{basename}{postfix}{extension}")

        output_rasters.append(out_path)

    geoprocessing.align_and_resize_raster_stack(
        input_rasters,
        output_rasters,
        resample_method_list,
        target_pixel_size,
        bounding_box_mode,
        base_vector_path_list=base_vector_path_list,
        raster_align_index=raster_align_index,
        base_projection_wkt_list=base_projection_wkt_list,
        target_projection_wkt=target_projection_wkt,
        vector_mask_options=vector_mask_options,
        gdal_warp_options=gdal_warp_options,
        raster_driver_creation_tuple=raster_driver_creation_tuple,
    )

    return 1
