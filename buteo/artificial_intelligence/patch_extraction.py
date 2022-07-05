"""
This module generates patches/tiles from a raster.

TODO:
    - Improve documentation
    - Explain options
"""

import sys; sys.path.append("../../") # Path: buteo/artificial_intelligence/patch_extraction.py
import os
from uuid import uuid4

import numpy as np
from osgeo import ogr, gdal


from buteo.raster.io import (
    raster_to_array,
    array_to_raster,
    is_raster,
    raster_to_metadata,
    stack_rasters_vrt,
)
from buteo.raster.align import rasters_are_aligned, align_rasters
from buteo.raster.clip import clip_raster
from buteo.raster.resample import resample_raster
from buteo.vector.io import vector_to_metadata, is_vector, open_vector
from buteo.vector.attributes import vector_get_fids
from buteo.vector.rasterize import rasterize_vector
from buteo.artificial_intelligence.ml_utils import get_offsets
from buteo.artificial_intelligence.patch_utils import get_overlaps
from buteo.utils import progress


def extract_patches(
    raster_list,
    outdir,
    tile_size=32,
    zones=None,
    options=None,
):
    """
    Generate patches for machine learning from rasters
    """
    base_options = {
        "overlaps": True,
        "border_check": True,
        "merge_output": True,
        "force_align": False,
        "output_zone_masks": False,
        "apply_mask": None,
        "mask_reference": None,
        "tolerance": 0.0,
        "fill_value": 0,
        "zone_layer_id": 0,
        "prefix": "",
        "postfix": "",
    }

    if options is None:
        options = base_options
    else:
        for key in options:
            if key not in base_options:
                raise ValueError(f"Invalid option: {key}")
            base_options[key] = options[key]
        options = base_options


    if zones is not None and not is_vector(zones):
        raise TypeError("Clip geom is invalid. Did you input a valid geometry?")

    if not isinstance(raster_list, list):
        raster_list = [raster_list]

    for raster in raster_list:
        if not is_raster(raster):
            raise TypeError("raster_list is not a list of rasters.")

    if not os.path.isdir(outdir):
        raise ValueError(
            "Outdir does not exist. Please create before running the function."
        )

    if not rasters_are_aligned(raster_list, same_extent=True):
        if options["force_align"]:
            print(
                "Rasters were not aligned. Realigning rasters due to force_align=True option."
            )
            raster_list = align_rasters(raster_list, postfix="")
        else:
            raise ValueError("Rasters in raster_list are not aligned.")

    offsets = get_offsets(tile_size) if options["overlaps"] else [[0, 0]]
    raster_metadata = raster_to_metadata(raster_list[0], create_geometry=True)

    if zones is None:
        zones = raster_metadata["extent_datasource_path"]

    zones_meta = vector_to_metadata(zones)

    mem_driver = ogr.GetDriverByName("ESRI Shapefile")

    if zones_meta["layer_count"] == 0:
        raise ValueError("Vector contains no layers.")

    zones_layer_meta = zones_meta["layers"][options["zone_layer_id"]]

    if zones_layer_meta["geom_type"] not in ["Multi Polygon", "Polygon"]:
        raise ValueError("clip geom is not Polygon or Multi Polygon.")

    zones_ogr = open_vector(zones)
    zones_layer = zones_ogr.GetLayer(options["zone_layer_id"])
    feature_defn = zones_layer.GetLayerDefn()
    fids = vector_get_fids(zones_ogr, options["zone_layer_id"])

    progress(0, len(fids) * len(raster_list), "processing fids")
    processed_fids = []
    processed = 0

    outputs = []

    for idx, raster in enumerate(raster_list):
        name = os.path.splitext(os.path.basename(raster))[0]
        list_extracted = []
        list_masks = []

        for fid in fids:
            feature = zones_layer.GetFeature(fid)
            geom = feature.GetGeometryRef()
            fid_path = f"/vsimem/fid_mem_{uuid4().int}_{str(fid)}.shp"
            fid_ds = mem_driver.CreateDataSource(fid_path)
            fid_ds_lyr = fid_ds.CreateLayer(
                "fid_layer",
                geom_type=ogr.wkbPolygon,
                srs=zones_layer_meta["projection_osr"],
            )
            copied_feature = ogr.Feature(feature_defn)
            copied_feature.SetGeometry(geom)
            fid_ds_lyr.CreateFeature(copied_feature)

            fid_ds.FlushCache()
            fid_ds.SyncToDisk()

            valid_path = f"/vsimem/{options['prefix']}validmask_{str(fid)}{options['postfix']}.tif"

            if options["mask_reference"] is not None:
                extent = clip_raster(
                    options["mask_reference"],
                    clip_geom=fid_path,
                    adjust_bbox=True,
                    all_touch=False,
                    to_extent=True,
                )

                tmp_rasterized_vector = rasterize_vector(
                    fid_path,
                    options["mask_reference"],
                    extent=extent,
                )

                resample_raster(
                    tmp_rasterized_vector,
                    target_size=raster,
                    resample_alg="nearest",
                    out_path=valid_path,
                    postfix="",
                )

                gdal.Unlink(tmp_rasterized_vector)
            else:
                extent = clip_raster(
                    raster,
                    clip_geom=fid_path,
                    adjust_bbox=True,
                    all_touch=False,
                    to_extent=True,
                )

                rasterize_vector(
                    fid_path,
                    (raster_metadata["pixel_width"], raster_metadata["pixel_height"]),
                    out_path=valid_path,
                    extent=extent,
                )           

            gdal.Unlink(extent)
            valid_arr = raster_to_array(valid_path)

            uuid = str(uuid4().int)

            raster_clip_path = f"/vsimem/raster_{uuid}_{str(idx)}_clipped.tif"

            try:
                clip_raster(
                    raster,
                    clip_geom=valid_path,
                    out_path=raster_clip_path,
                    all_touch=False,
                    adjust_bbox=False,
                )
            except Exception as error_message:
                print(f"Warning: {raster} did not intersect geom with fid: {fid}.")
                print(error_message)

                gdal.Unlink(fid_path)

                continue

            arr = raster_to_array(raster_clip_path)

            if arr.shape[:2] != valid_arr.shape[:2]:
                raise Exception(
                    f"Error while matching array shapes. Raster: {arr.shape}, Valid: {valid_arr.shape}"
                )

            arr_offsets = get_overlaps(arr, offsets, tile_size, options["border_check"])

            arr = np.concatenate(arr_offsets)
            valid_offsets = np.concatenate(
                get_overlaps(valid_arr, offsets, tile_size, options["border_check"])
            )

            valid_mask = (
                (1 - (valid_offsets.sum(axis=(1, 2)) / (tile_size * tile_size)))
                <= options["tolerance"]
            )[:, 0]

            arr = arr[valid_mask]
            valid_masked = valid_offsets[valid_mask]

            if options["merge_output"]:
                list_extracted.append(arr)
                list_masks.append(valid_masked)

            else:
                out_path = f"{outdir}{options['prefix']}{str(fid)}_{name}{options['postfix']}.npy"
                np.save(out_path, arr.filled(options["fill_value"]))

                outputs.append(out_path)

                if options["output_zone_masks"]:
                    np.save(
                        f"{outdir}{options['prefix']}{str(fid)}_mask_{name}{options['postfix']}.npy",
                        valid_masked.filled(options["fill_value"]),
                    )

            if fid not in processed_fids:
                processed_fids.append(fid)

            processed += 1
            progress(processed, len(fids) * len(raster_list), "processing fids")

            if not options["merge_output"]:
                gdal.Unlink(fid_path)

            gdal.Unlink(valid_path)

        if options["merge_output"]:

            out_arr = np.ma.concatenate(list_extracted).filled(options["fill_value"])
            out_path = f"{outdir}{options['prefix']}{name}{options['postfix']}.npy"

            if options["apply_mask"] is None:
                apply_mask = np.ones(out_arr.shape[0], dtype="bool")
            else:
                apply_mask = options["apply_mask"]

            np.save(out_path, out_arr[apply_mask])
            outputs.append(out_path)

            if options["output_zone_masks"]:
                np.save(
                    f"{outdir}{options['prefix']}mask_{name}{options['postfix']}.npy",
                    np.ma.concatenate(list_masks).filled(options["fill_value"])[apply_mask],
                )

    return outputs


def rasterize_labels(
    geom,
    reference,
    *,
    class_attrib=None,
    out_path=None,
    resample_from=None,
    resample_to=None,
    resample_alg="average",
    resample_scale=None,
    align=True,
    dtype="float32",
    ras_dtype="uint8",
):
    if not is_vector(geom):
        raise TypeError(
            "label geom is invalid. Did you input a valid geometry?"
        )

    if resample_from is None and resample_to is None:
        rasterized = rasterize_vector(
            geom,
            reference,
            extent=reference,
            attribute=class_attrib,
            dtype=ras_dtype,
        )
    elif resample_from is not None and resample_to is None:
        rasterized_01 = rasterize_vector(
            geom,
            resample_from,
            extent=reference,
            attribute=class_attrib,
            dtype=ras_dtype,
        )
        rasterized = resample_raster(
            rasterized_01,
            reference,
            resample_alg=resample_alg,
            dtype=dtype,
            dst_nodata=None,
        )
        gdal.Unlink(rasterized_01)
    elif resample_from is None and resample_to is not None:
        rasterized = rasterize_vector(
            geom,
            resample_to,
            extent=reference,
            attribute=class_attrib,
            dtype=ras_dtype,
        )
    elif resample_from is not None and resample_to is not None:
        rasterized_01 = rasterize_vector(
            geom,
            resample_from,
            extent=reference,
            attribute=class_attrib,
            dtype=ras_dtype,
        )
        rasterized = resample_raster(
            rasterized_01,
            resample_to,
            resample_alg=resample_alg,
            dtype=dtype,
            dst_nodata=None,
        )
        gdal.Unlink(rasterized_01)

    if align:
        aligned = align_rasters(rasterized, master=reference, dst_nodata=None)
        gdal.Unlink(rasterized)
        rasterized = aligned

    arr = raster_to_array(rasterized)
    if isinstance(arr, np.ma.MaskedArray):
        arr.fill_value = 0
        arr = arr.filled(0)

    if resample_scale is not None:
        return array_to_raster(arr * resample_scale, reference=reference, out_path=out_path, set_nodata=None)
    else:
        return array_to_raster(arr, reference=reference, out_path=out_path, set_nodata=None)


def create_mask(geom, reference, out_path=None):
    if not is_vector(geom):
        raise TypeError(
            "label geom is invalid. Did you input a valid geometry?"
        )

    mask = rasterize_vector(
        geom,
        reference,
        out_path=out_path,
        extent=reference,
        attribute=None,
        dtype="uint8",
    )

    return mask


def create_labels(
    geom,
    reference, *,
    out_path=None,
    tmp_folder=None,
    grid=None,
    resample_from=0.2,
    resample_scale=100.0,
    round_label=None,
):
    label_stack = []

    cells = [geom] if grid is None else grid

    for cell in cells:
        name = os.path.splitext(os.path.basename(cell))[0]
        bounds = clip_raster(
            reference,
            cell,
            to_extent=True,
            all_touch=False,
            adjust_bbox=True,
        )

        labels = rasterize_labels(
            geom,
            bounds,
            resample_from=resample_from,
            resample_scale=resample_scale,
        )
        labels_set = array_to_raster(raster_to_array(labels), reference=labels, out_path=tmp_folder + f"{name}_labels_10m.tif", set_nodata=None)
        label_stack.append(labels_set)
        gdal.Unlink(labels)
        gdal.Unlink(bounds)

    stacked = stack_rasters_vrt(label_stack, tmp_folder + "labels_10m.vrt", seperate=False)

    aligned = align_rasters(
        stacked,
        master=reference,
        dst_nodata=None,
    )

    aligned_arr = raster_to_array(aligned)

    gdal.Unlink(stacked)
    gdal.Unlink(aligned)

    if isinstance(aligned_arr, np.ma.MaskedArray):
        aligned_arr.fill_value = 0
        aligned_arr = aligned_arr.filled(0)
    
    if round_label is not None:
        aligned_arr = np.round(aligned_arr, round_label)

    array_to_raster(
        aligned_arr,
        reference=reference,
        out_path=out_path,
        set_nodata=None,
    )

    # clean
    for tmp_label in label_stack:
        try:
            os.remove(tmp_label)
        except:
            pass

    try:
        os.remove(stacked)
    except:
        pass

    return out_path