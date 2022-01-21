import sys
import os
import numpy as np
from osgeo import ogr, gdal
from uuid import uuid4

sys.path.append("../../")

from buteo.raster.io import (
    raster_to_array,
    array_to_raster,
    is_raster,
    raster_to_metadata,
)
from buteo.raster.align import rasters_are_aligned, align_rasters
from buteo.raster.clip import clip_raster
from buteo.raster.resample import resample_raster
from buteo.vector.io import vector_to_metadata, is_vector, open_vector
from buteo.vector.intersect import intersect_vector
from buteo.vector.attributes import vector_get_fids
from buteo.vector.rasterize import rasterize_vector
from buteo.machine_learning.ml_utils import get_offsets
from buteo.machine_learning.patch_utils import get_overlaps
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
        "force_align": True,
        "output_raster_labels": True,
        "label_geom": None,
        "label_res": 0.2,
        "label_mult": 100,
        "tolerance": 0.0,
        "fill_value": 0,
        "zone_layer_id": 0,
        "align_with_size": 20,
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
                "Rasters we not aligned. Realigning rasters due to force_align=True option."
            )
            raster_list = align_rasters(raster_list)
        else:
            raise ValueError("Rasters in raster_list are not aligned.")

    offsets = get_offsets(tile_size) if options["overlaps"] else [[0, 0]]
    raster_metadata = raster_to_metadata(raster_list[0], create_geometry=True)
    pixel_size = min(raster_metadata["pixel_height"], raster_metadata["pixel_width"])

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
    labels_processed = False

    for idx, raster in enumerate(raster_list):
        name = os.path.splitext(os.path.basename(raster))[0]
        list_extracted = []
        list_masks = []
        list_labels = []

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

            rasterize_vector(
                fid_path,
                pixel_size,
                out_path=valid_path,
                extent=fid_path,
            )
            valid_arr = raster_to_array(valid_path)

            if options["label_geom"] is not None and fid not in processed_fids:
                if not is_vector(options["label_geom"]):
                    raise TypeError(
                        "label geom is invalid. Did you input a valid geometry?"
                    )

                uuid = str(uuid4().int)

                label_clip_path = f"/vsimem/fid_{uuid}_{str(fid)}_clipped.shp"
                label_ras_path = f"/vsimem/fid_{uuid}_{str(fid)}_rasterized.tif"
                label_warp_path = f"/vsimem/fid_{uuid}_{str(fid)}_resampled.tif"

                intersect_vector(
                    options["label_geom"], fid_ds, out_path=label_clip_path
                )

                try:
                    rasterize_vector(
                        label_clip_path,
                        options["label_res"],
                        out_path=label_ras_path,
                        extent=valid_path,
                    )

                except Exception:
                    array_to_raster(
                        np.zeros(valid_arr.shape, dtype="float32"),
                        valid_path,
                        out_path=label_ras_path,
                    )

                resample_raster(
                    label_ras_path,
                    pixel_size,
                    resample_alg="average",
                    out_path=label_warp_path,
                )

                labels_arr = (
                    raster_to_array(label_warp_path) * options["label_mult"]
                ).astype("float32")

                if options["output_raster_labels"]:
                    array_to_raster(
                        labels_arr,
                        label_warp_path,
                        out_path=f"{outdir}{options['prefix']}label_{str(fid)}{options['postfix']}.tif",
                    )

            raster_clip_path = f"/vsimem/raster_{uuid}_{str(idx)}_clipped.tif"

            try:
                clip_raster(
                    raster,
                    valid_path,
                    raster_clip_path,
                    all_touch=False,
                    adjust_bbox=False,
                )
            except Exception as e:
                print(f"Warning: {raster} did not intersect geom with fid: {fid}.")
                print(e)

                if options["label_geom"] is not None:
                    gdal.Unlink(label_clip_path)
                    gdal.Unlink(label_ras_path)
                    gdal.Unlink(label_warp_path)
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

            if options["label_geom"] is not None and not labels_processed:
                labels_masked = np.concatenate(
                    get_overlaps(
                        labels_arr, offsets, tile_size, options["border_check"]
                    )
                )[valid_mask]

            if options["merge_output"]:
                list_extracted.append(arr)
                list_masks.append(valid_masked)

                if options["label_geom"] is not None and not labels_processed:
                    list_labels.append(labels_masked)
            else:
                np.save(
                    f"{outdir}{options['prefix']}{str(fid)}_{name}{options['postfix']}.npy",
                    arr.filled(options["fill_value"]),
                )

                np.save(
                    f"{outdir}{options['prefix']}{str(fid)}_mask_{name}{options['postfix']}.npy",
                    valid_masked.filled(options["fill_value"]),
                )

                if options["label_geom"] is not None and not labels_processed:
                    np.save(
                        f"{outdir}{options['prefix']}{str(fid)}_label_{name}{options['postfix']}.npy",
                        valid_masked.filled(options["fill_value"]),
                    )

            if fid not in processed_fids:
                processed_fids.append(fid)

            processed += 1
            progress(processed, len(fids) * len(raster_list), "processing fids")

            if not options["merge_output"]:
                gdal.Unlink(label_clip_path)
                gdal.Unlink(label_ras_path)
                gdal.Unlink(label_warp_path)
                gdal.Unlink(fid_path)

            gdal.Unlink(valid_path)

        if options["merge_output"]:
            np.save(
                f"{outdir}{options['prefix']}{name}{options['postfix']}.npy",
                np.ma.concatenate(list_extracted).filled(options["fill_value"]),
            )
            np.save(
                f"{outdir}{options['prefix']}mask_{name}{options['postfix']}.npy",
                np.ma.concatenate(list_masks).filled(options["fill_value"]),
            )

            if options["label_geom"] is not None and not labels_processed:
                np.save(
                    f"{outdir}{options['prefix']}label_{name}{options['postfix']}.npy",
                    np.ma.concatenate(list_labels).filled(options["fill_value"]),
                )
                labels_processed = True

    progress(1, 1, "processing fids")

    return 1
