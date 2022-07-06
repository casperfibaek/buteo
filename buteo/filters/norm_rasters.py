"""
Normalise rasters as part of the preprocessing pipeline.

TODO:
    - Improve documentation
"""

import sys; sys.path.append("../../") # Path: buteo/filters/norm_rasters.py
import os

import numpy as np

from buteo.raster.io import raster_to_array, array_to_raster


def standardise_filter(in_raster):
    copy = in_raster.copy()
    copy = copy.astype("float32")

    for idx in range(copy.shape[2]):
        m = np.nanmean(copy[:, :, idx])
        s = np.nanstd(copy[:, :, idx])

        copy[:, :, idx] = (copy[:, :, idx] - m) / s

    return copy


def mad_filter(in_raster):
    copy = in_raster.copy()
    copy = copy.astype("float32")

    for idx in range(copy.shape[2]):
        m = np.nanmedian(copy[:, :, idx])
        mad = np.nanmedian(np.abs(copy[:, :, idx] - m))
        s = mad * 1.4826

        copy[:, :, idx] = (copy[:, :, idx] - m) / s

    return copy


def robust_scaler_filter(in_raster, min_q=0.25, max_q=0.75):
    copy = in_raster.copy()
    copy = copy.astype("float32")

    for idx in range(copy.shape[2]):
        q1 = np.nanquantile(copy[:, :, idx], min_q)
        q3 = np.nanquantile(copy[:, :, idx], max_q)
        iqr = q3 - q1

        copy[:, :, idx] = (copy[:, :, idx] - q1) / iqr

    return copy


def normalise_filter(in_raster):
    copy = in_raster.copy()
    copy = copy.astype("float32")

    for idx in range(in_raster.shape[2]):
        mi = np.nanmin(copy[:, :, idx])
        ma = np.nanmax(copy[:, :, idx])
        copy[:, :, idx] = (copy[:, :, idx] - mi) / (ma - mi)

    return copy


def norm_to_range(
    in_raster, min_target, max_target, min_og=-99999, max_og=99999, truncate=True
):

    copy = in_raster.copy()
    copy = copy.astype("float32")

    for idx in range(in_raster.shape[2]):
        if min_og == -9999:
            if isinstance(in_raster, np.ma.MaskedArray):
                min_og = copy[:, :, idx].min()
            else:
                min_og = in_raster.nanmin()

        if max_og == -9999:
            if isinstance(in_raster, np.ma.MaskedArray):
                max_og = copy[:, :, idx].max()
            else:
                max_og = in_raster.nanmax()

        if truncate:
            if isinstance(in_raster, np.ma.MaskedArray):
                copy[:, :, idx] = np.ma.clip(copy[:, :, idx], min_og, max_og)
            else:
                copy[:, :, idx] = np.clip(copy[:, :, idx], min_og, max_og)

        to_range = np.interp(
            copy[:, :, idx], (min_og, max_og), (min_target, max_target)
        )

        if isinstance(in_raster, np.ma.MaskedArray):
            to_range = np.ma.masked_where(copy[:, :, idx].mask, to_range)
            to_range.fill_value = in_raster.fill_value

        copy[:, :, idx] = to_range

    return copy


def norm_rasters(
    in_rasters,
    out_folder,
    method="normalise",
    split_bands=False,
    min_target=0,
    max_target=1,
    min_og=-9999,
    max_og=-9999,
    truncate=True,
    prefix="",
    postfix="",
    overwrite=True,
):
    if not isinstance(in_rasters, list):
        in_rasters = [in_rasters]

    normed_rasters = []
    for in_raster in in_rasters:
        name = os.path.splitext(os.path.basename(in_raster))[0]

        raster = raster_to_array(in_raster)

        if method == "normalise":
            normed = norm_to_range(raster, min_target, max_target, truncate=False)
        elif method == "standardise":
            normed = standardise_filter(raster)
        elif method == "median_absolute_deviation":
            normed = mad_filter(raster)
        elif method == "range":
            normed = norm_to_range(
                raster,
                min_target=min_target,
                max_target=max_target,
                min_og=min_og,
                max_og=max_og,
                truncate=truncate,
            )
        elif method == "robust_quantile":
            normed = robust_scaler_filter(
                raster,
                min_q=0.25,
                max_q=0.75,
            )
        elif method == "robust_98":
            normed = robust_scaler_filter(raster, min_q=0.02, max_q=0.98)
        else:
            raise Exception(f"Method {method} not recognised")

        if split_bands:
            for idx in range(raster.shape[2]):
                band = idx + 1
                raster_name = prefix + name + f"_B{band}" + postfix + ".tif"

                normed_rasters.append(
                    array_to_raster(
                        normed[:, :, idx][..., np.newaxis],
                        reference=in_raster,
                        out_path=out_folder + raster_name,
                        overwrite=overwrite,
                    )
                )
        else:
            raster_name = prefix + name + postfix + ".tif"

            normed_rasters.append(
                array_to_raster(
                    normed,
                    reference=in_raster,
                    out_path=out_folder + raster_name,
                    overwrite=overwrite,
                )
            )

    if isinstance(in_rasters, list):
        return normed_rasters

    return normed_rasters[0]
