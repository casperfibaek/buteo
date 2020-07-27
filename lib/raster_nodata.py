import numpy as np
from lib.raster_io import raster_to_array, array_to_raster


def remove_nodata(raster_or_array, out_raster=None):
    if out_raster is None:
        if isinstance(raster_or_array, np.ndarray):
            return raster_or_array.filled()
        else:
            return raster_to_array(raster_or_array).filled()
    else:
        array_to_raster(
            raster_to_array(raster_or_array).filled(),
            raster_or_array,
            raster,
            dst_nodata=False,
        )
        return 1


def set_nodata(arr, move_current_mask=True, value="max"):
    print(arr)
    # if value is max or min set nodata to dtype min or max

    # if move_current_mask is True;
    #


# Functions to handle nodata
# Fill gaps with nodata
# Set nodata
# Mask data
# Remove nodata
# Invert nodata
# Copy mask

