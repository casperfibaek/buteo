from numpy.ma import clip
from buteo.raster.io import internal_raster_to_metadata, open_raster
from buteo.vector.clip import internal_clip_vector
from buteo.vector.io import internal_vector_to_metadata, open_vector
from math import ceil
from osgeo import gdal
from uuid import uuid4


def rasterize_vector(
    vector,
    pixel_size,
    out_path=None,
    extent=None,
    all_touch=False,
    optim="raster",
    band=1,
    fill_value=0,
    nodata_value=None,
    check_memory=True,
    burn_value=1,
    attribute=None,
):
    vector_fn = vector

    if out_path is None:
        raster_fn = f"/vsimem/{str(uuid4())}.tif"
    else:
        raster_fn = out_path

    # Open the data source and read in the extent
    source_ds = open_vector(vector_fn)
    source_meta = internal_vector_to_metadata(vector_fn)
    source_layer = source_ds.GetLayer()
    x_min, x_max, y_min, y_max = source_layer.GetExtent()

    # Create the destination data source
    x_res = int((x_max - x_min) / pixel_size)
    y_res = int((y_max - y_min) / pixel_size)

    if extent is not None:
        extent_vector = internal_vector_to_metadata(extent)
        extent_dict = extent_vector["extent_dict"]
        x_res = int((extent_dict["right"] - extent_dict["left"]) / pixel_size)
        y_res = int((extent_dict["top"] - extent_dict["bottom"]) / pixel_size)
        x_min = extent_dict["left"]
        y_max = extent_dict["top"]

    if check_memory is False:
        gdal.SetConfigOption("CHECK_DISK_FREE_SPACE", "FALSE")

    try:
        target_ds = gdal.GetDriverByName("GTiff").Create(
            raster_fn,
            x_res,
            y_res,
            1,
            gdal.GDT_Byte,
        )
    finally:
        gdal.SetConfigOption("CHECK_DISK_FREE_SPACE", "TRUE")

    if target_ds is None:
        raise Exception("Unable to rasterize.")

    target_ds.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
    target_ds.SetProjection(source_meta["projection"])

    band = target_ds.GetRasterBand(1)
    band.Fill(fill_value)

    if nodata_value is not None:
        band.SetNoDataValue(nodata_value)

    options = []
    if all_touch == True:
        options.append("ALL_TOUCHED=TRUE")
    else:
        options.append("ALL_TOUCHED=FALSE")

    if optim == "raster":
        options.append("OPTIM=RASTER")
    elif optim == "vector":
        options.append("OPTIM=VECTOR")
    else:
        options.append("OPTIM=AUTO")

    if attribute is None:
        gdal.RasterizeLayer(
            target_ds, [1], source_layer, burn_values=[burn_value], options=options
        )
    else:
        options.append(f"ATTRIBUTE={attribute}")
        gdal.RasterizeLayer(target_ds, [1], source_layer, options=options)

    return raster_fn
