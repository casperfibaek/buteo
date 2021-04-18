from buteo.raster.io import internal_raster_to_metadata
from buteo.vector.clip import internal_clip_vector
from osgeo import gdal


def rasterize_vector(
    vector,
    reference,
    out_path=None,
    all_touch=False,
    optim="raster",
    band=1,
):

    # Create destination dataframe
    driver = gdal.GetDriverByName("GTiff")

    metadata = internal_raster_to_metadata(reference)

    destination = driver.Create(
        out_path,  # Location of the saved raster, ignored if driver is memory.
        metadata["width"],  # Dataframe width in pixels (e.g. 1920px).
        metadata["height"],  # Dataframe height in pixels (e.g. 1280px).
        1,  # The number of bands required.
        gdal.GDT_Byte,  # Datatype of the destination.
    )

    destination.SetGeoTransform(metadata["transform"])
    destination.SetProjection(metadata["projection_osr"])

    # Rasterize and retrieve data
    destination_band = destination.GetRasterBand(band)
    destination_band.Fill(0)

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

    clip_vector = internal_clip_vector(vector, reference)

    gdal.RasterizeLayer(destination, [1], clip_vector, burn_values=[1], options=options)

    return out_path
