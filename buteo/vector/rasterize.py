from osgeo import gdal


def rasterize_vector(
    vector,
    extent,
    raster_size,
    projection,
    all_touch=False,
    optim="raster",
    band=1,
    antialias=False,
):

    # Create destination dataframe
    driver = gdal.GetDriverByName("MEM")

    destination = driver.Create(
        "in_memory_raster",  # Location of the saved raster, ignored if driver is memory.
        int(raster_size[0]),  # Dataframe width in pixels (e.g. 1920px).
        int(raster_size[1]),  # Dataframe height in pixels (e.g. 1280px).
        1,  # The number of bands required.
        gdal.GDT_Byte,  # Datatype of the destination.
    )

    destination.SetGeoTransform(
        (extent[0], raster_size[2], 0, extent[3], 0, -raster_size[3])
    )
    destination.SetProjection(projection)

    # Rasterize and retrieve data
    destination_band = destination.GetRasterBand(band)
    destination_band.Fill(1)

    if antialias is False:
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

        gdal.RasterizeLayer(destination, [1], vector, burn_values=[0], options=options)
