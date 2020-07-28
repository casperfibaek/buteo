import sys; sys.path.append('..'); sys.path.append('../lib/')
from lib.raster_io import raster_to_array, array_to_raster
from lib.stats_filters import scale_to_range_filter, truncate_filter

folder = "C:\\Users\\caspe\\Desktop\\Paper_2_StruturalDensity\\"

array_to_raster(
    scale_to_range_filter(
        truncate_filter(
            raster_to_array(folder + "b02.tif"),
        0, 2500),
    0, 1).astype('float32'),
    folder + "b02_scaled.tif",
    folder + "b02.tif"
)

array_to_raster(
    scale_to_range_filter(
        truncate_filter(
            raster_to_array(folder + "b03.tif"),
        0, 3000),
    0, 1).astype('float32'),
    folder + "b03_scaled.tif",
    folder + "b03.tif"
)

array_to_raster(
    scale_to_range_filter(
        truncate_filter(
            raster_to_array(folder + "b04.tif"),
        0, 4000),
    0, 1).astype('float32'),
    folder + "b04_scaled.tif",
    folder + "b04.tif"
)

array_to_raster(
    scale_to_range_filter(
        truncate_filter(
            raster_to_array(folder + "b08.tif"),
        0, 5500),
    0, 1).astype('float32'),
    folder + "b08_scaled.tif",
    folder + "b08.tif"
)

array_to_raster(
    scale_to_range_filter(
        truncate_filter(
            raster_to_array(folder + "coh.tif"),
        0, 1),
    0, 1).astype('float32'),
    folder + "coh_scaled.tif",
    folder + "coh.tif"
)

array_to_raster(
    scale_to_range_filter(
        truncate_filter(
            raster_to_array(folder + "bs.tif"),
        0, 3),
    0, 1).astype('float32'),
    folder + "bs_scaled.tif",
    folder + "bs.tif"
)
