from glob import glob


folder = "C:/Users/caspe/Desktop/test_area/tmp2/"
master = "C:/Users/caspe/Desktop/test_area/S2_mosaic/B04_10m.tif"
vv_paths = sort_rasters(glob(folder + "*Gamma0_VV*.tif"))
vh_paths = sort_rasters(glob(folder + "*Gamma0_VH*.tif"))

out_dir = folder + "out/"
tmp_dir = folder + "tmp/"

mosaic_s1(
    vv_paths,
    out_dir + "VV_10m.tif",
    tmp_dir,
    "C:/Users/caspe/Desktop/test_area/S2_mosaic/B04_10m_1.tif",
    chunks=5,
    skip_completed=True,
)

mosaic_s1(
    vh_paths,
    out_dir + "VH_10m.tif",
    tmp_dir,
    "C:/Users/caspe/Desktop/test_area/S2_mosaic/B04_10m_1.tif",
    chunks=5,
    skip_completed=True,
)
