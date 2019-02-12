from rasterUtils.clipRaster import clipRaster

test = clipRaster(
    '../raster/S2B_MSIL2A_20180702T104019_N0208_R008_T32VNJ_20180702T150728.SAFE/GRANULE/L2A_T32VNJ_A006898_20180702T104021/IMG_DATA/R10m/T32VNJ_20180702T104019_B02_10m.jp2',
    cutline='../geometry/roses.geojson',
)

print(test)
