import sys, os
from glob import glob

sys.path.append("../../")
from buteo.earth_observation.s1_mosaic import mosaic_sentinel1
from buteo.earth_observation.s1_preprocess import (
    backscatter,
    backscatter_step1,
    backscatter_step2,
    convert_to_tiff,
)


# folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/tmp/"
# tmp = folder + "tmp/"
# raw = folder + "raw/"
# dst = folder + "dst/"

# for idx, image in enumerate(glob(raw + "*.zip")):
#     try:
#         backscatter_step1(
#             image, tmp + os.path.splitext(os.path.basename(image))[0] + "_step1.dim"
#         )
#     except:
#         print(f"Error with image: {image}")

# for idx, image in enumerate(glob(tmp + "*_step1.dim")):
#     try:
#         backscatter_step2(
#             image,
#             tmp + os.path.splitext(os.path.basename(image))[0] + "_step2.dim",
#         )
#     except:
#         print(f"Error with image: {image}")

# for idx, image in enumerate(glob(tmp + "*_step2.dim")):
#     try:
#         convert_to_tiff(
#             image,
#             dst,
#         )
#     except:
#         print(f"Error with image: {image}")


folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/tanzania_mwanza/"

mosaic_sentinel1(
    folder + "S1/august/",
    folder + "S1/august/dst/",
    folder + "S1/august/tmp/",
    interest_area=folder + "vector/mwanza_extent.gpkg",
    target_projection=32736,
    kernel_size=3,
    overlap=0.00,
    step_size=1.0,
    quantile=0.5,
    max_images=0,
    weighted=True,
    overwrite=False,
    use_tiles=False,
    high_memory=True,
    polarization="VV",
    prefix="",
    postfix="",
)

mosaic_sentinel1(
    folder + "S1/august/",
    folder + "S1/august/dst/",
    folder + "S1/august/tmp/",
    interest_area=folder + "vector/mwanza_extent.gpkg",
    target_projection=32736,
    kernel_size=3,
    overlap=0.00,
    step_size=1.0,
    quantile=0.5,
    max_images=0,
    weighted=True,
    overwrite=False,
    use_tiles=False,
    high_memory=True,
    polarization="VH",
    prefix="",
    postfix="",
)
