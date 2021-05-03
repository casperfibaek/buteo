import sys

sys.path.append("C:/Users/caspe/Desktop/buteo/")

import numpy as np
from buteo.machine_learning.patch_extraction import extract_patches

if __name__ == "__main__":
    folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/upsampling/"

    out_dir = folder + "out/"

    high_res = [
        folder + "32UMF_B02_10m.tif",
        folder + "32UMF_B03_10m.tif",
        folder + "32UMF_B04_10m.tif",
        folder + "32UMF_B08_10m.tif",
    ]

    low_res = [
        # folder + "32UMF_B08_20m.tif",
        folder + "32UMF_B11_20m.tif",
        folder + "32UMF_B12_20m.tif",
    ]

    # path_np, path_geom = extract_patches(
    #     high_res,
    #     out_dir=out_dir,
    #     prefix="",
    #     postfix="_patches",
    #     size=128,
    #     offsets=[(32, 32), (64, 64), (96, 96)],
    #     generate_grid_geom=True,
    #     generate_zero_offset=True,
    #     generate_border_patches=True,
    #     # clip_geom=vector,
    #     verify_output=True,
    #     verification_samples=100,
    #     verbose=1,
    # )

    # path_np, path_geom = extract_patches(
    #     low_res,
    #     out_dir=out_dir,
    #     prefix="",
    #     postfix="_patches",
    #     size=64,
    #     offsets=[(16, 16), (32, 32), (48, 48)],
    #     generate_grid_geom=True,
    #     generate_zero_offset=True,
    #     generate_border_patches=True,
    #     # clip_geom=vector,
    #     verify_output=True,
    #     verification_samples=100,
    #     verbose=1,
    # )

    import pdb

    pdb.set_trace()

