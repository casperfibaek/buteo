import sys

sys.path.append("../../")
from glob import glob
from buteo.machine_learning.patch_extraction import extract_patches

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/ghana/raster/"

s2_10m = glob(folder + "west/*10m.tif")
s2_20m = glob(folder + "west/*20m.tif")
s1_VV = glob(folder + "west/*VV.tif")
s1_VH = glob(folder + "west/*VH.tif")

area = folder + "west/area.tif"
volume = folder + "west/volume.tif"
people = folder + "west/people.tif"

images = []

for img in s2_10m:
    images.append(img)

for img in s1_VV:
    images.append(img)

for img in s1_VH:
    images.append(img)

images.append(area)

path_np, path_geom = extract_patches(
    images,
    out_dir=folder + "patches/",
    prefix="",
    postfix="",
    size=64,
    # offsets=[(32, 32), (64, 64), (96, 96)],
    offsets=[(16, 16), (32, 32), (48, 48)],
    # offsets=[(32, 32)],
    generate_grid_geom=True,
    generate_zero_offset=True,
    generate_border_patches=True,
    # clip_geom=vector,
    verify_output=True,
    verification_samples=100,
    verbose=1,
)

images = s2_20m

path_np, path_geom = extract_patches(
    images,
    out_dir=folder + "patches/",
    prefix="",
    postfix="",
    size=32,
    # offsets=[(16, 16), (32, 32), (48, 48)],
    offsets=[(8, 8), (16, 16), (24, 24)],
    # offsets=[(16, 16)],
    generate_grid_geom=True,
    generate_zero_offset=True,
    generate_border_patches=True,
    # clip_geom=vector,
    verify_output=True,
    verification_samples=100,
    verbose=1,
)
