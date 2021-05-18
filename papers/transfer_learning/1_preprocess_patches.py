import sys

sys.path.append("../../")
from glob import glob
from buteo.machine_learning.patch_extraction import extract_patches

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/"

target = "denmark"

region = folder + f"{target}/"

vector = folder + f"vector/denmark_polygon_border_region_removed.gpkg"

s2_10m = glob(region + "raster/*10m.tif")
s1_VV = glob(region + "raster/*VV.tif")
s1_VH = glob(region + "raster/*VH.tif")

area = region + "raster/area.tif"
volume = region + "raster/volume.tif"
people = region + "raster/people.tif"

images = []

for img in s2_10m:
    images.append(img)

for img in s1_VV:
    images.append(img)

for img in s1_VH:
    images.append(img)

images.append(area)
images.append(volume)
images.append(people)

path_np, path_geom = extract_patches(
    images,
    out_dir=region,
    prefix="",
    postfix="",
    size=128,
    offsets=[(32, 32), (64, 64), (96, 96)],
    generate_grid_geom=True,
    generate_zero_offset=True,
    generate_border_patches=True,
    clip_geom=vector,
    verify_output=True,
    verification_samples=100,
    verbose=1,
)

s2_20m = glob(region + "raster/*20m.tif")
images = s2_20m

path_np, path_geom = extract_patches(
    images,
    out_dir=region,
    prefix="",
    postfix="",
    size=64,
    offsets=[(16, 16), (32, 32), (48, 48)],
    generate_grid_geom=True,
    generate_zero_offset=True,
    generate_border_patches=True,
    clip_geom=vector,
    verify_output=True,
    verification_samples=100,
    verbose=1,
)
