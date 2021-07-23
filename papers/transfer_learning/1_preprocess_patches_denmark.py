import sys

sys.path.append("../../")
from glob import glob
from buteo.machine_learning.patch_extraction import extract_patches

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/analysis/denmark/"

vector_train = folder + "vector/denmark_border_region_removed_without_roskilde.gpkg"
vector_test = folder + "vector/roskilde.gpkg"

s2_10m = glob(folder + "raster/*_10m.tif")
s2_20m = glob(folder + "raster/*_20m.tif")
s1_VV = glob(folder + "raster/*VV.tif")
s1_VH = glob(folder + "raster/*VH.tif")

area = folder + "raster/area.tif"
volume = folder + "raster/volume.tif"
people = folder + "raster/people.tif"

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

extract_patches(
    images,
    out_dir=folder,
    prefix="",
    postfix="_train",
    size=64,
    offsets=[(32, 32)],
    generate_grid_geom=True,
    generate_zero_offset=True,
    generate_border_patches=True,
    clip_geom=vector_train,
    verify_output=True,
    verification_samples=100,
    verbose=1,
)

extract_patches(
    images,
    out_dir=folder,
    prefix="",
    postfix="_test",
    size=64,
    offsets=[(32, 32)],
    generate_grid_geom=True,
    generate_zero_offset=True,
    generate_border_patches=True,
    clip_geom=vector_test,
    verify_output=True,
    verification_samples=100,
    verbose=1,
)

images = s2_20m

path_np, path_geom = extract_patches(
    images,
    out_dir=folder,
    prefix="",
    postfix="_train",
    size=32,
    offsets=[(16, 16)],
    generate_grid_geom=True,
    generate_zero_offset=True,
    generate_border_patches=True,
    clip_geom=vector_train,
    verify_output=True,
    verification_samples=100,
    verbose=1,
)

path_np, path_geom = extract_patches(
    images,
    out_dir=folder,
    prefix="",
    postfix="_test",
    size=32,
    offsets=[(16, 16)],
    generate_grid_geom=True,
    generate_zero_offset=True,
    generate_border_patches=True,
    clip_geom=vector_test,
    verify_output=True,
    verification_samples=100,
    verbose=1,
)
