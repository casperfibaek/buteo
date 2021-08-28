import sys, os, numpy as np

sys.path.append("../../")
from buteo.machine_learning.patch_extraction import extract_patches
from glob import glob

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/odense_2020/"
outdir = folder + "patches/raw/"
tmpdir = folder + "tmp/"

m10 = [
    folder + "raster/2020_B02_10m.tif",
    folder + "raster/2020_B03_10m.tif",
    folder + "raster/2020_B04_10m.tif",
    folder + "raster/2020_B08_10m.tif",
    folder + "raster/2020_VH.tif",
    folder + "raster/2020_VV.tif",
    folder + "raster/2020_label_area.tif",
    folder + "raster/2020_label_volume.tif",
    folder + "raster/2020_label_people.tif",
    folder + "raster/2021_B02_10m.tif",
    folder + "raster/2021_B03_10m.tif",
    folder + "raster/2021_B04_10m.tif",
    folder + "raster/2021_B08_10m.tif",
    folder + "raster/2021_VH.tif",
    folder + "raster/2021_VV.tif",
    folder + "raster/2021_label_area.tif",
    folder + "raster/2021_label_volume.tif",
    folder + "raster/2021_label_people.tif",
]

m20 = [
    folder + "raster/2020_B05_20m.tif",
    folder + "raster/2020_B06_20m.tif",
    folder + "raster/2020_B07_20m.tif",
    folder + "raster/2020_B8A_20m.tif",
    folder + "raster/2020_B11_20m.tif",
    folder + "raster/2020_B12_20m.tif",
    folder + "raster/2020_validity_mask.tif",
    folder + "raster/2021_B05_20m.tif",
    folder + "raster/2021_B06_20m.tif",
    folder + "raster/2021_B07_20m.tif",
    folder + "raster/2021_B8A_20m.tif",
    folder + "raster/2021_B11_20m.tif",
    folder + "raster/2021_B12_20m.tif",
    folder + "raster/2021_validity_mask.tif",
]

patch_size = 32

half = patch_size // 2
quart = half // 2
eighth = quart // 2

m10_patches = [(half, half)]
m20_patches = [(quart, quart)]

print(f"Processing patches..")

extract_patches(
    m10,
    out_dir=tmpdir,
    prefix="",
    postfix="",
    size=patch_size,
    offsets=m10_patches,
    generate_grid_geom=False,
    generate_zero_offset=False,
    generate_border_patches=False,
    # clip_geom=region,
    verify_output=False,
    verification_samples=100,
    verbose=1,
)

extract_patches(
    m20,
    out_dir=tmpdir,
    prefix="",
    postfix="",
    size=half,
    offsets=m20_patches,
    generate_grid_geom=False,
    generate_zero_offset=False,
    generate_border_patches=False,
    # clip_geom=region,
    verify_output=False,
    verification_samples=100,
    verbose=1,
)

validity_mask = np.load(folder + "tmp/2020_validity_mask.npy")
for img in glob(folder + "tmp/2020_*.npy"):
    name = os.path.splitext(os.path.basename(img))[0]

    layer = np.load(img)

    # at least half must be valid pixels.
    mask = validity_mask.sum(axis=(1, 2, 3)) > (
        (validity_mask.shape[1] * validity_mask.shape[2]) // 2
    )

    np.save(outdir + name + ".npy", layer[mask])

validity_mask = np.load(folder + "tmp/2021_validity_mask.npy")
for img in glob(folder + "tmp/2021_*.npy"):
    name = os.path.splitext(os.path.basename(img))[0]

    layer = np.load(img)

    # at least half must be valid pixels.
    mask = validity_mask.sum(axis=(1, 2, 3)) > (
        (validity_mask.shape[1] * validity_mask.shape[2]) // 2
    )

    np.save(outdir + name + ".npy", layer[mask])
