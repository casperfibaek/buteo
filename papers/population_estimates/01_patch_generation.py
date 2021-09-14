import sys, os, numpy as np

sys.path.append("../../")
from buteo.machine_learning.patch_extraction import extract_patches
from glob import glob

# folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/uganda_kampala/"
# folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/tanzania_dar/"
# folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/tanzania_kilimanjaro/"
folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/tanzania_mwanza/"
outdir = folder + "patches/raw/"
tmpdir = folder + "tmp/"

m10 = glob(folder + "*_10m.tif")
m20 = glob(folder + "*_20m.tif")

patch_size = 32

half = patch_size // 2
quart = half // 2
eighth = quart // 2

m10_patches = [(quart, quart), (half, half), (half + quart, half + quart)]
m20_patches = [(eighth, eighth), (quart, quart), (quart + eighth, quart + eighth)]

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
    verify_output=False,
    verification_samples=100,
    verbose=1,
)

validity_mask = np.load(folder + "tmp/validation_mask_10m.npy")

valid_threshold = 0.90

for img in glob(folder + "tmp/*.npy"):
    name = os.path.splitext(os.path.basename(img))[0]

    layer = np.load(img)

    mask = validity_mask.sum(axis=(1, 2, 3)) > (
        (validity_mask.shape[1] * validity_mask.shape[2]) * valid_threshold
    )

    np.save(outdir + name + ".npy", layer[mask])
