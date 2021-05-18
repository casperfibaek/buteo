import sys

sys.path.append("../../")
import numpy as np
from buteo.vector.attributes import vector_get_attribute_table


folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/"
bornholm = folder + "bornholm/"
denmark = folder + "denmark/"
tmp = folder + "tmp/"
dst = folder + "machine_learning_data/"

# ------- DENMARK ATTRIBUTES -----------
denmark_patches = denmark + "denmark_patches.gpkg"

denmark_attr = vector_get_attribute_table(denmark_patches)
denmark_attr["municipality"] = denmark_attr["municipality"].fillna(0)
denmark_attr = denmark_attr.astype({"municipality": int})
denmark_attr["fid"] -= 1

# ------- BORNHOLM ATTRIBUTES -----------
bornholm_patches = bornholm + "bornholm_patches.gpkg"

bornholm_attr = vector_get_attribute_table(bornholm_patches)
bornholm_attr["municipality"] = bornholm_attr["municipality"].fillna(0)
bornholm_attr = bornholm_attr.astype({"municipality": int})

# ------- COMMON PREPROCESSING ------------
def get_image_paths(folder):
    return [
        ["B02", folder + "2020_B02_10m.npy", folder + "2021_B02_10m.npy"],
        ["B03", folder + "2020_B03_10m.npy", folder + "2021_B03_10m.npy"],
        ["B04", folder + "2020_B04_10m.npy", folder + "2021_B04_10m.npy"],
        ["B08", folder + "2020_B08_10m.npy", folder + "2021_B08_10m.npy"],
        ["B11", folder + "2020_B11_20m.npy", folder + "2021_B11_20m.npy"],
        ["B12", folder + "2020_B12_20m.npy", folder + "2021_B12_20m.npy"],
        ["VH", folder + "2020_VH.npy", folder + "2021_VH.npy"],
        ["VV", folder + "2020_VV.npy", folder + "2021_VV.npy"],
        ["AREA", folder + "area.npy"],
        ["VOLUME", folder + "volume.npy"],
        ["PEOPLE", folder + "people.npy"],
    ]


# ------- DENMARK PREPROCESSING -----------
denmark_municipalities = denmark_attr["municipality"].unique()
denmark_paths = get_image_paths(denmark)

for idx in range(len(denmark_paths)):
    img = denmark_paths[idx]
    name = img[0]

    if len(img) == 3:
        path_2020 = img[1]
        path_2021 = img[2]

        img1 = np.load(path_2020)
        img2 = np.load(path_2021)
    else:
        path = img[1]
        img1 = np.load(path)
        img2 = img1

    for muni in denmark_municipalities:
        indices = denmark_attr[denmark_attr["municipality"] == muni]["fid"].values
        merged = np.concatenate([img1[indices], img2[indices]])

        np.save(tmp + f"{muni}_{name}.npy", merged)

# ------- BORNHOLM PREPROCESSING -----------
bornholm_municipalities = bornholm_attr["municipality"].unique()
bornholm_paths = get_image_paths(bornholm)

for idx in range(len(bornholm_paths)):
    img = bornholm_paths[idx]
    name = img[0]

    if len(img) == 3:
        path_2020 = img[1]
        path_2021 = img[2]

        img1 = np.load(path_2020)
        img2 = np.load(path_2021)
    else:
        path = img[1]
        img1 = np.load(path)
        img2 = img1

    for muni in bornholm_municipalities:
        indices = bornholm_attr[bornholm_attr["municipality"] == muni]["fid"].values
        merged = np.concatenate([img1[indices], img2[indices]])

        np.save(tmp + f"{muni}_{name}.npy", merged)

municipalities = np.concatenate([denmark_municipalities, bornholm_municipalities])
for muni in municipalities:
    images_10m = np.stack(
        [
            np.load(tmp + f"{muni}_B02.npy"),
            np.load(tmp + f"{muni}_B03.npy"),
            np.load(tmp + f"{muni}_B04.npy"),
            np.load(tmp + f"{muni}_B08.npy"),
        ],
        axis=3,
    )[:, :, :, :, 0]

    images_20m = np.stack(
        [
            np.load(tmp + f"{muni}_B11.npy"),
            np.load(tmp + f"{muni}_B12.npy"),
        ],
        axis=3,
    )[:, :, :, :, 0]

    images_sar = np.stack(
        [
            np.load(tmp + f"{muni}_VH.npy"),
            np.load(tmp + f"{muni}_VV.npy"),
        ],
        axis=3,
    )[:, :, :, :, 0]

    images_labels = np.stack(
        [
            np.load(tmp + f"{muni}_AREA.npy"),
            np.load(tmp + f"{muni}_VOLUME.npy"),
            np.load(tmp + f"{muni}_PEOPLE.npy"),
        ],
        axis=3,
    )[:, :, :, :, 0]

    np.save(dst + f"{muni}_RGBN.npy", images_10m)
    np.save(dst + f"{muni}_SWIR.npy", images_20m)
    np.save(dst + f"{muni}_SAR.npy", images_sar)
    np.save(dst + f"{muni}_LABELS.npy", images_labels)


np.save(dst + "municipalities.npy", municipalities[municipalities != 0])
