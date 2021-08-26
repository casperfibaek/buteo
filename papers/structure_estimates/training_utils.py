import numpy as np


def get_layer(folder, name, prefix="", tile_size="64x64"):

    if name == "area":
        return np.load(folder + f"patches/{tile_size}/{prefix}label_area.npy")
    elif name == "volume":
        return np.load(folder + f"patches/{tile_size}/{prefix}label_volume.npy")
    elif name == "people":
        return np.load(folder + f"patches/{tile_size}/{prefix}label_people.npy")
    elif name == "RGB":
        return np.load(folder + f"patches/{tile_size}/{prefix}RGB.npy")
    elif name == "RGBN":
        return np.load(folder + f"patches/{tile_size}/{prefix}RGBN.npy")
    elif name == "VVa":
        return np.load(folder + f"patches/{tile_size}/{prefix}vv_asc_v2.npy")
    elif name == "VVa_VHa":
        return np.concatenate(
            [
                np.load(folder + f"patches/{tile_size}/{prefix}vv_asc_v2.npy"),
                np.load(folder + f"patches/{tile_size}/{prefix}vh_asc_v2.npy"),
            ],
            axis=3,
        )
    elif name == "VVa_VVd":
        return np.concatenate(
            [
                np.load(folder + f"patches/{tile_size}/{prefix}vv_asc.npy"),
                np.load(folder + f"patches/{tile_size}/{prefix}vv_desc.npy"),
            ],
            axis=3,
        )
    elif name == "VVa_COHa":
        return np.concatenate(
            [
                np.load(folder + f"patches/{tile_size}/{prefix}vv_asc.npy"),
                np.load(folder + f"patches/{tile_size}/{prefix}coh_asc.npy"),
            ],
            axis=3,
        )
    elif name == "VVa_VVd_COHa_COHd":
        return np.concatenate(
            [
                np.load(folder + f"patches/{tile_size}/{prefix}vv_asc.npy"),
                np.load(folder + f"patches/{tile_size}/{prefix}vv_desc.npy"),
                np.load(folder + f"patches/{tile_size}/{prefix}coh_asc.npy"),
                np.load(folder + f"patches/{tile_size}/{prefix}coh_desc.npy"),
            ],
            axis=3,
        )
    elif name == "VVa_VHa_COHa":
        return np.concatenate(
            [
                np.load(folder + f"patches/{tile_size}/{prefix}vv_asc_v2.npy"),
                np.load(folder + f"patches/{tile_size}/{prefix}vh_asc_v2.npy"),
                np.load(folder + f"patches/{tile_size}/{prefix}coh_asc.npy"),
            ],
            axis=3,
        )
    elif name == "RGBN_SWIR":
        return [
            np.load(folder + f"patches/{tile_size}/{prefix}RGBN.npy"),
            np.load(folder + f"patches/{tile_size}/{prefix}SWIR.npy"),
        ]
    elif name == "RGBN_RE":
        return [
            np.load(folder + f"patches/{tile_size}/{prefix}RGBN.npy"),
            np.load(folder + f"patches/{tile_size}/{prefix}REDEDGE.npy"),
        ]
    elif name == "RGBN_RESWIR":
        return [
            np.load(folder + f"patches/{tile_size}/{prefix}RGBN.npy"),
            np.load(folder + f"patches/{tile_size}/{prefix}RESWIR.npy"),
        ]
    elif name == "RGBN_RESWIR_VVa_VVd_COHa_COHd":
        return [
            np.load(folder + f"patches/{tile_size}/{prefix}RGBN.npy"),
            np.concatenate(
                [
                    np.load(folder + f"patches/{tile_size}/{prefix}vv_asc.npy"),
                    np.load(folder + f"patches/{tile_size}/{prefix}vv_desc.npy"),
                    np.load(folder + f"patches/{tile_size}/{prefix}coh_asc.npy"),
                    np.load(folder + f"patches/{tile_size}/{prefix}coh_desc.npy"),
                ],
                axis=3,
            ),
            np.load(folder + f"patches/{tile_size}/{prefix}RESWIR.npy"),
        ]
    elif name == "RGBN_RESWIR_VVa_VHa":
        return [
            np.load(folder + f"patches/{tile_size}/{prefix}RGBN.npy"),
            np.concatenate(
                [
                    np.load(folder + f"patches/{tile_size}/{prefix}vv_asc_v2.npy"),
                    np.load(folder + f"patches/{tile_size}/{prefix}vh_asc_v2.npy"),
                ],
                axis=3,
            ),
            np.load(folder + f"patches/{tile_size}/{prefix}RESWIR.npy"),
        ]
    else:
        raise Exception("Could not find layer.")
