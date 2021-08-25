import numpy as np


def get_layer(folder, name, prefix=""):
    if name == "RGB":
        return np.load(folder + f"patches/{prefix}RGB.npy")
    elif name == "RGBN":
        return np.load(folder + f"patches/{prefix}RGBN.npy")
    elif name == "VVa":
        return np.load(folder + f"patches/{prefix}vv_asc_v2.npy")
    elif name == "VVa_VHa":
        return np.concatenate(
            [
                np.load(folder + f"patches/{prefix}vv_asc_v2.npy"),
                np.load(folder + f"patches/{prefix}vh_asc_v2.npy"),
            ],
            axis=3,
        )
    elif name == "VVa_VVd":
        return np.concatenate(
            [
                np.load(folder + f"patches/{prefix}vv_asc.npy"),
                np.load(folder + f"patches/{prefix}vv_desc.npy"),
            ],
            axis=3,
        )
    elif name == "VVa_COHa":
        return np.concatenate(
            [
                np.load(folder + f"patches/{prefix}vv_asc.npy"),
                np.load(folder + f"patches/{prefix}coh_asc.npy"),
            ],
            axis=3,
        )
    elif name == "VVa_VVd_COHa_COHd":
        return np.concatenate(
            [
                np.load(folder + f"patches/{prefix}vv_asc.npy"),
                np.load(folder + f"patches/{prefix}vv_desc.npy"),
                np.load(folder + f"patches/{prefix}coh_asc.npy"),
                np.load(folder + f"patches/{prefix}coh_desc.npy"),
            ],
            axis=3,
        )
    elif name == "VVa_VHa_COHa":
        return np.concatenate(
            [
                np.load(folder + f"patches/{prefix}vv_asc_v2.npy"),
                np.load(folder + f"patches/{prefix}vh_asc_v2.npy"),
                np.load(folder + f"patches/{prefix}coh_asc.npy"),
            ],
            axis=3,
        )
    elif name == "RGBN_SWIR":
        return [
            np.load(folder + f"patches/{prefix}RGBN.npy"),
            np.load(folder + f"patches/{prefix}SWIR.npy"),
        ]
    elif name == "RGBN_RE":
        return [
            np.load(folder + f"patches/{prefix}RGBN.npy"),
            np.load(folder + f"patches/{prefix}REDEDGE.npy"),
        ]
    elif name == "RGBN_RESWIR":
        return [
            np.load(folder + f"patches/{prefix}RGBN.npy"),
            np.load(folder + f"patches/{prefix}RESWIR.npy"),
        ]
    elif name == "RGBN_RESWIR_VVa_VVd_COHa_COHd":
        return [
            np.load(folder + f"patches/{prefix}RGBN.npy"),
            np.concatenate(
                [
                    np.load(folder + f"patches/{prefix}vv_asc.npy"),
                    np.load(folder + f"patches/{prefix}vv_desc.npy"),
                    np.load(folder + f"patches/{prefix}coh_asc.npy"),
                    np.load(folder + f"patches/{prefix}coh_desc.npy"),
                ],
                axis=3,
            ),
            np.load(folder + f"patches/{prefix}RESWIR.npy"),
        ]
    elif name == "RGBN_RESWIR_VVa_VHa":
        return [
            np.load(folder + f"patches/{prefix}RGBN.npy"),
            np.concatenate(
                [
                    np.load(folder + f"patches/{prefix}vv_asc_v2.npy"),
                    np.load(folder + f"patches/{prefix}vh_asc_v2.npy"),
                ],
                axis=3,
            ),
            np.load(folder + f"patches/{prefix}RESWIR.npy"),
        ]
    else:
        raise Exception("Could not find layer.")
