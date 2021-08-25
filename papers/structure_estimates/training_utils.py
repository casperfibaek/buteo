import numpy as np


def get_layer(folder, name):
    if name == "rgb":
        return np.load(folder + f"patches/RGB.npy")
    elif name == "rgbn":
        return np.load(folder + f"patches/RGBN.npy")
    elif name == "VVa":
        return np.load(folder + f"patches/_vv_asc_v2.npy")
    elif name == "VVa_VHa":
        return np.concatenate(
            [
                np.load(folder + f"patches/vv_asc_v2.npy"),
                np.load(folder + f"patches/vh_asc_v2.npy"),
            ],
            axis=3,
        )
    elif name == "VVa_VVd":
        return np.concatenate(
            [
                np.load(folder + f"patches/vv_asc.npy"),
                np.load(folder + f"patches/vv_desc.npy"),
            ],
            axis=3,
        )
    elif name == "VVa_COHa":
        return np.concatenate(
            [
                np.load(folder + f"patches/vv_asc.npy"),
                np.load(folder + f"patches/coh_asc.npy"),
            ],
            axis=3,
        )
    elif name == "VVa_VVd_COHa_COHd":
        return np.concatenate(
            [
                np.load(folder + f"patches/vv_asc.npy"),
                np.load(folder + f"patches/vv_desc.npy"),
                np.load(folder + f"patches/coh_asc.npy"),
                np.load(folder + f"patches/coh_desc.npy"),
            ],
            axis=3,
        )
    elif name == "VVa_VHa_COHa":
        return np.concatenate(
            [
                np.load(folder + f"patches/vv_asc_v2.npy"),
                np.load(folder + f"patches/vh_asc_v2.npy"),
                np.load(folder + f"patches/coh_asc.npy"),
            ],
            axis=3,
        )
    elif name == "RGBN_SWIR":
        return [
            np.load(folder + "patches/RGBN.npy"),
            np.load(folder + "patches/SWIR.npy"),
        ]
    elif name == "RGBN_RE":
        return [
            np.load(folder + "patches/RGBN.npy"),
            np.load(folder + "patches/REDEDGE.npy"),
        ]
    elif name == "RGBN_RESWIR":
        return [
            np.load(folder + "patches/RGBN.npy"),
            np.load(folder + "patches/RESWIR.npy"),
        ]
    elif name == "merge_test_join":
        return (
            [
                np.concatenate(
                    [
                        np.load(folder + f"patches/RGBN.npy"),
                        np.load(folder + f"patches/vv_asc_v2.npy"),
                        np.load(folder + f"patches/vh_asc_v2.npy"),
                    ],
                    axis=3,
                ),
                np.load(folder + "patches/SWIR.npy"),
            ],
        )
    elif name == "merge_test_down" or name == "merge_test_end":
        return (
            [
                np.load(folder + f"patches/RGBN.npy"),
                np.concatenate(
                    [
                        np.load(folder + f"patches/vv_asc_v2.npy"),
                        np.load(folder + f"patches/vh_asc_v2.npy"),
                    ],
                    axis=3,
                ),
                np.load(folder + "patches/SWIR.npy"),
            ],
        )
    else:
        raise Exception("Could not find layer.")
