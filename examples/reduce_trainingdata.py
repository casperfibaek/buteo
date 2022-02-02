import os
import numpy as np


main_folder = "C:/Users/caspe/Desktop/buteo_dataclip/ghana/"
sub_folders = os.listdir(main_folder)

labels = np.load(main_folder + "LABEL.npz")["label"]
labels[labels < 0] = 0.0
labels[labels > 100] = 100.0

rgbn = np.load(main_folder + "RGBN.npz")["rgbn"]
rgbn[rgbn < 0] = 0.0
rgbn[rgbn > 1] = 1.0
rgbn = (rgbn - rgbn.min()) / np.ptp(rgbn)

reswir = np.load(main_folder + "RESWIR.npz")["reswir"]
reswir[reswir < 0] = 0.0
reswir[reswir > 1] = 1.0
reswir = (reswir - reswir.min()) / np.ptp(reswir)

sar = np.load(main_folder + "SAR.npz")["sar"]
sar = np.nan_to_num(sar, copy=False, nan=0.0)
sar[sar < 0] = 0.0
sar[sar > 1] = 1.0

sar = (sar - sar.min()) / np.ptp(sar)

shuffle_mask = np.random.permutation(labels.shape[0])
labels = labels[shuffle_mask]
rgbn = rgbn[shuffle_mask]
reswir = reswir[shuffle_mask]
sar = sar[shuffle_mask]

test_size = labels.shape[0] // 10

# labels = labels[:55000]
# rgbn = rgbn[:55000]
# reswir = reswir[:55000]
# sar = sar[:55000]

# test_size = 5000

np.savez_compressed(
    main_folder + "ghana_test.npz",
    labels=labels[-test_size:],
    rgbn=rgbn[-test_size:],
    reswir=reswir[-test_size:],
    sar=sar[-test_size:],
)

np.savez_compressed(
    main_folder + "ghana_train.npz",
    labels=labels[:-test_size],
    rgbn=rgbn[:-test_size],
    reswir=reswir[:-test_size],
    sar=sar[:-test_size],
)


# shuffle_mask = None
# for layer_type in ["labels", "rgbn", "reswir", "sar"]:
#     arr = []
#     for folder_name in sub_folders:
#         folder = main_folder + folder_name + "/"

#         if not os.path.isdir(folder):
#             continue

#         if layer_type == "labels":
#             arr.append(np.load(folder + "LABEL_100k.npz")["label"])
#         elif layer_type == "rgbn":
#             arr.append(np.load(folder + "RGBN_100k.npz")["rgbn"])
#         elif layer_type == "reswir":
#             arr.append(np.load(folder + "RESWIR_100k.npz")["reswir"])
#         elif layer_type == "sar":
#             arr.append(np.load(folder + "SAR_100k.npz")["sar"])
    
#     arr = np.concatenate(arr, axis=0)

#     if shuffle_mask is None:
#         shuffle_mask = np.random.permutation(arr.shape[0])
    
#     np.savez_compressed(main_folder + layer_type + "_100k", **{layer_type: arr[shuffle_mask]})


# q2 = main_folder + "tanzania_mwanza_Q2/"
# q3 = main_folder + "tanzania_mwanza_Q3/"

# dst = main_folder + "tanzania_mwanza_Q2_Q3/"

# label_q2 = np.load(q2 + "LABEL.npz")["label"]
# label_q3 = np.load(q3 + "LABEL.npz")["label"]
# label_concat = np.concatenate((label_q2, label_q3))
# shuffle_mask = np.random.permutation(label_concat.shape[0])
# np.savez_compressed(dst + "LABEL", label=label_concat[shuffle_mask])
# label_q2 = None
# label_q3 = None

# rgbn_q2 = np.load(q2 + "RGBN.npz")["rgbn"]
# rgbn_q3 = np.load(q3 + "RGBN.npz")["rgbn"]
# np.savez_compressed(dst + "RGBN", rgbn=np.concatenate((rgbn_q2, rgbn_q3))[shuffle_mask])
# rgbn_q2 = None
# rgbn_q3 = None

# reswir_q2 = np.load(q2 + "RESWIR.npz")["reswir"]
# reswir_q3 = np.load(q3 + "RESWIR.npz")["reswir"]
# np.savez_compressed(dst + "RESWIR", reswir=np.concatenate((reswir_q2, reswir_q3))[shuffle_mask])
# reswir_q2 = None
# reswir_q3 = None

# sar_q2 = np.load(q2 + "SAR.npz")["sar"]
# sar_q3 = np.load(q3 + "SAR.npz")["sar"]
# np.savez_compressed(dst + "SAR", sar=np.concatenate((sar_q2, sar_q3))[shuffle_mask])
# sar_q2 = None
# sar_q3 = None

# exit()


# for folder_name in sub_folders:
#     if folder_name not in [
#         "denmark_2021_Q2",
#         "bornholm_2021_Q2",
#     ]:
#         continue

#     folder = main_folder + folder_name + "/"

#     label = np.load(folder + "LABEL.npz")["label"]
#     print(folder_name, label.shape[0])

#     sums = label.sum(axis=(1,2,3))
#     sums_argsort = np.argsort(sums)

#     if label.shape[0] > 50000:
#         top_10k = sums_argsort[-10000:]
#         rest_40k = sums_argsort[:-10000]
#         random_mask_40k = np.random.permutation(rest_40k.shape[0])
#         rest_40k = rest_40k[random_mask_40k]
#         rest_40k = rest_40k[:40000]

#         small_50k_mask = np.concatenate((top_10k, rest_40k))
#         small_50k_mask_random = np.random.permutation(small_50k_mask.shape[0])
#         small_50k_mask = small_50k_mask[small_50k_mask_random]

#         np.savez_compressed(folder + "LABEL_50k", label=label[small_50k_mask])
#         np.savez_compressed(folder + "RGBN_50k", rgbn=np.load(folder + "RGBN.npz")["rgbn"][small_50k_mask])
#         np.savez_compressed(folder + "RESWIR_50k", reswir=np.load(folder + "RESWIR.npz")["reswir"][small_50k_mask])
#         np.savez_compressed(folder + "SAR_50k", sar=np.load(folder + "SAR.npz")["sar"][small_50k_mask])
#     else:
#         np.savez_compressed(folder + "LABEL_50k", label=label)
#         np.savez_compressed(folder + "RGBN_50k", rgbn=np.load(folder + "RGBN.npz")["rgbn"])
#         np.savez_compressed(folder + "RESWIR_50k", reswir=np.load(folder + "RESWIR.npz")["reswir"])
#         np.savez_compressed(folder + "SAR_50k", sar=np.load(folder + "SAR.npz")["sar"])
    
#     if label.shape[0] > 100000:
#         top_20k = sums_argsort[-20000:]
#         rest_80k = sums_argsort[:-20000]
#         random_mask_80k = np.random.permutation(rest_80k.shape[0])
#         rest_80k = rest_80k[random_mask_80k]
#         rest_80k = rest_80k[:80000]

#         large_100k_mask = np.concatenate((top_20k, rest_80k))
#         large_100k_mask_random = np.random.permutation(large_100k_mask.shape[0])
#         large_100k_mask = large_100k_mask[large_100k_mask_random]

#         np.savez_compressed(folder + "LABEL_100k", label=label[large_100k_mask])
#         np.savez_compressed(folder + "RGBN_100k", rgbn=np.load(folder + "RGBN.npz")["rgbn"][large_100k_mask])
#         np.savez_compressed(folder + "RESWIR_100k", reswir=np.load(folder + "RESWIR.npz")["reswir"][large_100k_mask])
#         np.savez_compressed(folder + "SAR_100k", sar=np.load(folder + "SAR.npz")["sar"][large_100k_mask])
#     else:
#         np.savez_compressed(folder + "LABEL_100k", label=label)
#         np.savez_compressed(folder + "RGBN_100k", rgbn=np.load(folder + "RGBN.npz")["rgbn"])
#         np.savez_compressed(folder + "RESWIR_100k", reswir=np.load(folder + "RESWIR.npz")["reswir"])
#         np.savez_compressed(folder + "SAR_100k", sar=np.load(folder + "SAR.npz")["sar"])

#     continue

    # mask_ints = np.arange(label.shape[0])
    # mask_above = label.sum(axis=(1, 2, 3)) > 0
    # mask_below = ~mask_above

    # print(f"{folder_name} - Percent empty: {round((mask_below.sum() / label.shape[0]) * 100, 2)}")

    # masked_above = mask_ints[mask_above]
    # masked_below = mask_ints[mask_below]
    # max_empty = int(masked_above.shape[0] * 0.2)

    # final_mask = np.concatenate([
    #     masked_above,
    #     masked_below[:max_empty]
    # ])
    # final_shuffle = np.random.permutation(final_mask.shape[0])
    # final_mask = final_mask[final_shuffle]

    # np.savez_compressed(folder + "LABEL_reduced", label=label[final_mask])
    # label = None

    # rgbn = np.load(folder + "RGBN.npz")["rgbn"]
    # np.savez_compressed(folder + "RGBN_reduced", rgbn=rgbn[final_mask])
    # rgbn = None

    # reswir = np.load(folder + "RESWIR.npz")["reswir"]
    # np.savez_compressed(folder + "RESWIR_reduced", reswir=reswir[final_mask])
    # reswir = None
    # try:
    #     sar = np.load(folder + "SAR.npz")["sar"]
    # except:
    #     try:
    #         sar = np.load(folder + "SAR.npz")["reswir"]
    #     except:
    #         raise Exception("No SAR found.")

    # np.savez_compressed(folder + "SAR_reduced", sar=sar[final_mask])
    # sar = None
