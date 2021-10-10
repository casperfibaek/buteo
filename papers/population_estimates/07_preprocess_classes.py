import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/ghana/vector/classes_extra/patches/merged/"

label = np.load(folder + "class_label_class.npy")

argmax = np.reshape(
    np.argmax(label, axis=3), (label.shape[0], label.shape[1], label.shape[1], 1)
)

shaped = np.reshape(
    argmax, (argmax.shape[0], argmax.shape[1] * argmax.shape[2] * argmax.shape[3], 1)
)

# index masks
indices = np.arange(label.shape[0])
class0 = indices[(np.sum((shaped == 0), axis=1) > 0)[:, 0]]
class1 = indices[(np.sum((shaped == 1), axis=1) > 0)[:, 0]]
class2 = indices[(np.sum((shaped == 2), axis=1) > 0)[:, 0]]
class3 = indices[(np.sum((shaped == 3), axis=1) > 0)[:, 0]]


class0_train = class0
class1_train = class1
class2_train = class2
class3_train = class3

# class0_train, class0_test = train_test_split(
#     class0,
#     test_size=2000,
# )

# class1_train, class1_test = train_test_split(
#     class1,
#     test_size=2000,
# )

# class2_train, class2_test = train_test_split(
#     class2,
#     test_size=2000,
# )

# class3_train, class3_test = train_test_split(
#     class3,
#     test_size=500,
# )

# mask_test = np.concatenate(
#     [
#         class0_test,
#         class1_test,
#         class2_test,
#         class3_test,
#     ]
# )


limit = 10000

shuffle_mask = np.random.permutation(class0_train.shape[0])
class0_train = class0_train[shuffle_mask]
class0_train = np.repeat(class0_train, (limit // class0_train.shape[0]) + 1)[:limit]

shuffle_mask = np.random.permutation(class1_train.shape[0])
class1_train = class1_train[shuffle_mask]
class1_train = np.repeat(class1_train, (limit // class1_train.shape[0]) + 1)[:limit]

shuffle_mask = np.random.permutation(class2_train.shape[0])
class2_train = class2_train[shuffle_mask]
class2_train = np.repeat(class2_train, (limit // class2_train.shape[0]) + 1)[:limit]

shuffle_mask = np.random.permutation(class3_train.shape[0])
class3_train = class3_train[shuffle_mask]
class3_train = np.repeat(class3_train, (limit // class3_train.shape[0]) + 1)[:limit]

mask_train = np.concatenate(
    [
        class0_train,
        class1_train,
        class2_train,
        class3_train,
    ]
)

shuffle_mask = np.random.permutation(mask_train.shape[0])
mask_train = mask_train[shuffle_mask]

np.save(
    folder + "class_balanced_label_class.npy",
    np.load(folder + "class_label_class.npy")[mask_train],
)
np.save(
    folder + "class_balanced_RGBN.npy", np.load(folder + "class_RGBN.npy")[mask_train]
)
np.save(
    folder + "class_balanced_SAR.npy", np.load(folder + "class_SAR.npy")[mask_train]
)
np.save(
    folder + "class_balanced_RESWIR.npy",
    np.load(folder + "class_RESWIR.npy")[mask_train],
)

# np.save(
#     folder + "class_balanced_label_class_test.npy",
#     np.load(folder + "class_label_class.npy")[mask_test],
# )
# np.save(
#     folder + "class_balanced_RGBN_test.npy",
#     np.load(folder + "class_RGBN.npy")[mask_test],
# )
# np.save(
#     folder + "class_balanced_SAR_test.npy", np.load(folder + "class_SAR.npy")[mask_test]
# )
# np.save(
#     folder + "class_balanced_RESWIR_test.npy",
#     np.load(folder + "class_RESWIR.npy")[mask_test],
# )
