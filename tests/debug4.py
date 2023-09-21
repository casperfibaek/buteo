import os
import sys; sys.path.append("../")
import buteo as beo
import numpy as np

# kernel_np = np.array([
#     [0.4049764, 0.7942472, 0.4049764],
#     [0.7942472, 1.       , 0.7942472],
#     [0.4049764, 0.7942472, 0.4049764]], dtype="float32")

# kernel_np[1, 1] = 5.396

kernel_np = np.array([
    [0.4049764, 0.7942472, 0.4049764],
    [0.7942472, 0.       , 0.7942472],
    [0.4049764, 0.7942472, 0.4049764]], dtype="float32")

arr = np.array([
    [1, 2, 1, 1, 2],
    [1, 1, 2, 1, 3],
    [1, 1, 1, 1, 3],
    [1, 1, 1, 3, 3],
    [1, 1, 1, 3, 3],
], dtype="uint8")
arr = arr[:, :, np.newaxis]

classes_shape = np.array([1, 2, 3]).reshape(1, 1, -1)
classes_hot = (arr == classes_shape).astype(np.uint8)
convolved = beo.filter_operation(classes_hot, 1, radius=1, normalised=False, kernel=kernel_np, channel_last=True)

weight = kernel_np.sum() * (9 / 8)
valmax = np.max(convolved, axis=2, keepdims=True)
argmax = np.argmax(convolved, axis=2, keepdims=True)
argmax_hot = np.argmax(classes_hot, axis=2, keepdims=True)


weight = np.where(argmax == argmax_hot, convolved, valmax * (9/8))
convolved = np.where(classes_hot == 1, weight, convolved)

carl = convolved / convolved.sum(axis=2, keepdims=2)
done = np.round(carl, 3)

import pdb; pdb.set_trace()
