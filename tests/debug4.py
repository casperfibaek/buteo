""" Get simple EO data and labels for testing model architectures. """

import os
import sys; sys.path.append("../")
import buteo as beo
import numpy as np

arr1 = np.arange(0, 2048).reshape(32, 1, 8, 8)
arr2 = np.arange(2048, 4096).reshape(32, 1, 8, 8)

multi_array1 = beo.MultiArray([arr1, arr2], shuffle=True)
multi_array2 = beo.MultiArray([arr1, arr2])
multi_array2.set_shuffle_index(multi_array1.get_shuffle_index())

import pdb; pdb.set_trace()
