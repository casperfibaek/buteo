import numpy as np
import cv2

a = np.array([3, 7, 5, 8, 4, 1000, 1, 3, 3, 2])
b = np.array([1, 3, 4, 2, 1, 3, 1, 1, 2, 1])

def mean_match(src, target):
    return target * (src.mean() / target.mean())

def madstd(arr):
    med = np.median(arr)
    meddevabs = np.abs(arr - med)
    devmed = np.median(meddevabs)
    return devmed * 1.4826


def mvm(src, target):
    dif = src - np.median(src)
    ret = ((dif * madstd(target)) / madstd(src)) + np.median(target)
    ret2 = np.rint(ret).astype('uint16')
    # ret = np.abs((dif * target.std()) / src.std()) + target.mean()
    
    
    
aa = np.array([
    [1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 0, 1],
], dtype='uint8')



if __name__ == "__main__":
    import pdb; pdb.set_trace()

    bob = cv2.distanceTransform(aa, cv2.DIST_L2, 0)