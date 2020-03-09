import pyximport; pyximport.install()
import numpy as np
from stats_calc import global_statistics

arr = np.array([1,2,3,5,3,10,4,7,1,0], dtype=np.double)

bob = global_statistics(arr)

import pdb; pdb.set_trace()

