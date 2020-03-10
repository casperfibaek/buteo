import pyximport; pyximport.install()
import numpy as np
from stats_calc import global_statistics, enumerate_stats

arr = np.array([1,2,3,5,3,10,4,7,1,0], dtype=np.double)

stats = enumerate_stats(['q1', 'q2', 'q3', 'iqr'])
bob = global_statistics(arr, stats)

import pdb; pdb.set_trace()

