import yellow
import index
import time

img = yellow.readS2('S2A_MSIL2A_20180727T104021_N0208_R008_T32VNH_20180727T134459.SAFE')
start = time.time()
index.calc(img, [
    'ari',
    'nbr',
    'evi2',
    'evi',
    'moist',
    'cre',
    'mcari',
    'msavi2',
    's2rep',
    'ndvi',
    ], './indices/')
print(f'Execution took: {round(time.time() - start, 2)}s')