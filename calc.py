import yellow
import index

img = yellow.readS2('S2A_MSIL2A_20180727T104021_N0208_R008_T32VNH_20180727T134459.SAFE')
index.calc(img, ['ndwi'], './indices/')
