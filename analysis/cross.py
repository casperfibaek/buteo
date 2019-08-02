from glob import glob
import sys
sys.path.append('../lib')

from crosslayer_math import layers_math

top = 'E:\\sentinel_1_data\\denmark_2018\\2018_DSC\\66\\processed\\coreg_top_Spk.data\\'
bot = 'E:\\sentinel_1_data\\denmark_2018\\2018_DSC\\66\\processed\\coreg_bot_Spk.data\\'

top_vh = glob(f'{top}*VH*.img')
top_vv = glob(f'{top}*VV*.img')
bot_vh = glob(f'{bot}*VH*.img')
bot_vv = glob(f'{bot}*VV*.img')

layers_math(top_vh, top + 'top_VH_median.tif')
layers_math(top_vv, top + 'top_VV_median.tif')
layers_math(bot_vh, bot + 'bot_VH_median.tif')
layers_math(bot_vv, bot + 'bot_VV_median.tif')
