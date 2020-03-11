import sys; sys.path.append('..')
from matplotlib import pyplot as plt
from lib.raster_io import raster_to_array

folder = 'C:\\Users\\caspe\\Desktop\\Data\\Sentinel1\\'
in_raster = f'{folder}accra_s1.tif'
out_figure = f'{folder}accra_s1_figure.png'
data = raster_to_array(in_raster)

fig, ax = plt.subplots()
ax.axis('off')
im = ax.imshow(data, cmap='viridis', interpolation=None, vmin=0, vmax=1)  # vmin and vmax control interpolation

plt.colorbar(im, shrink=0.68)
fig.tight_layout()
fig.savefig(out_figure, transparent=True, dpi=300, papertype=None, bbox_inches='tight', pad_inches=0)
