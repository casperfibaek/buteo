import sys; sys.path.append('..')
from matplotlib import pyplot as plt
from lib.raster_io import raster_to_array

folder = 'C:\\Users\\caspe\\Desktop\\analysis_p2\\satellite_data\\'

in_raster = f'{folder}DEM_slope.tif'
out_figure = f'{folder}DEM_slope_fig.png'

data = raster_to_array(in_raster)

fig, ax = plt.subplots()
ax.axis('off')
im = ax.imshow(data, cmap='magma', interpolation='bilinear', vmin=0, vmax=30.0)  # vmin and vmax control interpolation

plt.colorbar(im, shrink=0.68)
fig.tight_layout()
fig.savefig(out_figure, transparent=True, dpi=300, papertype=None, bbox_inches='tight', pad_inches=0)
