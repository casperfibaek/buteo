import sys; sys.path.append('..')
from matplotlib import pyplot as plt
from lib.raster_io import raster_to_array, array_to_raster
from lib.stats_kernel import create_kernel


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


folder = 'C:\\Users\\caspe\\Desktop\\Data\\Sentinel1\\'
in_raster = f'{folder}accra_s1.tif'
out_figure = f'{folder}accra_s1_fkernel.png'
data = raster_to_array(in_raster)

kernel_size = 5
kernel = create_kernel(kernel_size, circular=True, weighted_edges=True, inverted=False, holed=False, normalise=True, weighted_distance=True, sigma=2, distance_calc='guassian', plot=True).astype(np.float)

fig, ax = plt.subplots()

im = ax.imshow(kernel, cmap='bone_r', interpolation=None) # no vmin=0, vmax=1

radius = kernel_size // 2
circle = plt.Circle((radius, radius), radius + 0.5, color='black', fill=False, linestyle='--')
ax.add_artist(circle)
ax.invert_yaxis()

for i in range(kernel_size):
    for j in range(kernel_size):
        ax.text(j, i, f"{kernel[i, j]:.4f}", ha="center", va="center", color="tomato")

# plt.figtext(0.5, 0.025, f"size: {kernel_size}x{kernel_size}, weighted_edges: {weighted_edges}, weighted_distance: {weighted_distance}, method: {distance_calc}", ha="center")
fig.tight_layout()
plt.colorbar(im)

fig.savefig(out_figure, transparent=True, dpi=300, papertype=None, bbox_inches='tight', pad_inches=0)
plt.show()
