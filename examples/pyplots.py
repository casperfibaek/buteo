kernel_size = 5
kernel = create_kernel(kernel_size, circular=True, weighted_edges=True, inverted=False, holed=False, normalise=True, weighted_distance=True, sigma=2, distance_calc='guassian', plot=True).astype(np.float)
print(kernel)
# kernel_sum = create_kernel(kernel_size, circular=True, weighted_edges=True, inverted=False, ring=False, holed=False, normalise=True, weighted_distance=True, distance_scale=True, distance_calc='linear', plot=False).astype(np.float)
# kernel_holed = create_kernel(kernel_size, circular=True, weighted_edges=True, inverted=False, ring=False, holed=True, normalise=True, weighted_distance=True, distance_scale=True, distance_calc='linear', plot=False).astype(np.float)
# kernel_ring = create_kernel(kernel_size, circular=True, weighted_edges=True, inverted=False, ring=True, holed=True, normalise=True, weighted_distance=True, distance_scale=True, distance_calc='linear', plot=False).astype(np.float)

# folder_s1_grd = 'C:\\Users\\caspe\\Desktop\\Data\\Sentinel1\\S1B_IW_GRDH_1SDV_20200204T181721_20200204T181746_020123_02616A_B5C4_Orb_NR_Cal_TF_TC_Stack.data\\'
# folder_s1 = 'C:\\Users\\caspe\\Desktop\\Data\\Sentinel1\\'
# folder_s2 = 'C:\\Users\\caspe\\Desktop\\Data\\Sentinel2\\'
# in_raster_coh = f'{folder_s1}coherence_accra.tif'
# in_raster_s1 = f'{folder}accra_s1.tif'
# in_raster_b4 = f'{folder_s2}accra_b4.jp2'
# in_raster_b8 = f'{folder}accra_b8.jp2'
# out_raster_mad_b4 = f'{folder}accra_b4_{kernel_size}x{kernel_size}_mad-std.tif'
# out_raster_mad_b8 = f'{folder}accra_b8_{kernel_size}x{kernel_size}_mad-std.tif'
# out_raster_meddev_b4 = f'{folder}accra_b4_{kernel_size}x{kernel_size}_med-dev.tif'
# out_raster_meddev_b8 = f'{folder}accra_b8_{kernel_size}x{kernel_size}_med-dev.tif'
# out_raster_ndvi = f'{folder}accra_ndvi_i.tif'
# out_raster_s1 = f'{folder}accra_med_5x5.tif'
# out_raster_coh = f'{folder_s1}coherence_accra_pow_med_5x5.tif'

# b4 = raster_to_array(in_raster_b4).astype(np.float)
# b8 = raster_to_array(in_raster_b8).astype(np.float)
# s1 = raster_to_array(in_raster_s1).astype(np.float)
# coh = raster_to_array(in_raster_coh).astype(np.float)

# array_to_raster(filter_2d(b4, kernel, 'mean').astype('float32'), out_raster=f'{folder_s2}filter_2d_test_mean.tif', reference_raster=in_raster_b4, dst_nodata=None)

#   Print kernel
# fig, ax = plt.subplots()
# im = ax.imshow(kernel, cmap='bone_r', interpolation=None)

# radius = floor((kernel_size - 1) / 2)
# circle = plt.Circle((radius, radius), radius + 0.5, color='black', fill=False, linestyle='--')
# ax.add_artist(circle)
# ax.invert_yaxis()

# for i in range(kernel_size):
#     for j in range(kernel_size):
#         ax.text(j, i, f"{kernel[i, j]:.4f}", ha="center", va="center", color="tomato")

# plt.colorbar(im, shrink=1)
# fig.tight_layout()
# fig.savefig(f'{folder_s2}kernel_5x5_weighted_power_holed.png', transparent=True, dpi=300, papertype=None, bbox_inches='tight', pad_inches=0)

#   Print images S
# fig, ax = plt.subplots()
# ax.axis('off')
# img = raster_to_array(f'{folder_s1}coherence_3x3-median_project-area.tif').astype('float32')
# im = ax.imshow(img, cmap='viridis', interpolation=None, vmin=0, vmax=1)

# plt.colorbar(im, shrink=0.68)
# # plt.colorbar(im, shrink=0.8)
# fig.tight_layout()
# fig.savefig(f'{folder_s1}coherence_3x3-median_project-area.png', transparent=True, dpi=300, papertype=None, bbox_inches='tight', pad_inches=0)


# surf = raster_to_array(f'{folder_s1}surf_v2.tif').astype(np.float)
# coh = raster_to_array(f'{folder_s1}coherence_accra.tif').astype(np.float)
# grd_1 = raster_to_array(f'{folder_s1_grd}Gamma0_VV_mst_04Feb2020.img').astype(np.float)
# grd_2 = raster_to_array(f'{folder_s1_grd}Gamma0_VV_slv1_10Feb2020.img').astype(np.float)
# stack = np.array([grd_1,  grd_2])

# before2 = time()
# array_to_raster(median_3d(stack, kernel).astype('float32'), out_raster=f'{folder_s1}backscatter_3x3-median.tif', reference_raster=f'{folder_s1_grd}Gamma0_VV_mst_04Feb2020.img', dst_nodata=None)
# array_to_raster(median(coh, kernel).astype('float32'), out_raster=f'{folder_s1}coherence_7x7-median.tif', reference_raster=f'{folder_s1}coherence_accra.tif', dst_nodata=None)
# array_to_raster(mad_3d(stack, kernel).astype('float32'), out_raster=f'{folder_s1}grd_5x5-3d-mad.tif', reference_raster=f'{folder_s1}Gamma0_VV_mst_04Feb2020.img', dst_nodata=None)
# array_to_raster((((b4 - b8) / (b4 + b8)) + 1).astype('float32'), out_raster=out_raster_ndvi, reference_raster=in_raster_b4, dst_nodata=None)
# array_to_raster(mad(b8, kernel).astype('float32'), out_raster=out_raster_mad_b8, reference_raster=in_raster_b8, dst_nodata=None)
# array_to_raster(mean(surf, kernel).astype('float32'), out_raster=f'{folder_s1}surf_v2_250m-density_weighted_v2.tif', reference_raster=f'{folder_s1}surf_v2.tif', dst_nodata=None)
# array_to_raster(mean(np.abs(median(b8, kernel_holed) - b8), kernel).astype('float32'), out_raster=out_raster_meddev_b8, reference_raster=in_raster_b4, dst_nodata=None)
# array_to_raster(np.power(median(coh, kernel), 2).astype('float32'), out_raster=out_raster_coh, reference_raster=in_raster_coh, dst_nodata=None)
# print(time() - before2)

# Urban syria
# fig, ax = plt.subplots()

# im = ax.imshow(kernel, cmap='bone_r', interpolation=None, vmin=0, vmax=1)

# circle = plt.Circle((radius, radius), radius + 0.5, color='black', fill=False, linestyle='--')
# ax.add_artist(circle)
# ax.invert_yaxis()

# for i in range(width):
#     for j in range(width):
#         ax.text(j, i, f"{kernel[i, j]:.4f}", ha="center", va="center", color="tomato")

# plt.figtext(0.5, 0.025, f"size: {width}x{width}, weighted_edges: {weighted_edges}, weighted_distance: {weighted_distance}, method: {distance_calc}", ha="center")
# fig.tight_layout()
# plt.colorbar(im)

# fig.savefig(f'C:\\Users\\caspe\\Desktop\\7x7_gauss_sigma3.png', transparent=True, dpi=300, papertype=None, bbox_inches='tight', pad_inches=0)
# plt.show()
