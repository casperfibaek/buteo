import sys; sys.path.append("../")
import buteo as beo
import numpy as np
import matplotlib.pyplot as plt

image_path = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/projects/buteo/tests/features/test_image_rgb_8bit.tif"
arr = beo.raster_to_array(image_path, pixel_offsets=[900, 500, 100, 100], cast=np.float32, filled=True, fill_value=0.0)


out_path = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/projects/buteo/tests/features/test_image_rgb_8bit_encoded.tif"
beo.array_to_raster(beo.encode_arr_position(arr), reference=image_path, out_path=out_path, pixel_offsets=[900, 500, 100, 100])


# Mask2D = beo.MaskImages(
#     masking_functions=[
#         beo.MaskPixels2D(p=0.05),
#         beo.MaskElipse2D(p=0.2, max_height=0.25, max_width=0.25),
#         beo.MaskLines2DBezier(p=0.025),
#         beo.MaskRectangle2D(p=0.2, max_height=0.25, max_width=0.25),
#     ],
#     per_channel=False,
#     method=0,
#     min_val=0,
#     max_val=255,
#     channel_last=True,
#     inplace=False,
# )

# Mask3D = beo.MaskImages(
#     masking_functions=[
#         beo.MaskPixels3D(p=0.05),
#         beo.MaskElipse3D(p=0.2, max_height=0.25, max_width=0.25),
#         beo.MaskLines3DBezier(p=0.025),
#         beo.MaskRectangle3D(p=0.2, max_height=0.25, max_width=0.25),
#         beo.MaskChannels(p=0.1, max_channels=1),
#     ],
#     per_channel=True,
#     method=3,
#     min_val=0,
#     max_val=255,
#     channel_last=True,
#     inplace=False,
# )

# from tqdm import tqdm
# count = 10
# for _ in tqdm(range(count), total=count):
#     noise_2d = Mask2D(arr)
#     noise_3d = Mask3D(arr)

#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

#     ax1.imshow(np.rint(arr).astype(np.uint8), vmin=0, vmax=255)
#     ax1.set_title('Image 1')
#     ax1.axis('off')

#     ax2.imshow(np.rint(noise_2d).astype(np.uint8), vmin=0, vmax=255, cmap='gray')
#     ax2.set_title('Image 2')
#     ax2.axis('off')

#     ax3.imshow(np.rint(noise_3d).astype(np.uint8), vmin=0, vmax=255, cmap='gray')
#     ax3.set_title('Image 3')
#     ax3.axis('off')

#     plt.tight_layout()
#     plt.show()
#     plt.close()
