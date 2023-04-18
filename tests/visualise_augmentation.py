""" This is a testing script for visually inspecting the augmentation functions. """
# Standard library
import sys; sys.path.append("../")
import os

import numpy as np

import buteo.ai.augmentation_funcs as augmentation
from buteo.raster import raster_to_array, array_to_raster

FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/buteo/tests/"

path_img = os.path.join(FOLDER, "test_image_rgb_8bit.tif")

arr = raster_to_array(path_img)
shortest_side = min(arr.shape[0], arr.shape[1])
offset = (0, 0, shortest_side, shortest_side)

arr = raster_to_array(path_img, pixel_offsets=offset)

array_to_raster(
    arr,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit.tif"),
    pixel_offsets=offset,
)

blurred_x, _blurred_y = augmentation.augmentation_blur(
    arr, None,
    chance=1.0,
    intensity=1.0,
    channel_last=True,
)
array_to_raster(
    blurred_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_blur.tif"),
    pixel_offsets=offset,
)

sharpen_x, _sharpen_y = augmentation.augmentation_sharpen(
    arr, None,
    chance=1.0,
    intensity=1.0,
    channel_last=True,
)
array_to_raster(
    sharpen_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_sharpen.tif"),
    pixel_offsets=offset,
)

rot90_x, _rot90_y = augmentation.augmentation_rotation(
    arr, None,
    chance=1.0,
    k=1,
    channel_last=True,
)
array_to_raster(
    rot90_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_rot90.tif"),
    pixel_offsets=offset,
)

rot180_x, _rot180_y = augmentation.augmentation_rotation(
    arr, None,
    chance=1.0,
    k=2,
    channel_last=True,
)
array_to_raster(
    rot180_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_rot180.tif"),
    pixel_offsets=offset,
)

rot270_x, _rot270_y = augmentation.augmentation_rotation(
    arr, None,
    chance=1.0,
    k=3,
    channel_last=True,
)
array_to_raster(
    rot270_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_rot270.tif"),
    pixel_offsets=offset,
)

mirror_x, _mirror_y = augmentation.augmentation_mirror(
    arr, None,
    chance=1.0,
    k=1,
    channel_last=True,
)
array_to_raster(
    mirror_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_mirror_horisontal.tif"),
    pixel_offsets=offset,
)

mirror_x, _mirror_y = augmentation.augmentation_mirror(
    arr, None,
    chance=1.0,
    k=2,
    channel_last=True,
)
array_to_raster(
    mirror_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_mirror_vertical.tif"),
    pixel_offsets=offset,
)

mirror_x, _mirror_y = augmentation.augmentation_mirror(
    arr, None,
    chance=1.0,
    k=3,
    channel_last=True,
)
array_to_raster(
    mirror_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_mirror_horisontal_vertical.tif"),
    pixel_offsets=offset,
)

noise_x, _noise_y = augmentation.augmentation_noise(
    arr, None,
    chance=1.0,
    max_amount=10.0,
    additive=True,
    channel_last=True,
)
array_to_raster(
    noise_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_noise_additive.tif"),
    pixel_offsets=offset,
)

noise_x, _noise_y = augmentation.augmentation_noise(
    arr, None,
    chance=1.0,
    max_amount=0.1,
    additive=False,
    channel_last=True,
)
array_to_raster(
    noise_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_noise_multiplicative.tif"),
    pixel_offsets=offset,
)

channel_scale_x, _channel_scale_y = augmentation.augmentation_channel_scale(
    arr, None,
    chance=1.0,
    max_amount=10.0,
    additive=True,
    channel_last=True,
)
array_to_raster(
    channel_scale_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_channel_scale_multiplicative.tif"),
    pixel_offsets=offset,
)

channel_scale_x, _channel_scale_y = augmentation.augmentation_channel_scale(
    arr, None,
    chance=1.0,
    max_amount=0.1,
    additive=False,
    channel_last=True,
)
array_to_raster(
    channel_scale_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_channel_scale_additive.tif"),
    pixel_offsets=offset,
)

contrast_x, _contrast_y = augmentation.augmentation_contrast(
    arr, None,
    chance=1.0,
    max_amount=0.1,
    channel_last=True,
)
array_to_raster(
    contrast_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_contrast_01.tif"),
    pixel_offsets=offset,
)

contrast_x, _contrast_y = augmentation.augmentation_contrast(
    arr, None,
    chance=1.0,
    max_amount=0.2,
    channel_last=True,
)
array_to_raster(
    contrast_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_contrast_02.tif"),
    pixel_offsets=offset,
)

drop_pixel_x, _drop_pixel_y = augmentation.augmentation_drop_pixel(
    arr, None,
    chance=1.0,
    drop_probability=0.1,
    drop_value=0.0,
    channel_last=True,
)
array_to_raster(
    drop_pixel_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_drop_pixel.tif"),
    pixel_offsets=offset,
)

drop_channel_x, _drop_channel_y = augmentation.augmentation_drop_channel(
    arr, None,
    chance=1.0,
    drop_probability=0.5,
    drop_value=0.0,
    channel_last=True,
)
array_to_raster(
    drop_channel_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_drop_channel.tif"),
    pixel_offsets=offset,
)

blur_x, _blur_y = augmentation.augmentation_blur(
    arr, None,
    chance=1.0,
    channel_last=True,
)
array_to_raster(
    blur_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_blurred_1.tif"),
    pixel_offsets=offset,
)

blur_x, _blur_y = augmentation.augmentation_blur(
    arr, None,
    chance=1.0,
    intensity=0.5,
    channel_last=True,
)
array_to_raster(
    blur_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_blurred_05.tif"),
    pixel_offsets=offset,
)

sharp_x, _sharp_y = augmentation.augmentation_sharpen(
    arr, None,
    chance=1.0,
    channel_last=True,
)
array_to_raster(
    sharp_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_sharpened_1.tif"),
    pixel_offsets=offset,
)

sharp_x, _sharp_y = augmentation.augmentation_sharpen(
    arr, None,
    chance=1.0,
    intensity=0.5,
    channel_last=True,
)
array_to_raster(
    sharp_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_sharpened_05.tif"),
    pixel_offsets=offset,
)

misaligned_x, _misaligned_y = augmentation.augmentation_misalign_pixels(
    arr, None,
    chance=1.0,
    max_offset=0.5,
    channel_last=True,
)
array_to_raster(
    misaligned_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_misaligned_05.tif"),
    pixel_offsets=offset,
)

misaligned_x, _misaligned_y = augmentation.augmentation_misalign_pixels(
    arr, None,
    chance=1.0,
    max_offset=1.0,
    channel_last=True,
)
array_to_raster(
    misaligned_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_misaligned_1.tif"),
    pixel_offsets=offset,
)


arr_target = raster_to_array(path_img, pixel_offsets=[0, 0, 1000, 1000])
arr_source = raster_to_array(path_img, pixel_offsets=[1000, 1000, 1000, 1000])

cutmix_x, _cutmix_y = augmentation.augmentation_cutmix(
    arr_target, np.random.randint(0, 2, size=arr_target.shape),
    arr_source, np.random.randint(0, 2, size=arr_target.shape),
    max_size=0.75,
    min_size=0.25,
    feather=True,
    feather_dist=6,
    chance=1.0,
    channel_last=True,
)
array_to_raster(
    cutmix_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_cutmix_feather.tif"),
    pixel_offsets=[0, 0, 1000, 1000],
)

cutmix_x, _cutmix_y = augmentation.augmentation_cutmix(
    arr_target, np.random.randint(0, 2, size=arr_target.shape),
    arr_source, np.random.randint(0, 2, size=arr_target.shape),
    max_size=0.75,
    min_size=0.25,
    feather=False,
    chance=1.0,
    channel_last=True,
)
array_to_raster(
    cutmix_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_cutmix.tif"),
    pixel_offsets=[0, 0, 1000, 1000],
)

mixup_x, _mixup_y = augmentation.augmentation_mixup(
    arr_target, np.random.randint(0, 2, size=arr_target.shape),
    arr_source, np.random.randint(0, 2, size=arr_target.shape),
    chance=1.0,
    channel_last=True,
)
array_to_raster(
    mixup_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_mixup.tif"),
    pixel_offsets=[0, 0, 1000, 1000],
)