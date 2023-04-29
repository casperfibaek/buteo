""" This is a testing script for visually inspecting the augmentation functions. """
# Standard library
import sys; sys.path.append("../")
import os

import numpy as np

from buteo.ai.augmentation_funcs import (
    augmentation_rotation,
    augmentation_blur,
    augmentation_mirror,
    augmentation_sharpen,
    augmentation_cutmix,
    augmentation_mixup,
    augmentation_channel_scale,
    augmentation_contrast,
    augmentation_noise_uniform,
    augmentation_noise_normal,
    augmentation_drop_channel,
    augmentation_drop_pixel,
    augmentation_misalign,
)
from buteo.raster import raster_to_array, array_to_raster

FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/buteo/tests/"

path_img = os.path.join(FOLDER, "test_image_rgb_8bit.tif")

arr = raster_to_array(path_img)
shortest_side = min(arr.shape[0], arr.shape[1])
offset = (0, 0, shortest_side, shortest_side)

arr = raster_to_array(path_img, pixel_offsets=offset, cast=np.float32)

array_to_raster(
    arr,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit.tif"),
    pixel_offsets=offset,
)

blurred_x = augmentation_blur(arr)

array_to_raster(
    blurred_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_blur.tif"),
    pixel_offsets=offset,
)

sharpen_x = augmentation_sharpen(arr)
array_to_raster(
    sharpen_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_sharpen.tif"),
    pixel_offsets=offset,
)

rot90_x = augmentation_rotation(arr, k=1)
array_to_raster(
    rot90_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_rot90.tif"),
    pixel_offsets=offset,
)

rot180_x = augmentation_rotation(arr, k=2)
array_to_raster(
    rot180_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_rot180.tif"),
    pixel_offsets=offset,
)

rot270_x = augmentation_rotation(arr, k=3)
array_to_raster(
    rot270_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_rot270.tif"),
    pixel_offsets=offset,
)

mirror_x = augmentation_mirror(arr, k=1)
array_to_raster(
    mirror_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_mirror_horisontal.tif"),
    pixel_offsets=offset,
)

mirror_x = augmentation_mirror(arr, k=2)
array_to_raster(
    mirror_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_mirror_vertical.tif"),
    pixel_offsets=offset,
)

mirror_x= augmentation_mirror(arr, k=3)
array_to_raster(
    mirror_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_mirror_horisontal_vertical.tif"),
    pixel_offsets=offset,
)

noise_x = augmentation_noise_normal(arr, max_amount=10.0, additive=True)
array_to_raster(
    noise_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_noise_normal_additive.tif"),
    pixel_offsets=offset,
)

noise_x = augmentation_noise_normal(arr, max_amount=0.1, additive=False)
array_to_raster(
    noise_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_noise_normal_multiplicative.tif"),
    pixel_offsets=offset,
)

noise_x = augmentation_noise_uniform(arr, max_amount=25.0, additive=True)
array_to_raster(
    noise_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_noise_uniform_additive.tif"),
    pixel_offsets=offset,
)

noise_x = augmentation_noise_uniform(arr, max_amount=0.1, additive=False)
array_to_raster(
    noise_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_noise_uniform_multiplicative.tif"),
    pixel_offsets=offset,
)

channel_scale_x = augmentation_channel_scale(arr, max_amount=25.0, additive=True)
array_to_raster(
    channel_scale_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_channel_scale_multiplicative.tif"),
    pixel_offsets=offset,
)

channel_scale_x = augmentation_channel_scale(arr, max_amount=0.1, additive=False)
array_to_raster(
    channel_scale_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_channel_scale_additive.tif"),
    pixel_offsets=offset,
)

contrast_x = augmentation_contrast(arr, max_amount=0.1)
array_to_raster(
    contrast_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_contrast_01.tif"),
    pixel_offsets=offset,
)

contrast_x = augmentation_contrast(arr, max_amount=0.2)
array_to_raster(
    contrast_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_contrast_02.tif"),
    pixel_offsets=offset,
)

drop_pixel_x = augmentation_drop_pixel(arr, drop_value=0.0)
array_to_raster(
    drop_pixel_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_drop_pixel.tif"),
    pixel_offsets=offset,
)

drop_channel_x = augmentation_drop_channel(arr, drop_value=0.0)
array_to_raster(
    drop_channel_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_drop_channel.tif"),
    pixel_offsets=offset,
)

misaligned_x = augmentation_misalign(arr, max_offset=1.0)
array_to_raster(
    misaligned_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_misaligned.tif"),
    pixel_offsets=offset,
)

arr_target = raster_to_array(path_img, pixel_offsets=[0, 0, 1000, 1000])
arr_source = raster_to_array(path_img, pixel_offsets=[1000, 1000, 1000, 1000])

cutmix_x, _cutmix_y = augmentation_cutmix(
    arr_target, np.random.randint(0, 2, size=arr_target.shape),
    arr_source, np.random.randint(0, 2, size=arr_target.shape),
    max_size=0.75,
    min_size=0.25,
)
array_to_raster(
    cutmix_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_cutmix.tif"),
    pixel_offsets=[0, 0, 1000, 1000],
)

mixup_x, _mixup_y = augmentation_mixup(
    arr_target, np.random.randint(0, 2, size=arr_target.shape),
    arr_source, np.random.randint(0, 2, size=arr_target.shape),
)
array_to_raster(
    mixup_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_mixup.tif"),
    pixel_offsets=[0, 0, 1000, 1000],
)
