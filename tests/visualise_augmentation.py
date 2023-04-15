""" This is a testing script for visually inspecting the augmentation functions. """
# Standard library
import sys; sys.path.append("../")
import os

from buteo.ai import augmentation
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
    arr,
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
    arr,
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
    arr,
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
    arr,
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
    arr,
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
    arr,
    chance=1.0,
    channel_last=True,
)
array_to_raster(
    mirror_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_mirror.tif"),
    pixel_offsets=offset,
)

noise_x, _noise_y = augmentation.augmentation_noise(
    arr,
    chance=1.0,
    amount=0.1,
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
    arr,
    chance=1.0,
    amount=0.1,
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
    arr,
    chance=1.0,
    amount=0.1,
    additive=False,
    channel_last=True,
)
array_to_raster(
    channel_scale_x,
    reference=path_img,
    out_path=os.path.join(FOLDER, "tmp_test_image_rgb_8bit_channel_scale.tif"),
    pixel_offsets=offset,
)