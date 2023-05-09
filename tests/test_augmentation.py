""" Tests for ai/augmentation.py """
# pylint: disable=missing-function-docstring, bare-except

# Standard library
import sys; sys.path.append("../")

# External
import numpy as np
import os

# Internal
from buteo.ai.augmentation_funcs import (
    augmentation_rotation,
    augmentation_rotation_xy,
    augmentation_mirror,
    augmentation_mirror_xy,
    augmentation_noise_uniform,
    augmentation_noise_normal,
    augmentation_channel_scale,
    augmentation_contrast,
    augmentation_blur,
    augmentation_sharpen,
    augmentation_drop_channel,
    augmentation_drop_pixel,
    augmentation_cutmix,
    augmentation_mixup,
)
from buteo.ai.augmentation import AugmentationDataset
from buteo.raster.core_raster_io import raster_to_array
from buteo.array.patches import array_to_patches
from buteo.ai.selection import split_train_val


test_image = os.path.join(os.path.dirname(__file__), "test_image_rgb_8bit.tif")

def test_augmentation_dataset():
    arr = raster_to_array(test_image, filled=True, fill_value=0.0, cast=np.float32)
    patches = array_to_patches(arr, 256, n_offsets=1)
    x_train, x_val, y_train, y_val = split_train_val(patches, patches, val_size=0.2, random_state=42)
    dataset = AugmentationDataset(x_train, y_train, augmentations=[
        {"name": "rotation_xy", "chance": 0.5},
        {"name": "mirror_xy", "chance": 0.5},
    ])
    for idx, (X, y) in enumerate(dataset):
        assert X.shape == (3, 256, 256)
        assert y.shape == (3, 256, 256)

        if idx > 10:
            break

    dataset = AugmentationDataset(x_val, y_val, augmentations=[
        {"name": "rotation_xy", "chance": 1.0},
        {"name": "mirror_xy", "chance": 1.0},
        {"name": "noise_uniform", "chance": 1.0, "additive": False },
        {"name": "noise_uniform", "chance": 1.0, "additive": True },
        {"name": "noise_normal", "chance": 1.0, "additive": False },
        {"name": "noise_normal", "chance": 1.0, "additive": True },
        {"name": "channel_scale", "chance": 1.0 },
        {"name": "contrast", "chance": 1.0 },
        {"name": "drop_pixel", "chance": 1.0, "drop_probability": 0.1 },
        {"name": "drop_channel", "chance": 1.0, "drop_value": 0.0 },
        {"name": "blur", "chance": 1.0 },
        {"name": "sharpen", "chance": 1.0 },
        {"name": "misalign", "chance": 1.0, "max_offset": 10 },
        {"name": "cutmix", "chance": 1.0 },
        {"name": "mixup", "chance": 1.0 },
    ])

    for idx, (X, y) in enumerate(dataset):
        assert X.shape == (3, 256, 256)
        assert y.shape == (3, 256, 256)

        if idx > 10:
            break


def test_augmentation_rotation():
    """ Test if the rotation augmentation works, sum and shape """
    image = np.random.random((4, 4, 3))

    rotated_image = augmentation_rotation(image)
    assert rotated_image.shape == image.shape
    assert np.allclose(rotated_image.sum(), image.sum())

def test_augmentation_rotation_xy():
    """ Test if the rotation augmentation works, sum and shape """
    # Create a dummy image and label
    image = np.random.random((4, 4, 3))
    label = np.random.randint(0, 2, (4, 4, 3))

    # Test with label
    rotated_image, rotated_label = augmentation_rotation_xy(image, label, k=1)
    assert rotated_image.shape == image.shape
    assert rotated_label.shape == label.shape
    assert np.allclose(rotated_image.sum(), image.sum())
    assert np.allclose(rotated_label.sum(), label.sum())

def test_augmentation_mirror_xy():
    """ Test if the mirror augmentation works, sum and shape """
    image = np.random.random((32, 32, 3))
    label = np.random.randint(0, 2, (32, 32, 3))

    flipped_image, flipped_label = augmentation_mirror_xy(image, label)

    assert flipped_image.shape == image.shape
    assert flipped_label.shape == label.shape
    assert np.allclose(flipped_image.sum(), image.sum())
    assert np.allclose(flipped_label.sum(), label.sum())

def test_augmentation_mirror():
    """ Test if the mirror augmentation works, sum and shape """
    image = np.random.random((32, 32, 3))

    flipped_image = augmentation_mirror(image)

    assert flipped_image.shape == image.shape
    assert np.allclose(flipped_image.sum(), image.sum())

def test_augmentation_pixel_noise_uniform():
    """ Test if the pixel noise augmentation works. (UNIFORM) """
    # Create a dummy image
    image = np.ones((4, 4, 3))

    # Test additive noise
    noisy_image = augmentation_noise_uniform(image, additive=True)
    assert noisy_image.shape == image.shape

    # Test multiplicative noise
    noisy_image = augmentation_noise_uniform(image, additive=False)
    assert noisy_image.shape == image.shape

def test_augmentation_pixel_noise_normal():
    """ Test if the pixel noise augmentation works. (NORMAL) """
    # Create a dummy image
    image = np.ones((4, 4, 3))

    # Test additive noise
    noisy_image = augmentation_noise_normal(image, additive=True)
    assert noisy_image.shape == image.shape

    # Test multiplicative noise
    noisy_image = augmentation_noise_normal(image, additive=False)
    assert noisy_image.shape == image.shape

def test_augmentation_channel_scale():
    """ Test if the channel scale augmentation works. """
    # Create a dummy image
    image = np.ones((4, 4, 3))

    # Test additive scaling
    scaled_image = augmentation_channel_scale(image, additive=True)
    assert scaled_image.shape == image.shape

    # Test multiplicative scaling
    scaled_image = augmentation_channel_scale(image, additive=False)
    assert scaled_image.shape == image.shape

def test_augmentation_contrast():
    """ Test if the contrast augmentation works. """
    # Create a dummy image
    image = np.ones((4, 4, 3))

    # Test contrast adjustment
    contrast_image = augmentation_contrast(image, max_amount=0.25)
    assert contrast_image.shape == image.shape

def test_augmentation_drop_channel():
    """Test if a random channel is dropped and the shape and sum is preserved."""
    np.random.seed(42)
    image = np.random.rand(28, 28, 3)

    aug_image = augmentation_drop_channel(image, drop_value=0.0, channel_last=True)

    assert aug_image.shape == image.shape
    assert aug_image.sum() < image.sum()

def test_augmentation_drop_pixel():
    """Test if a random pixel is dropped and the shape and sum is preserved."""
    np.random.seed(42)
    image = np.random.rand(28, 28, 3)

    aug_image = augmentation_drop_pixel(image, drop_probability=0.1, drop_value=0.0, channel_last=True)

    assert aug_image.shape == image.shape
    assert aug_image.sum() < image.sum()

def test_augmentation_blur():
    """Test if the image is blurred based on the chance parameter."""
    np.random.seed(42)
    image = np.random.rand(28, 28, 3).astype(np.float32, copy=False)

    blurred_image = augmentation_blur(image, channel_last=True)
    assert blurred_image.shape == image.shape
    assert not np.allclose(blurred_image, image)

def test_augmentation_sharpen():
    """Test if the image is sharpened based on the chance parameter."""
    np.random.seed(42)
    image = np.random.rand(28, 28, 3)

    sharpened_image = augmentation_sharpen(image, channel_last=True)
    assert sharpened_image.shape == image.shape
    assert not np.allclose(sharpened_image, image)

def test_augmentation_cutmix_batch():
    """ Test if the cutmix augmentation works. """
    image1 = np.random.randint(0, 255, size=(32, 32, 3))
    image2 = np.random.randint(0, 255, size=(32, 32, 3))
    cutmix_image, _ = augmentation_cutmix(image1, image1, image2, image2, channel_last=True)

    assert not np.array_equal(image1, cutmix_image)
    assert cutmix_image.shape == image1.shape

def test_augmentation_mixup():
    """ Test if the mixup augmentation works. """
    image1 = np.random.randint(0, 255, size=(32, 32, 3))
    image2 = np.random.randint(0, 255, size=(32, 32, 3))
    mixup_image, _ = augmentation_mixup(image1, image1, image2, image2, channel_last=True)

    assert mixup_image.shape == image1.shape
    assert not np.array_equal(image1, mixup_image)
