""" Tests for ai/augmentation.py """


# Standard library
import sys; sys.path.append("../")

# External
import numpy as np

# Internal
from buteo.ai.augmentation_funcs import (
    augmentation_rotation,
    augmentation_noise,
    augmentation_channel_scale,
    augmentation_contrast,
    augmentation_mirror,
    augmentation_blur,
    augmentation_sharpen,
    augmentation_drop_channel,
    augmentation_cutmix,
    augmentation_mixup,
)


def test_augmentation_rotation():
    """ Test if the rotation augmentation works, sum and shape """
    # Create a dummy image and label
    image = np.random.random((4, 4, 3))
    label = np.random.randint(0, 2, (4, 4, 3))

    # Test without label
    rotated_image, _ = augmentation_rotation(image, None, chance=1.0, k=1)
    assert rotated_image.shape == image.shape
    assert np.allclose(rotated_image.sum(), image.sum())

    # Test with label
    rotated_image, rotated_label = augmentation_rotation(image, label, chance=1.0, k=1)
    assert rotated_image.shape == image.shape
    assert rotated_label.shape == label.shape

def test_augmentation_mirror():
    """ Test if the mirror augmentation works, sum and shape """
    image = np.random.random((32, 32, 3))
    labels = np.random.randint(0, 2, (32, 32, 3))

    flipped_image, flipped_label = augmentation_mirror(image, labels)

    assert flipped_image.shape == image.shape
    assert flipped_label.shape == (32, 32, 3)
    assert np.allclose(flipped_image.sum(), image.sum())
    assert np.allclose(flipped_label.sum(), labels.sum())

def test_augmentation_pixel_noise():
    """ Test if the pixel noise augmentation works."""
    # Create a dummy image
    image = np.ones((4, 4, 3))

    # Test additive noise
    noisy_image, _ = augmentation_noise(image, None, additive=True)
    assert noisy_image.shape == image.shape

    # Test multiplicative noise
    noisy_image, _ = augmentation_noise(image, None, additive=False)
    assert noisy_image.shape == image.shape

def test_augmentation_channel_scale():
    """ Test if the channel scale augmentation works. """
    # Create a dummy image
    image = np.ones((4, 4, 3))

    # Test additive scaling
    scaled_image, _ = augmentation_channel_scale(image, None, additive=True)
    assert scaled_image.shape == image.shape

    # Test multiplicative scaling
    scaled_image, _ = augmentation_channel_scale(image, None, additive=False)
    assert scaled_image.shape == image.shape

def test_augmentation_contrast():
    """ Test if the contrast augmentation works. """
    # Create a dummy image
    image = np.ones((4, 4, 3))

    # Test contrast adjustment
    contrast_image, _ = augmentation_contrast(image, None, chance=1.0, max_amount=0.25)
    assert contrast_image.shape == image.shape

def test_augmentation_drop_channel():
    """Test if a random channel is dropped and the shape and sum is preserved."""
    np.random.seed(42)
    image = np.random.rand(28, 28, 3)
    label = np.random.randint(0, 10, (28, 28, 3))

    aug_image, aug_label = augmentation_drop_channel(image, y=label, chance=1.0, drop_probability=1.0, drop_value=0.0, channel_last=True)

    assert aug_image.shape == image.shape
    assert aug_label.shape == label.shape
    assert aug_image.sum() < image.sum()
    assert np.allclose(aug_label.sum(), label.sum())

def test_augmentation_blur():
    """Test if the image is blurred based on the chance parameter and the intensity."""
    np.random.seed(42)
    image = np.random.rand(28, 28, 3).astype(np.float32, copy=False)

    # Test with chance=1.0, should always blur
    blurred_image, _ = augmentation_blur(image, None, chance=1.0, intensity=1.0, channel_last=True)
    assert blurred_image.shape == image.shape
    assert not np.allclose(blurred_image, image)

    # Test with chance=0.0, should not blur
    not_blurred_image, _ = augmentation_blur(image, None, chance=0.0, intensity=1.0, channel_last=True)
    assert not_blurred_image.shape == image.shape
    assert np.allclose(not_blurred_image, image)

def test_augmentation_sharpen():
    """Test if the image is sharpened based on the chance parameter and the intensity."""
    np.random.seed(42)
    image = np.random.rand(28, 28, 3)

    # Test with chance=1.0, should always sharpen
    sharpened_image, _ = augmentation_sharpen(image, None, chance=1.0, intensity=1.0, channel_last=True)
    assert sharpened_image.shape == image.shape
    assert not np.allclose(sharpened_image, image)

    # Test with chance=0.0, should not sharpen
    not_sharpened_image, _ = augmentation_sharpen(image, None, chance=0.0, intensity=1.0, channel_last=True)
    assert not_sharpened_image.shape == image.shape
    assert np.allclose(not_sharpened_image, image)

def test_augmentation_cutmix_batch():
    """ Test if the cutmix augmentation works (batch). No changes. """
    image1 = np.random.randint(0, 255, size=(32, 32, 3))
    image2 = np.random.randint(0, 255, size=(32, 32, 3))
    cutmix_image, _ = augmentation_cutmix(image1, image1, image2, image2, chance=1.0, channel_last=True)

    assert not np.array_equal(image1, cutmix_image)
    assert cutmix_image.shape == image1.shape

def test_augmentation_mixup():
    """ Test if the mixup augmentation works (batch). """
    image1 = np.random.randint(0, 255, size=(32, 32, 3))
    image2 = np.random.randint(0, 255, size=(32, 32, 3))
    mixup_image, _ = augmentation_mixup(image1, image1, image2, image2, chance=1.0, channel_last=True)

    assert mixup_image.shape == image1.shape
    assert not np.array_equal(image1, mixup_image)
