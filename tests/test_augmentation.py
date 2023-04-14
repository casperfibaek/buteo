""" Tests for ai/augmentation.py """


# Standard library
import sys; sys.path.append("../")

# External
import numpy as np
import pytest

# Internal
from buteo.ai.augmentation import (
    rotate_arr,
    augmentation_rotation,
    augmentation_noise,
    augmentation_channel_scale,
    augmentation_contrast,
    augmentation_mirror,
    augmentation_blur,
    augmentation_sharpen,
    augmentation_rotation_batch,
    augmentation_noise_batch,
    augmentation_channel_scale_batch,
    augmentation_contrast_batch,
    augmentation_mirror_batch,
    augmentation_drop_channel,
    augmentation_drop_channel_batch,
    augmentation_cutmix_batch,
    augmentation_mixup_batch,
)

def test_rotate_90():
    ch_first = np.random.randint(0, 2, (3, 4, 4))
    ch_last = np.random.randint(0, 2, (4, 4, 3))

    # Test 90-degree rotation for channel_last=True, k=1
    res_90_cl = np.rot90(ch_last, k=-1, axes=(0, 1))
    assert np.array_equal(rotate_arr(ch_last, 1, True), res_90_cl)

    # Test 90-degree rotation for channel_last=True, k=2
    res_90_cl = np.rot90(ch_last, k=-2, axes=(0, 1))
    assert np.array_equal(rotate_arr(ch_last, 2, True), res_90_cl)

    # Test 90-degree rotation for channel_last=True, k=3
    res_90_cl = np.rot90(ch_last, k=-3, axes=(0, 1))
    assert np.array_equal(rotate_arr(ch_last, 3, True), res_90_cl)

    # Test 90-degree rotation for channel_last=False, k=1
    res_90_cl = np.rot90(ch_first, k=-1, axes=(1, 2))
    assert np.array_equal(rotate_arr(ch_first, 1, False), res_90_cl)

    # Test 90-degree rotation for channel_last=False, k=2
    res_90_cl = np.rot90(ch_first, k=-2, axes=(1, 2))
    assert np.array_equal(rotate_arr(ch_first, 2, False), res_90_cl)

    # Test 90-degree rotation for channel_last=False, k=3
    res_90_cl = np.rot90(ch_first, k=-3, axes=(1, 2))
    assert np.array_equal(rotate_arr(ch_first, 3, False), res_90_cl)


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

def test_augmentation_random_rotation_batch():
    """ Test if the rotation augmentation works, sum and shape """
    # Create a dummy batch of images and labels
    images = np.random.random((16, 4, 4, 3))
    labels = np.random.randint(0, 2, (16, 4, 4, 3))

    # Test without labels
    rotated_images, _ = augmentation_rotation_batch(images, None, chance=1.0)
    assert rotated_images.shape == images.shape
    assert np.allclose(rotated_images.sum(), images.sum())

    # Test with labels
    rotated_images, rotated_labels = augmentation_rotation_batch(images, labels, chance=1.0)
    assert rotated_images.shape == images.shape
    assert rotated_labels.shape == labels.shape

def test_augmentation_mirror():
    """ Test if the mirror augmentation works, sum and shape """
    image = np.random.random((32, 32, 3))
    labels = np.random.randint(0, 2, (32, 32, 3))

    flipped_image, flipped_label = augmentation_mirror(image, labels)

    assert flipped_image.shape == image.shape
    assert flipped_label.shape == (32, 32, 3)
    assert np.allclose(flipped_image.sum(), image.sum())
    assert np.allclose(flipped_label.sum(), labels.sum())

def test_augmentation_mirror_batch():
    """ Test if the mirror augmentation works, sum and shape """
    image = np.random.random((16, 32, 32, 3))
    labels = np.random.randint(0, 2, (16, 32, 32, 3))

    flipped_image, flipped_label = augmentation_mirror_batch(image, labels)
    assert flipped_image.shape == image.shape
    assert flipped_label.shape == (16, 32, 32, 3)
    assert np.allclose(flipped_image.sum(), image.sum())
    assert np.allclose(flipped_label.sum(), labels.sum())

def test_augmentation_pixel_noise():
    """ Test if the pixel noise augmentation works."""
    # Create a dummy image
    image = np.ones((4, 4, 3))

    # Test additive noise
    noisy_image, _ = augmentation_noise(image, additive=True)
    assert noisy_image.shape == image.shape

    # Test multiplicative noise
    noisy_image, _ = augmentation_noise(image, additive=False)
    assert noisy_image.shape == image.shape

def test_augmentation_pixel_noise_batch():
    """ Test if the pixel noise augmentation works. """
    # Create a dummy batch of images
    images = np.ones((4, 4, 4, 3))

    # Test additive noise
    noisy_images, _ = augmentation_noise_batch(images, additive=True)
    assert noisy_images.shape == images.shape

    # Test multiplicative noise
    noisy_images, _ = augmentation_noise_batch(images, additive=False)
    assert noisy_images.shape == images.shape

def test_augmentation_channel_scale():
    """ Test if the channel scale augmentation works. """
    # Create a dummy image
    image = np.ones((4, 4, 3))

    # Test additive scaling
    scaled_image, _ = augmentation_channel_scale(image, additive=True)
    assert scaled_image.shape == image.shape

    # Test multiplicative scaling
    scaled_image, _ = augmentation_channel_scale(image, additive=False)
    assert scaled_image.shape == image.shape

def test_augmentation_channel_scale_batch():
    """ Test if the channel scale augmentation works (batch). """
    # Create a dummy batch of images
    images = np.ones((4, 4, 4, 3))

    # Test additive scaling
    scaled_images, _ = augmentation_channel_scale_batch(images, additive=True)
    assert scaled_images.shape == images.shape

    # Test multiplicative scaling
    scaled_images, _ = augmentation_channel_scale_batch(images, additive=False)
    assert scaled_images.shape == images.shape

def test_augmentation_contrast():
    """ Test if the contrast augmentation works. """
    # Create a dummy image
    image = np.ones((4, 4, 3))

    # Test contrast adjustment
    contrast_image, _ = augmentation_contrast(image, contrast_factor=0.25)
    assert contrast_image.shape == image.shape

def test_augmentation_contrast_batch():
    """ Test if the contrast augmentation works (batch). """
    # Create a dummy batch of images
    images = np.ones((4, 4, 4, 3))

    # Test contrast adjustment
    contrast_images, _ = augmentation_contrast_batch(images, contrast_factor=0.25)
    assert contrast_images.shape == images.shape

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

def test_augmentation_drop_channel_batch():
    """Test if a random channel is dropped from a batch of images and the shape and sum is preserved."""
    np.random.seed(42)
    image_batch = np.random.rand(16, 28, 28, 3)
    label_batch = np.random.randint(0, 10, (16, 28, 28, 3))

    aug_image_batch, aug_label_batch = augmentation_drop_channel_batch(image_batch, y=label_batch, chance=1.0, drop_probability=1.0, drop_value=0.0, channel_last=True)

    assert aug_image_batch.shape == image_batch.shape
    assert aug_label_batch.shape == label_batch.shape
    assert aug_image_batch.sum() < image_batch.sum()
    assert np.allclose(aug_label_batch.sum(), label_batch.sum())

def test_augmentation_blur():
    """Test if the image is blurred based on the chance parameter and the intensity."""
    np.random.seed(42)
    image = np.random.rand(28, 28, 3).astype(np.float32, copy=False)

    # Test with chance=1.0, should always blur
    blurred_image, _ = augmentation_blur(image, chance=1.0, intensity=1.0, channel_last=True)
    assert blurred_image.shape == image.shape
    assert not np.allclose(blurred_image, image)

    # Test with chance=0.0, should not blur
    not_blurred_image, _ = augmentation_blur(image, chance=0.0, intensity=1.0, channel_last=True)
    assert not_blurred_image.shape == image.shape
    assert np.allclose(not_blurred_image, image)

def test_augmentation_sharpen():
    """Test if the image is sharpened based on the chance parameter and the intensity."""
    np.random.seed(42)
    image = np.random.rand(28, 28, 3)

    # Test with chance=1.0, should always sharpen
    sharpened_image, _ = augmentation_sharpen(image, chance=1.0, intensity=1.0, channel_last=True)
    assert sharpened_image.shape == image.shape
    assert not np.allclose(sharpened_image, image)

    # Test with chance=0.0, should not sharpen
    not_sharpened_image, _ = augmentation_sharpen(image, chance=0.0, intensity=1.0, channel_last=True)
    assert not_sharpened_image.shape == image.shape
    assert np.allclose(not_sharpened_image, image)


@pytest.fixture
def create_images():
    """ Create a batch of images for testing. """
    batch_size = 10
    height = 32
    width = 32
    channels = 3
    images = np.random.randint(0, 255, size=(batch_size, height, width, channels))
    return images

def test_augmentation_cutmix_batch(create_images):
    """ Test if the cutmix augmentation works (batch). """
    images = create_images
    cutmix_images, _ = augmentation_cutmix_batch(images, chance=1.0, max_size=0.5, max_mixes=1.0, channel_last=True)

    assert cutmix_images.shape == images.shape
    assert not np.array_equal(cutmix_images, images)

def test_augmentation_cutmix_batch_no_change(create_images):
    """ Test if the cutmix augmentation works (batch). No changes. """
    images = create_images
    cutmix_images, _ = augmentation_cutmix_batch(images, chance=0.0, channel_last=True)

    assert cutmix_images.shape == images.shape
    assert np.array_equal(cutmix_images, images)

def test_augmentation_mixup(create_images):
    images = create_images
    mixup_images, _ = augmentation_mixup_batch(images, chance=1.0, max_mixes=1.0, channel_last=True)

    assert mixup_images.shape == images.shape
    assert not np.array_equal(mixup_images, images)

def test_augmentation_mixup_no_change(create_images):
    images = create_images
    mixup_images, _ = augmentation_mixup_batch(images, chance=0.0, channel_last=True)

    assert mixup_images.shape == images.shape
    assert np.array_equal(mixup_images, images)
