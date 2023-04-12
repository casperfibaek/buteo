""" Tests for ai/augmentation.py """


# Standard library
import sys; sys.path.append("../")

# External
import numpy as np

# Internal
from buteo.ai.augmentation import (
    augmentation_rotation,
    augmentation_pixel_noise,
    augmentation_channel_scale,
    augmentation_contrast,
    augmentation_mirror,
    augmentation_rotation_batch,
    augmentation_pixel_noise_batch,
    augmentation_channel_scale_batch,
    augmentation_contrast_batch,
    augmentation_mirror_batch,
)


def test_augmentation_rotation():
    """ Test if the rotation augmentation works, sum and shape """
    # Create a dummy image and label
    image = np.random.random((4, 4, 3))
    label = np.random.randint(0, 2, (4, 4, 3))

    # Test without label
    rotated_image = augmentation_rotation(image)
    assert rotated_image.shape == image.shape
    assert np.allclose(rotated_image.sum(), image.sum())

    # Test with label
    rotated_image, rotated_label = augmentation_rotation(image, label)
    assert rotated_image.shape == image.shape
    assert rotated_label.shape == label.shape

def test_augmentation_random_rotation_batch():
    """ Test if the rotation augmentation works, sum and shape """ 
    # Create a dummy batch of images and labels
    images = np.random.random((16, 4, 4, 3))
    labels = np.random.randint(0, 2, (16, 4, 4, 3))

    # Test without labels
    rotated_images = augmentation_rotation_batch(images)
    assert rotated_images.shape == images.shape
    assert np.allclose(rotated_images.sum(), images.sum())

    # Test with labels
    rotated_images, rotated_labels = augmentation_rotation_batch(images, labels)
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
    noisy_image = augmentation_pixel_noise(image, additive=True)
    assert noisy_image.shape == image.shape

    # Test multiplicative noise
    noisy_image = augmentation_pixel_noise(image, additive=False)
    assert noisy_image.shape == image.shape

def test_augmentation_pixel_noise_batch():
    """ Test if the pixel noise augmentation works. """
    # Create a dummy batch of images
    images = np.ones((4, 4, 4, 3))

    # Test additive noise
    noisy_images = augmentation_pixel_noise_batch(images, additive=True)
    assert noisy_images.shape == images.shape

    # Test multiplicative noise
    noisy_images = augmentation_pixel_noise_batch(images, additive=False)
    assert noisy_images.shape == images.shape

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

def test_augmentation_channel_scale_batch():
    """ Test if the channel scale augmentation works (batch). """
    # Create a dummy batch of images
    images = np.ones((4, 4, 4, 3))

    # Test additive scaling
    scaled_images = augmentation_channel_scale_batch(images, additive=True)
    assert scaled_images.shape == images.shape

    # Test multiplicative scaling
    scaled_images = augmentation_channel_scale_batch(images, additive=False)
    assert scaled_images.shape == images.shape

def test_augmentation_contrast():
    """ Test if the contrast augmentation works. """
    # Create a dummy image
    image = np.ones((4, 4, 3))

    # Test contrast adjustment
    contrast_image = augmentation_contrast(image, contrast_factor=0.25)
    assert contrast_image.shape == image.shape

def test_augmentation_contrast_batch():
    """ Test if the contrast augmentation works (batch). """
    # Create a dummy batch of images
    images = np.ones((4, 4, 4, 3))

    # Test contrast adjustment
    contrast_images = augmentation_contrast_batch(images, contrast_factor=0.25)
    assert contrast_images.shape == images.shape
