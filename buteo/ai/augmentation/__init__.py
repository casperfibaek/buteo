""" Augmentation module init file. """

# Import all functions from subdirectories
from buteo.ai.augmentation.basic import (
    augmentation_rotation,
    augmentation_rotation_xy,
    augmentation_mirror,
    augmentation_mirror_xy,
    AugmentationRotation,
    AugmentationRotationXY,
    AugmentationMirror,
    AugmentationMirrorXY,
)

from buteo.ai.augmentation.noise import (
    augmentation_noise_uniform,
    augmentation_noise_normal,
    AugmentationNoiseUniform,
    AugmentationNoiseNormal,
)

from buteo.ai.augmentation.transform import (
    augmentation_channel_scale,
    augmentation_contrast,
    augmentation_blur,
    augmentation_blur_xy,
    augmentation_sharpen,
    augmentation_sharpen_xy,
    augmentation_misalign,
    AugmentationChannelScale,
    AugmentationContrast,
    AugmentationBlur,
    AugmentationBlurXY,
    AugmentationSharpen,
    AugmentationSharpenXY,
    AugmentationMisalign,
    AugmentationMisalignLabel,
)

from buteo.ai.augmentation.mix import (
    augmentation_cutmix,
    augmentation_mixup,
    AugmentationCutmix,
    AugmentationMixup,
)

from buteo.ai.augmentation.labels import (
    augmentation_label_smoothing,
    AugmentationLabelSmoothing,
)

# Re-export all functions for backward compatibility
__all__ = [
    # Basic augmentations
    "augmentation_rotation",
    "augmentation_rotation_xy",
    "augmentation_mirror",
    "augmentation_mirror_xy",
    "AugmentationRotation",
    "AugmentationRotationXY",
    "AugmentationMirror",
    "AugmentationMirrorXY",
    
    # Noise augmentations
    "augmentation_noise_uniform",
    "augmentation_noise_normal",
    "AugmentationNoiseUniform",
    "AugmentationNoiseNormal",
    
    # Transform augmentations
    "augmentation_channel_scale",
    "augmentation_contrast",
    "augmentation_blur",
    "augmentation_blur_xy",
    "augmentation_sharpen",
    "augmentation_sharpen_xy",
    "augmentation_misalign",
    "AugmentationChannelScale",
    "AugmentationContrast",
    "AugmentationBlur",
    "AugmentationBlurXY",
    "AugmentationSharpen",
    "AugmentationSharpenXY",
    "AugmentationMisalign",
    "AugmentationMisalignLabel",
    
    # Mix augmentations
    "augmentation_cutmix",
    "augmentation_mixup",
    "AugmentationCutmix",
    "AugmentationMixup",
    
    # Label augmentations
    "augmentation_label_smoothing",
    "AugmentationLabelSmoothing",
]
