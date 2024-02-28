import random

import cv2
import numpy as np
from albumentations.augmentations.utils import (
    clipped,
    get_opencv_dtype_from_numpy,
    preserve_shape,
)
from qudida import DomainAdapter
from skimage.exposure import match_histograms
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@clipped
@preserve_shape
def fourier_domain_adaptation(img: np.ndarray, target_img: np.ndarray, blend_ratio: float) -> np.ndarray:
    """
    Fourier Domain Adaptation from https://github.com/YanchaoYang/FDA

    Args:
        img:  source image
        target_img:  target image for domain adaptation
        beta: coefficient from source paper

    Returns:
        transformed image

    """
    beta = np.random.choice(np.arange(0.01, 0.1, 0.01))
    initial_type = img.dtype
    img = np.squeeze(img)
    target_img = np.squeeze(target_img)

    if img.shape[:2] != target_img.shape[:2]:
        target_img = cv2.resize(target_img, dsize=(img.shape[1], img.shape[0]))

    # get fft of both source and target
    fft_src = np.fft.fft2(img.astype(np.float32), axes=(0, 1))
    fft_trg = np.fft.fft2(target_img.astype(np.float32), axes=(0, 1))

    # extract amplitude and phase of both fft-s
    amplitude_src, phase_src = np.abs(fft_src), np.angle(fft_src)
    amplitude_trg = np.abs(fft_trg)

    # mutate the amplitude part of source with target
    amplitude_src = np.fft.fftshift(amplitude_src, axes=(0, 1))
    amplitude_trg = np.fft.fftshift(amplitude_trg, axes=(0, 1))
    height, width = amplitude_src.shape[:2]
    border = np.floor(min(height, width) * beta).astype(int)
    center_y, center_x = np.floor([height / 2.0, width / 2.0]).astype(int)

    y1, y2 = center_y - border, center_y + border + 1
    x1, x2 = center_x - border, center_x + border + 1

    amplitude_src[y1:y2, x1:x2] = amplitude_trg[y1:y2, x1:x2]
    amplitude_src = np.fft.ifftshift(amplitude_src, axes=(0, 1))

    # get mutated image
    src_image_transformed = np.fft.ifft2(amplitude_src * np.exp(1j * phase_src), axes=(0, 1))
    transformed = np.real(src_image_transformed)

    transformed = (img.astype("float32") * (1 - blend_ratio) + transformed.astype("float32") * blend_ratio).astype(
        initial_type)

    return transformed


@preserve_shape
def histogram_match_adaptation(img: np.ndarray, reference_image: np.ndarray, blend_ratio: float) -> np.ndarray:
    if img.dtype != reference_image.dtype:
        raise RuntimeError(
            f"Dtype of image and reference image must be the same. Got {img.dtype} and {reference_image.dtype}"
        )
    if img.shape[:2] != reference_image.shape[:2]:
        reference_image = cv2.resize(reference_image, dsize=(img.shape[1], img.shape[0]))

    img, reference_image = np.squeeze(img), np.squeeze(reference_image)

    try:
        matched = match_histograms(img, reference_image, channel_axis=2 if len(img.shape) == 3 else None)
    except TypeError:
        matched = match_histograms(img, reference_image, multichannel=True)  # case for scikit-image<0.19.1
    transformed = cv2.addWeighted(matched, blend_ratio, img, 1 - blend_ratio, 0,
                                  dtype=get_opencv_dtype_from_numpy(img.dtype))
    return transformed


@preserve_shape
def pixel_distribution_adaptation(
        img: np.ndarray, ref: np.ndarray, blend_ratio: float = 0.5) -> np.ndarray:
    initial_type = img.dtype
    transform_type = random.choice(["pca", "standard", "minmax"])
    transformer = {"pca": PCA, "standard": StandardScaler, "minmax": MinMaxScaler}[transform_type]()
    adapter = DomainAdapter(transformer=transformer, ref_img=ref)
    result = adapter(img).astype("float32")
    transformed = (img.astype("float32") * (1 - blend_ratio) + result * blend_ratio).astype(initial_type)
    return transformed


def domain_aug(img: np.ndarray, reference_image: np.ndarray, blend_ratio: float,
               domain_aug_methods: list) -> np.ndarray:

    aug = np.random.choice(domain_aug_methods)
    if aug == 'FDA':
        apply_aug = fourier_domain_adaptation
    elif aug == 'HM':
        apply_aug = histogram_match_adaptation
    elif aug == 'PDA':
        apply_aug = pixel_distribution_adaptation
    return apply_aug(img, reference_image, blend_ratio)
