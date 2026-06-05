"""
Image quality metrics: PSNR and SSIM.
"""

import numpy as np
from skimage.metrics import structural_similarity


def psnr(original: np.ndarray, reconstructed: np.ndarray,
         data_range: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio (dB).
    PSNR = 10 * log10(data_range^2 / MSE)
    Higher is better; > 30 dB is generally acceptable.
    """
    mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10((data_range ** 2) / mse)


def ssim_metric(original: np.ndarray, reconstructed: np.ndarray,
                data_range: float = 1.0) -> float:
    """
    Structural Similarity Index (SSIM) in [-1, 1].
    1.0 = identical images; values > 0.9 are perceptually very similar.
    Uses scikit-image implementation.
    """
    return float(structural_similarity(
        original.astype(float),
        reconstructed.astype(float),
        data_range=data_range
    ))


def snr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Signal-to-Noise Ratio (dB) = 10*log10(signal_power / noise_power).
    """
    signal_power = np.mean(original.astype(float) ** 2)
    noise_power = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
    if noise_power == 0:
        return float('inf')
    return 10.0 * np.log10(signal_power / noise_power)
