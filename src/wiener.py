"""
Wiener deconvolution for image restoration.

Given a blurred (and possibly noisy) image g and the known PSF h:
    G(u,v) = H(u,v) * F(u,v) + N(u,v)

The Wiener filter estimates F by:
    F_hat(u,v) = [H*(u,v) / (|H(u,v)|^2 + K)] * G(u,v)

where K = noise-to-signal power ratio (regularization parameter).
K = 0 gives the inverse filter (amplifies noise);
K → ∞ gives a zero output.
Optimal K minimizes MSE between F_hat and F.
In practice we select K by maximizing SSIM against a reference.
"""

import numpy as np
from .filters2d import psf_to_otf, add_gaussian_noise, blur_image
from .metrics import ssim_metric, psnr


def wiener_deconvolve(blurred: np.ndarray, psf: np.ndarray, K: float) -> np.ndarray:
    """
    Wiener deconvolution.

    Parameters
    ----------
    blurred : 2D array, degraded image in [0, 1]
    psf     : 2D array, same shape as blurred, in unshifted layout
    K       : regularization constant (noise-to-signal ratio proxy)

    Returns
    -------
    restored : 2D array, restored image clipped to [0, 1]
    """
    G = np.fft.fft2(blurred)
    H = psf_to_otf(psf)

    H_conj = np.conj(H)
    H_mag2 = np.abs(H) ** 2

    # Wiener filter transfer function
    W = H_conj / (H_mag2 + K)

    F_hat = W * G
    restored = np.real(np.fft.ifft2(F_hat))
    return np.clip(restored, 0.0, 1.0)


def sweep_K(blurred: np.ndarray, psf: np.ndarray, original: np.ndarray,
            K_values: list) -> dict:
    """
    Sweep K over a list of values and evaluate each restoration by SSIM and PSNR.

    Returns
    -------
    results : dict with keys = K values, values = dict(ssim, psnr, restored)
    """
    results = {}
    for K in K_values:
        restored = wiener_deconvolve(blurred, psf, K)
        s = ssim_metric(original, restored)
        p = psnr(original, restored)
        results[K] = {'ssim': s, 'psnr': p, 'restored': restored}
    return results


def best_K_by_ssim(results: dict):
    """Return the K value that maximises SSIM."""
    return max(results, key=lambda k: results[k]['ssim'])


def degrade_image(image: np.ndarray, psf: np.ndarray,
                  noise_sigma: float = 0.01, rng=None):
    """
    Convenience: blur an image with psf then add Gaussian noise.
    Returns (blurred_noisy, blurred_clean).
    """
    blurred = blur_image(image, psf)
    blurred = np.clip(blurred, 0, 1)
    noisy = add_gaussian_noise(blurred, noise_sigma, rng)
    return noisy, blurred
