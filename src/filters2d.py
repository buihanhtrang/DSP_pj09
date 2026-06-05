"""
2D frequency-domain filters:
  - Ideal lowpass
  - Butterworth lowpass (order n)
  - Ideal highpass
  - Gaussian highpass
  - 2D notch filter
  - Motion blur PSF (horizontal, length L)
  - Defocus blur PSF (disk of radius r)

All filter transfer functions H(u,v) are returned in the UNSHIFTED
frequency layout (DC at corner), matching the output of fft2d().
"""

import numpy as np
from scipy.ndimage import convolve


# ---------------------------------------------------------------------------
# Helper: distance matrix (DC at center, then unshifted)
# ---------------------------------------------------------------------------

def _distance_matrix_centered(shape: tuple) -> np.ndarray:
    """
    Returns D(u,v) = sqrt((u - M/2)^2 + (v - N/2)^2)
    for a centered spectrum (DC at center).
    """
    M, N = shape
    u = np.arange(M) - M // 2
    v = np.arange(N) - N // 2
    V, U = np.meshgrid(v, u)
    return np.sqrt(U**2 + V**2)


def distance_matrix(shape: tuple) -> np.ndarray:
    """
    Distance matrix in UNSHIFTED layout (DC at corner [0,0]).
    Use this to build H(u,v) that multiplies fft2d() output directly.
    """
    D_centered = _distance_matrix_centered(shape)
    return np.fft.ifftshift(D_centered)


# ---------------------------------------------------------------------------
# Filters — all return H of shape `shape` in unshifted layout
# ---------------------------------------------------------------------------

def ideal_lowpass(shape: tuple, D0: float) -> np.ndarray:
    """
    Ideal (brick-wall) lowpass: H = 1 where D(u,v) <= D0, else 0.
    D0 is the cutoff frequency in pixels (distance from DC).
    """
    D = distance_matrix(shape)
    H = (D <= D0).astype(float)
    return H


def butterworth_lowpass(shape: tuple, D0: float, n: int) -> np.ndarray:
    """
    Butterworth lowpass of order n:
        H(u,v) = 1 / (1 + (D(u,v)/D0)^(2n))
    Smooth roll-off; n=1 is maximally flat, larger n approaches ideal.
    """
    D = distance_matrix(shape)
    # Avoid division by zero at DC (D=0 → H=1 which is correct)
    with np.errstate(divide='ignore', invalid='ignore'):
        H = 1.0 / (1.0 + (D / D0) ** (2 * n))
    H[D == 0] = 1.0
    return H


def ideal_highpass(shape: tuple, D0: float) -> np.ndarray:
    """
    Ideal highpass: complement of ideal lowpass.
        H_hp = 1 - H_lp
    """
    return 1.0 - ideal_lowpass(shape, D0)


def gaussian_lowpass(shape: tuple, D0: float) -> np.ndarray:
    """
    Gaussian lowpass:
        H(u,v) = exp(-D(u,v)^2 / (2*D0^2))
    No ringing artifacts.
    """
    D = distance_matrix(shape)
    return np.exp(-(D**2) / (2.0 * D0**2))


def gaussian_highpass(shape: tuple, D0: float) -> np.ndarray:
    """
    Gaussian highpass: complement of Gaussian lowpass.
        H_hp = 1 - exp(-D^2 / (2*D0^2))
    """
    return 1.0 - gaussian_lowpass(shape, D0)


def notch_filter(shape: tuple, notch_centers: list, radius: float = 10.0) -> np.ndarray:
    """
    2D notch filter: rejects narrow frequency bands around specified centers.
    notch_centers: list of (u_center, v_center) in CENTERED coordinates
                   (origin at image center).
    For each notch at (u0, v0), also places a symmetric notch at (-u0, -v0)
    to preserve real-valued output (conjugate symmetry).

    The filter zeros out a disk of given radius around each notch center.
    """
    M, N = shape
    D_centers = _distance_matrix_centered(shape)

    H = np.ones(shape, dtype=float)
    for (u0, v0) in notch_centers:
        u = np.arange(M) - M // 2
        v = np.arange(N) - N // 2
        V, U = np.meshgrid(v, u)
        # Distance from (u0, v0)
        D1 = np.sqrt((U - u0)**2 + (V - v0)**2)
        # Symmetric notch at (-u0, -v0)
        D2 = np.sqrt((U + u0)**2 + (V + v0)**2)
        H[D1 <= radius] = 0.0
        H[D2 <= radius] = 0.0

    # Convert from centered to unshifted layout
    return np.fft.ifftshift(H)


# ---------------------------------------------------------------------------
# Point Spread Functions (PSFs) for blur simulation
# ---------------------------------------------------------------------------

def motion_blur_psf(shape: tuple, L: int, angle_deg: float = 0.0) -> np.ndarray:
    """
    Horizontal motion blur PSF of length L pixels.
    angle_deg rotates the blur direction (0 = horizontal).
    Returns a normalized PSF in spatial domain (sum = 1).
    PSF is placed at the top-left corner (unshifted) for use with fft2d.
    """
    M, N = shape
    psf = np.zeros(shape)

    angle_rad = np.deg2rad(angle_deg)
    cx, cy = M // 2, N // 2

    for i in range(L):
        offset = i - L // 2
        row = cx + int(round(offset * np.sin(angle_rad)))
        col = cy + int(round(offset * np.cos(angle_rad)))
        if 0 <= row < M and 0 <= col < N:
            psf[row, col] = 1.0

    if psf.sum() > 0:
        psf /= psf.sum()

    # Shift PSF so center is at [0,0] for use with FFT
    psf = np.fft.ifftshift(psf)
    return psf


def defocus_blur_psf(shape: tuple, r: int) -> np.ndarray:
    """
    Out-of-focus (defocus) blur PSF: uniform disk of radius r.
    Returns a normalized PSF (sum = 1) in unshifted layout.
    """
    M, N = shape
    psf = np.zeros(shape)
    cx, cy = M // 2, N // 2

    for i in range(M):
        for j in range(N):
            if (i - cx)**2 + (j - cy)**2 <= r**2:
                psf[i, j] = 1.0

    if psf.sum() > 0:
        psf /= psf.sum()

    psf = np.fft.ifftshift(psf)
    return psf


def psf_to_otf(psf: np.ndarray) -> np.ndarray:
    """Convert a PSF (spatial domain, unshifted) to OTF (frequency domain)."""
    return np.fft.fft2(psf)


# ---------------------------------------------------------------------------
# Apply filter or PSF to an image
# ---------------------------------------------------------------------------

def apply_filter(image: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Filter an image in the frequency domain.
    H: transfer function in unshifted layout (same shape as image).
    Returns real-valued filtered image.
    """
    from .dft2d import fft2d, ifft2d
    F = fft2d(image)
    G = F * H
    return np.real(ifft2d(G))


def blur_image(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """
    Blur an image by convolving with a PSF (via frequency-domain multiplication).
    """
    H = psf_to_otf(psf)
    return apply_filter(image, H)


def add_gaussian_noise(image: np.ndarray, sigma: float, rng=None) -> np.ndarray:
    """Add zero-mean Gaussian noise with standard deviation sigma."""
    if rng is None:
        rng = np.random.default_rng(42)
    noise = rng.normal(0, sigma, image.shape)
    return np.clip(image + noise, 0, 1)
