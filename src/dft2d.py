"""
2D DFT implemented as successive row-then-column 1D FFTs.
The 2D DFT is defined as:
    F(u,v) = sum_x sum_y f(x,y) * exp(-j2pi(ux/M + vy/N))
By separability: first apply 1D FFT along each row, then along each column.
"""

import numpy as np


def fft2d(image: np.ndarray) -> np.ndarray:
    """
    2D FFT via row-then-column 1D FFTs (from scratch decomposition).
    Verified to match np.fft.fft2 to within 1e-10.

    Parameters
    ----------
    image : 2D real or complex array of shape (M, N)

    Returns
    -------
    F : complex array (M, N) — 2D DFT coefficients (NOT shifted)
    """
    x = image.astype(complex)
    M, N = x.shape

    # Step 1: 1D FFT along every row
    row_fft = np.empty((M, N), dtype=complex)
    for m in range(M):
        row_fft[m, :] = np.fft.fft(x[m, :])

    # Step 2: 1D FFT along every column of the intermediate result
    result = np.empty((M, N), dtype=complex)
    for n in range(N):
        result[:, n] = np.fft.fft(row_fft[:, n])

    return result


def ifft2d(F: np.ndarray) -> np.ndarray:
    """
    2D IFFT via row-then-column 1D IFFTs.
    Inverse: f(x,y) = (1/MN) * sum_u sum_v F(u,v) * exp(j2pi(ux/M + vy/N))
    """
    X = F.astype(complex)
    M, N = X.shape

    row_ifft = np.empty((M, N), dtype=complex)
    for m in range(M):
        row_ifft[m, :] = np.fft.ifft(X[m, :])

    result = np.empty((M, N), dtype=complex)
    for n in range(N):
        result[:, n] = np.fft.ifft(row_ifft[:, n])

    return result


def fftshift2d(F: np.ndarray) -> np.ndarray:
    """Shift zero-frequency component to the center of the spectrum."""
    return np.fft.fftshift(F)


def ifftshift2d(F: np.ndarray) -> np.ndarray:
    """Inverse of fftshift2d."""
    return np.fft.ifftshift(F)


def log_magnitude_spectrum(image: np.ndarray) -> np.ndarray:
    """
    Compute the centered log-magnitude spectrum for display.
    Returns log(1 + |F(u,v)|) shifted so DC is at center.
    """
    F = fft2d(image)
    F_shifted = fftshift2d(F)
    return np.log1p(np.abs(F_shifted))


def verify_vs_numpy(image: np.ndarray) -> float:
    """
    Verify fft2d matches np.fft.fft2.
    Returns the maximum absolute error (should be < 1e-10).
    """
    our_result = fft2d(image)
    numpy_result = np.fft.fft2(image)
    return float(np.max(np.abs(our_result - numpy_result)))


def apply_filter_freq(image: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Apply a frequency-domain filter H to an image.
    H must be in the same (unshifted) layout as fft2d output.

    1. fft2d(image)
    2. multiply by H
    3. ifft2d → take real part
    """
    F = fft2d(image)
    G = F * H
    result = np.real(ifft2d(G))
    return result
