"""
Verification tests for the 2D DFT implementation.
Run from proj09_2d_image/:
    python -m pytest tests/verify_dft.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.dft2d import fft2d, ifft2d, verify_vs_numpy


# ---------------------------------------------------------------------------
# 2D DFT accuracy tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('shape', [(32, 32), (64, 128), (128, 64), (256, 256)])
def test_fft2d_matches_numpy(shape):
    """fft2d must match np.fft.fft2 to within 1e-10 for various sizes."""
    rng = np.random.default_rng(0)
    image = rng.random(shape)
    err = verify_vs_numpy(image)
    assert err < 1e-10, f'Max error {err:.2e} exceeds 1e-10 for shape {shape}'


def test_ifft2d_inverts_fft2d():
    """ifft2d(fft2d(x)) == x to within 1e-10."""
    rng = np.random.default_rng(1)
    x = rng.random((64, 64))
    err = np.max(np.abs(np.real(ifft2d(fft2d(x))) - x))
    assert err < 1e-10, f'Reconstruction error {err:.2e} exceeds 1e-10'


def test_fft2d_linearity():
    """F[a*x + b*y] = a*F[x] + b*F[y]."""
    rng = np.random.default_rng(2)
    x = rng.random((64, 64))
    y = rng.random((64, 64))
    a, b = 2.0, -0.5
    err = np.max(np.abs(fft2d(a * x + b * y) - (a * fft2d(x) + b * fft2d(y))))
    assert err < 1e-10, f'Linearity error {err:.2e}'


def test_fft2d_parseval():
    """Parseval: sum|f|^2 = (1/MN) * sum|F|^2."""
    rng = np.random.default_rng(3)
    x = rng.random((64, 64))
    M, N = x.shape
    spatial_energy = np.sum(x**2)
    spectral_energy = np.sum(np.abs(fft2d(x))**2) / (M * N)
    assert abs(spatial_energy - spectral_energy) / spatial_energy < 1e-8


def test_fft2d_dc_is_sum():
    """F[0,0] = sum of all pixel values (DC component)."""
    rng = np.random.default_rng(4)
    x = rng.random((32, 32))
    dc = fft2d(x)[0, 0]
    expected = x.sum()
    assert abs(dc - expected) < 1e-8, f'DC={dc:.4f} vs sum={expected:.4f}'


def test_fft2d_impulse():
    """FFT of a unit impulse at origin = all-ones."""
    x = np.zeros((32, 32))
    x[0, 0] = 1.0
    F = fft2d(x)
    assert np.allclose(np.abs(F), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Filter tests
# ---------------------------------------------------------------------------

def test_ideal_lowpass_passes_dc():
    """After ideal lowpass, DC component should be preserved."""
    from src.filters2d import ideal_lowpass, apply_filter
    rng = np.random.default_rng(5)
    image = rng.random((64, 64))
    H = ideal_lowpass((64, 64), D0=30)
    filtered = apply_filter(image, H)
    # Mean should be close to original (DC preserved)
    assert abs(filtered.mean() - image.mean()) < 1e-6


def test_ideal_highpass_removes_dc():
    """After ideal highpass, the mean (DC) should be near zero."""
    from src.filters2d import ideal_highpass, apply_filter
    rng = np.random.default_rng(6)
    image = rng.random((64, 64))
    H = ideal_highpass((64, 64), D0=1)
    filtered = apply_filter(image, H)
    assert abs(filtered.mean()) < 0.01


def test_butterworth_order_convergence():
    """
    Higher Butterworth order -> sharper roll-off.
    Check: passband (D < 0.5*D0) is flat near 1, stopband (D > 2*D0) is near 0.
    Note: at exactly D=D0, Butterworth always gives H=0.5 by definition.
    """
    from src.filters2d import butterworth_lowpass, distance_matrix
    shape = (128, 128)
    D0 = 30
    D = distance_matrix(shape)
    H_bw = butterworth_lowpass(shape, D0, n=10)

    passband = H_bw[D < D0 * 0.5]
    stopband = H_bw[D > D0 * 2.0]

    assert passband.min() > 0.95, f'Passband min={passband.min():.3f} < 0.95'
    assert stopband.max() < 0.05, f'Stopband max={stopband.max():.3f} > 0.05'


# ---------------------------------------------------------------------------
# Wiener deconvolution test
# ---------------------------------------------------------------------------

def test_wiener_k0_is_inverse_filter():
    """With K≈0, Wiener should approximately recover the original."""
    from src.filters2d import motion_blur_psf
    from src.wiener import degrade_image, wiener_deconvolve
    from src.metrics import psnr

    rng = np.random.default_rng(7)
    image = np.random.default_rng(0).random((64, 64))
    psf = motion_blur_psf((64, 64), L=5)
    blurred, _ = degrade_image(image, psf, noise_sigma=0.0, rng=rng)
    restored = wiener_deconvolve(blurred, psf, K=1e-9)
    p = psnr(image, restored)
    # Near-inverse filter on noiseless blur should give high PSNR
    assert p > 25, f'Wiener K≈0 PSNR={p:.1f} dB too low'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
