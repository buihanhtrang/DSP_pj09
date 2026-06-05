"""
Texture analysis via 2D power spectrum and KNN classification.

Pipeline:
  1. Generate (or load) texture images — 4 classes, multiple samples each.
  2. Compute 2D power spectrum = |FFT2D|^2.
  3. Extract directional energy by integrating over angular bins.
  4. Build feature vectors from directional energy histograms.
  5. Classify with K-Nearest Neighbors (leave-one-out cross-validation).
"""

import numpy as np
from .dft2d import fft2d, fftshift2d


# ---------------------------------------------------------------------------
# Synthetic texture generators
# ---------------------------------------------------------------------------

def _make_horizontal_stripes(size=128, freq=8, thickness_ratio=0.5, rng=None):
    """Sinusoidal horizontal stripes — energy concentrated on vertical axis."""
    if rng is None:
        rng = np.random.default_rng()
    x = np.arange(size)
    stripe = (np.sin(2 * np.pi * freq * x / size) > 0).astype(float)
    img = np.tile(stripe[:, np.newaxis], (1, size))
    img += rng.normal(0, 0.05, img.shape)
    return np.clip(img, 0, 1)


def _make_vertical_stripes(size=128, freq=8, rng=None):
    """Sinusoidal vertical stripes — energy concentrated on horizontal axis."""
    if rng is None:
        rng = np.random.default_rng()
    x = np.arange(size)
    stripe = (np.sin(2 * np.pi * freq * x / size) > 0).astype(float)
    img = np.tile(stripe[np.newaxis, :], (size, 1))
    img += rng.normal(0, 0.05, img.shape)
    return np.clip(img, 0, 1)


def _make_diagonal_stripes(size=128, freq=8, rng=None):
    """45-degree diagonal stripes — energy along the diagonal axis."""
    if rng is None:
        rng = np.random.default_rng()
    x = np.arange(size)
    X, Y = np.meshgrid(x, x)
    img = (np.sin(2 * np.pi * freq * (X + Y) / size) > 0).astype(float)
    img += rng.normal(0, 0.05, img.shape)
    return np.clip(img, 0, 1)


def _make_checkerboard(size=128, freq=8, rng=None):
    """Checkerboard pattern — energy at 4 corners of the spectrum."""
    if rng is None:
        rng = np.random.default_rng()
    x = np.arange(size)
    X, Y = np.meshgrid(x, x)
    img = ((np.floor(freq * X / size) + np.floor(freq * Y / size)) % 2).astype(float)
    img += rng.normal(0, 0.05, img.shape)
    return np.clip(img, 0, 1)


CLASS_NAMES = ['Horizontal', 'Vertical', 'Diagonal', 'Checkerboard']
_GENERATORS = [
    _make_horizontal_stripes,
    _make_vertical_stripes,
    _make_diagonal_stripes,
    _make_checkerboard,
]


def generate_textures(n_samples_per_class: int = 6,
                      size: int = 128) -> tuple:
    """
    Generate synthetic texture dataset.

    Returns
    -------
    images : list of 2D arrays
    labels : list of int class indices
    """
    images, labels = [], []
    rng = np.random.default_rng(0)
    freqs = [6, 8, 10, 12, 14, 16]

    for class_idx, gen in enumerate(_GENERATORS):
        for i in range(n_samples_per_class):
            freq = freqs[i % len(freqs)]
            img = gen(size=size, freq=freq, rng=rng)
            images.append(img)
            labels.append(class_idx)

    return images, labels


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def power_spectrum_2d(image: np.ndarray) -> np.ndarray:
    """
    2D power spectrum = |F(u,v)|^2, centered (DC at center).
    """
    F = fft2d(image)
    F_shifted = fftshift2d(F)
    return np.abs(F_shifted) ** 2


def directional_energy(spectrum: np.ndarray, n_bins: int = 36) -> np.ndarray:
    """
    Integrate the 2D power spectrum over angular bins to get directional energy.

    Divides [0, pi) into n_bins equal angular sectors and sums all spectrum
    values whose angle (from DC) falls in each sector.

    Parameters
    ----------
    spectrum : 2D centered power spectrum (DC at center)
    n_bins   : number of angular bins (default 36 → 5° each)

    Returns
    -------
    energy : 1D array of length n_bins, L2-normalized
    """
    M, N = spectrum.shape
    u = np.arange(M) - M // 2
    v = np.arange(N) - N // 2
    V, U = np.meshgrid(v, u)

    # Angle in [0, pi) — we fold negative angles (spectral symmetry)
    angles = np.arctan2(U, V) % np.pi  # shape (M, N)

    bin_edges = np.linspace(0, np.pi, n_bins + 1)
    energy = np.zeros(n_bins)

    for b in range(n_bins):
        mask = (angles >= bin_edges[b]) & (angles < bin_edges[b + 1])
        # Exclude DC
        r = np.sqrt(U**2 + V**2)
        mask &= (r > 2)
        energy[b] = spectrum[mask].sum()

    norm = np.linalg.norm(energy)
    if norm > 0:
        energy /= norm
    return energy


def radial_energy(spectrum: np.ndarray, n_bins: int = 32) -> np.ndarray:
    """
    Integrate power spectrum over radial (frequency) bins.
    Captures how energy is distributed across spatial frequencies.
    """
    M, N = spectrum.shape
    u = np.arange(M) - M // 2
    v = np.arange(N) - N // 2
    V, U = np.meshgrid(v, u)
    R = np.sqrt(U**2 + V**2)

    max_r = np.sqrt((M // 2)**2 + (N // 2)**2)
    bin_edges = np.linspace(0, max_r, n_bins + 1)
    energy = np.zeros(n_bins)

    for b in range(n_bins):
        mask = (R >= bin_edges[b]) & (R < bin_edges[b + 1])
        energy[b] = spectrum[mask].sum()

    norm = np.linalg.norm(energy)
    if norm > 0:
        energy /= norm
    return energy


def extract_features(images: list, n_angle_bins: int = 36,
                     n_radial_bins: int = 32) -> np.ndarray:
    """
    Extract feature vectors from a list of texture images.
    Feature = concatenation of directional energy + radial energy.

    Returns
    -------
    features : array of shape (n_images, n_angle_bins + n_radial_bins)
    """
    features = []
    for img in images:
        spec = power_spectrum_2d(img)
        dir_e = directional_energy(spec, n_angle_bins)
        rad_e = radial_energy(spec, n_radial_bins)
        features.append(np.concatenate([dir_e, rad_e]))
    return np.array(features)


# ---------------------------------------------------------------------------
# KNN classifier
# ---------------------------------------------------------------------------

def knn_predict(train_X: np.ndarray, train_y: list,
                test_x: np.ndarray, k: int = 3) -> int:
    """Predict class for a single test sample using k-NN (Euclidean distance)."""
    dists = np.linalg.norm(train_X - test_x, axis=1)
    nearest_k = np.argsort(dists)[:k]
    neighbor_labels = [train_y[i] for i in nearest_k]
    # Majority vote
    counts = np.bincount(neighbor_labels, minlength=max(train_y) + 1)
    return int(np.argmax(counts))


def leave_one_out_cv(features: np.ndarray, labels: list,
                     k: int = 3) -> tuple:
    """
    Leave-one-out cross-validation for KNN.

    Returns
    -------
    predictions : list of predicted labels
    accuracy    : float in [0, 1]
    confusion   : 2D array — confusion[true, pred]
    """
    n = len(labels)
    n_classes = len(set(labels))
    predictions = []
    confusion = np.zeros((n_classes, n_classes), dtype=int)

    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        train_X = features[train_idx]
        train_y = [labels[j] for j in train_idx]
        pred = knn_predict(train_X, train_y, features[i], k)
        predictions.append(pred)
        confusion[labels[i], pred] += 1

    accuracy = np.trace(confusion) / confusion.sum()
    return predictions, float(accuracy), confusion
