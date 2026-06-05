"""
Horn-Schunck dense optical flow algorithm.

Minimizes the energy functional:
    E(u,v) = ∫∫ [(Ix*u + Iy*v + It)^2 + alpha^2 * (|∇u|^2 + |∇v|^2)] dxdy

where:
  Ix, Iy = spatial image gradients (x and y directions)
  It      = temporal gradient (frame2 - frame1)
  u, v    = horizontal and vertical flow components
  alpha   = smoothness regularization weight

Euler-Lagrange equations lead to the iterative update:
  u^(n+1) = ū^n - Ix * (Ix*ū^n + Iy*v̄^n + It) / (alpha^2 + Ix^2 + Iy^2)
  v^(n+1) = v̄^n - Iy * (Ix*ū^n + Iy*v̄^n + It) / (alpha^2 + Ix^2 + Iy^2)

where ū, v̄ are local spatial averages using the 3×3 averaging kernel:
  K = [[1/12, 1/6, 1/12],
       [1/6,   0,  1/6],
       [1/12, 1/6, 1/12]]
"""

import numpy as np
from scipy.ndimage import convolve


# Horn-Schunck averaging kernel (excludes center pixel)
_HS_KERNEL = np.array([[1/12, 1/6, 1/12],
                        [1/6,  0.0, 1/6],
                        [1/12, 1/6, 1/12]], dtype=float)


def _compute_gradients(frame1: np.ndarray,
                       frame2: np.ndarray) -> tuple:
    """
    Compute spatial (Ix, Iy) and temporal (It) gradients.

    Uses the 2×2 block finite-difference scheme from the original
    Horn & Schunck (1981) paper, padded back to original size.

    Ix = d/dx of average of frame1 and frame2
    Iy = d/dy of average of frame1 and frame2
    It = frame2 - frame1 (temporal difference)
    """
    f1 = frame1.astype(float)
    f2 = frame2.astype(float)

    # Spatial gradients on the average of the two frames
    avg = (f1 + f2) / 2.0
    Ix = np.gradient(avg, axis=1)   # d/dx (horizontal)
    Iy = np.gradient(avg, axis=0)   # d/dy (vertical)
    It = f2 - f1                    # temporal

    return Ix, Iy, It


def horn_schunck(frame1: np.ndarray, frame2: np.ndarray,
                 alpha: float = 1.0,
                 n_iter: int = 100) -> tuple:
    """
    Horn-Schunck dense optical flow.

    Parameters
    ----------
    frame1  : first frame, grayscale float in [0,1], shape (M, N)
    frame2  : second frame, same shape
    alpha   : smoothness weight. Larger alpha → smoother flow field.
              Typical range: 0.5 – 10. Use larger values for noisy images.
    n_iter  : number of Gauss-Seidel iterations (100–200 is usually enough)

    Returns
    -------
    u : horizontal flow component, shape (M, N)
    v : vertical flow component, shape (M, N)
    """
    Ix, Iy, It = _compute_gradients(frame1, frame2)

    M, N = frame1.shape
    u = np.zeros((M, N), dtype=float)
    v = np.zeros((M, N), dtype=float)

    alpha_sq = alpha ** 2
    denom = alpha_sq + Ix**2 + Iy**2  # constant across iterations

    for _ in range(n_iter):
        # Local averages using HS kernel
        u_bar = convolve(u, _HS_KERNEL, mode='reflect')
        v_bar = convolve(v, _HS_KERNEL, mode='reflect')

        # Common numerator term
        num = Ix * u_bar + Iy * v_bar + It

        u = u_bar - Ix * num / denom
        v = v_bar - Iy * num / denom

    return u, v


def flow_magnitude(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Point-wise magnitude of the flow vector: sqrt(u^2 + v^2)."""
    return np.sqrt(u**2 + v**2)


def mean_flow_magnitude(u: np.ndarray, v: np.ndarray) -> float:
    """Mean magnitude over the entire flow field."""
    return float(np.mean(flow_magnitude(u, v)))


def flow_to_color(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Encode optical flow as a color image using HSV color wheel.
    Hue encodes direction; saturation is always 1; value encodes magnitude.

    Returns
    -------
    rgb : uint8 array of shape (M, N, 3)
    """
    import matplotlib.colors as mcolors

    mag = flow_magnitude(u, v)
    angle = np.arctan2(-v, u)  # angle in [-pi, pi]

    # Normalize hue to [0, 1] from angle in [-pi, pi]
    hue = (angle + np.pi) / (2 * np.pi)

    # Normalize magnitude to [0, 1]
    max_mag = mag.max()
    val = mag / max_mag if max_mag > 0 else mag

    M, N = u.shape
    hsv = np.stack([hue, np.ones((M, N)), val], axis=-1)
    rgb = mcolors.hsv_to_rgb(hsv)
    return (rgb * 255).astype(np.uint8)
