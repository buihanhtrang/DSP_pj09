"""
Video motion analysis using Horn-Schunck optical flow.

Pipeline:
  1. Load or generate a sequence of grayscale frames.
  2. Compute optical flow between consecutive frame pairs.
  3. Compute mean motion magnitude per frame.
  4. Detect the frame with maximum motion.
  5. Visualize the motion magnitude curve.
"""

import numpy as np
from .optical_flow import horn_schunck, mean_flow_magnitude


def generate_synthetic_video(n_frames: int = 30,
                              size: int = 128,
                              rng=None) -> list:
    """
    Generate a synthetic grayscale video with a moving rectangle.

    A white rectangle translates horizontally across a dark background,
    creating a clear optical flow signal.

    Returns
    -------
    frames : list of n_frames 2D arrays in [0, 1]
    """
    if rng is None:
        rng = np.random.default_rng(42)

    frames = []
    rect_h, rect_w = size // 4, size // 4
    rect_y = size // 2 - rect_h // 2  # vertical center

    for i in range(n_frames):
        frame = np.zeros((size, size), dtype=float)

        # Rectangle moves from left to right
        progress = i / max(n_frames - 1, 1)
        rect_x = int(progress * (size - rect_w))

        r1, r2 = rect_y, rect_y + rect_h
        c1, c2 = rect_x, rect_x + rect_w
        frame[r1:r2, c1:c2] = 1.0

        # Add mild background texture and noise
        x = np.linspace(0, 4 * np.pi, size)
        X, Y = np.meshgrid(x, x)
        frame += 0.05 * np.sin(X) * np.cos(Y)
        frame += rng.normal(0, 0.02, frame.shape)
        frame = np.clip(frame, 0, 1)
        frames.append(frame)

    return frames


def process_video(frames: list,
                  alpha: float = 1.0,
                  n_iter: int = 100) -> dict:
    """
    Run Horn-Schunck optical flow on all consecutive frame pairs.

    Parameters
    ----------
    frames : list of 2D grayscale frames
    alpha  : HS smoothness parameter
    n_iter : HS iteration count

    Returns
    -------
    results : dict with keys:
        'flows'      : list of (u, v) tuples, length n_frames-1
        'magnitudes' : list of mean magnitude per frame pair
        'max_frame'  : index of the frame pair with maximum motion
    """
    flows = []
    magnitudes = []

    n = len(frames)
    for i in range(n - 1):
        u, v = horn_schunck(frames[i], frames[i + 1],
                             alpha=alpha, n_iter=n_iter)
        flows.append((u, v))
        magnitudes.append(mean_flow_magnitude(u, v))

    max_frame = int(np.argmax(magnitudes))
    return {
        'flows': flows,
        'magnitudes': magnitudes,
        'max_frame': max_frame,
    }


def load_frames_from_files(paths: list) -> list:
    """
    Load grayscale frames from a list of image file paths.

    Returns list of float arrays in [0, 1].
    """
    from PIL import Image
    frames = []
    for p in paths:
        img = Image.open(p).convert('L')
        arr = np.array(img, dtype=float) / 255.0
        frames.append(arr)
    return frames
