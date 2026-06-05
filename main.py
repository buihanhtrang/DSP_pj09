"""
Project 09 — 2D Image Processing & Video Motion Analysis
Main demo script: runs all components and saves figures to report/figures/.

Usage:
    cd proj09_2d_image
    python main.py

Requires:
    pip install numpy scipy matplotlib scikit-image Pillow
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless rendering for saving figures

# Make src importable when run from proj09_2d_image/
sys.path.insert(0, os.path.dirname(__file__))

from src.dft2d import fft2d, ifft2d, verify_vs_numpy, log_magnitude_spectrum
from src.filters2d import (
    ideal_lowpass, butterworth_lowpass, ideal_highpass,
    gaussian_highpass, notch_filter, apply_filter,
    motion_blur_psf, defocus_blur_psf, blur_image, add_gaussian_noise
)
from src.wiener import wiener_deconvolve, sweep_K, best_K_by_ssim, degrade_image
from src.texture import (generate_textures, extract_features,
                          leave_one_out_cv, CLASS_NAMES)
from src.optical_flow import horn_schunck, flow_magnitude
from src.video_analysis import generate_synthetic_video, process_video
from src.metrics import psnr, ssim_metric
from src.visualization import (
    filter_gallery, wiener_sweep_figure,
    texture_gallery, confusion_matrix_figure,
    flow_visualization, motion_magnitude_curve,
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'report', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def fig_path(name: str) -> str:
    return os.path.join(FIGURES_DIR, name)


# ---------------------------------------------------------------------------
# Load or generate a test image
# ---------------------------------------------------------------------------

def get_test_image(size=256) -> np.ndarray:
    try:
        from skimage import data
        img = data.camera()
        from skimage.transform import resize
        img = resize(img, (size, size), anti_aliasing=True)
        return img.astype(float) / img.max()
    except Exception:
        # Fallback: synthetic image with geometric shapes
        rng = np.random.default_rng(0)
        img = np.zeros((size, size))
        # Circles
        cx, cy = size // 2, size // 2
        for r in [size // 8, size // 4, size * 3 // 8]:
            for i in range(size):
                for j in range(size):
                    d = np.sqrt((i - cx)**2 + (j - cy)**2)
                    if abs(d - r) < 3:
                        img[i, j] = 1.0
        img += rng.normal(0, 0.02, img.shape)
        return np.clip(img, 0, 1)


# ---------------------------------------------------------------------------
# Step 1: DFT verification
# ---------------------------------------------------------------------------

def step1_verify_dft(image):
    err = verify_vs_numpy(image)
    print(f'[Step 1] 2D DFT max error vs numpy.fft.fft2: {err:.2e}')
    assert err < 1e-10, f'DFT error {err} exceeds tolerance!'
    print('         PASS - within 1e-10 tolerance.')


# ---------------------------------------------------------------------------
# Step 2: Filter gallery
# ---------------------------------------------------------------------------

def step2_filter_gallery(image):
    print('[Step 2] Generating filter gallery ...')
    fig = filter_gallery(image, save_path=fig_path('filter_gallery.png'))
    print('         Saved -> report/figures/filter_gallery.png')
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 3: Wiener deconvolution
# ---------------------------------------------------------------------------

def step3_wiener(image):
    print('[Step 3] Wiener deconvolution ...')
    psf = motion_blur_psf(image.shape, L=30, angle_deg=0)
    blurred, _ = degrade_image(image, psf, noise_sigma=0.02)

    K_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    results = sweep_K(blurred, psf, image, K_values)
    best_K = best_K_by_ssim(results)

    print(f'         K sweep results:')
    for K in K_values:
        r = results[K]
        marker = ' <- best' if K == best_K else ''
        print(f'           K={K:.0e}  SSIM={r["ssim"]:.4f}  PSNR={r["psnr"]:.1f} dB{marker}')

    fig = wiener_sweep_figure(image, blurred, results, best_K,
                               save_path=fig_path('wiener_deconvolution.png'))
    import matplotlib.pyplot as plt
    plt.close(fig)
    print('         Saved -> report/figures/wiener_deconvolution.png')


# ---------------------------------------------------------------------------
# Step 4: Texture analysis & KNN
# ---------------------------------------------------------------------------

def step4_texture():
    print('[Step 4] Texture analysis & KNN classification ...')
    images, labels = generate_textures(n_samples_per_class=6, size=128)

    # Gallery figure
    fig = texture_gallery(images, labels, CLASS_NAMES,
                          save_path=fig_path('texture_gallery.png'))
    import matplotlib.pyplot as plt
    plt.close(fig)

    # Extract features and classify
    features = extract_features(images, n_angle_bins=36, n_radial_bins=32)
    predictions, accuracy, confusion = leave_one_out_cv(features, labels, k=3)
    print(f'         LOO-CV accuracy (k=3): {accuracy:.1%}')
    print(f'         Confusion matrix:\n{confusion}')

    fig = confusion_matrix_figure(confusion, CLASS_NAMES, accuracy,
                                   save_path=fig_path('texture_confusion.png'))
    plt.close(fig)
    print('         Saved -> report/figures/texture_gallery.png, texture_confusion.png')


# ---------------------------------------------------------------------------
# Step 5: Horn-Schunck optical flow
# ---------------------------------------------------------------------------

def step5_optical_flow(image):
    print('[Step 5] Horn-Schunck optical flow ...')

    # Create a synthetic second frame by translating a crop
    rng = np.random.default_rng(7)
    shift = 5  # pixel translation
    frame1 = image.copy()
    frame2 = np.roll(image, shift, axis=1)   # shift right by `shift` px
    frame2 = add_gaussian_noise(frame2, 0.01, rng)

    u, v = horn_schunck(frame1, frame2, alpha=1.0, n_iter=150)

    mag = flow_magnitude(u, v)
    mean_mag = float(np.mean(mag))
    print(f'         Mean flow magnitude: {mean_mag:.3f} px')
    print(f'         Expected: ~{shift} px horizontal shift')

    fig = flow_visualization(frame1, frame2, u, v, step=8,
                              save_path=fig_path('optical_flow.png'))
    import matplotlib.pyplot as plt
    plt.close(fig)
    print('         Saved -> report/figures/optical_flow.png')


# ---------------------------------------------------------------------------
# Step 6: Video motion analysis
# ---------------------------------------------------------------------------

def step6_video():
    print('[Step 6] Video motion analysis (30-frame synthetic sequence) ...')
    frames = generate_synthetic_video(n_frames=30, size=128)
    results = process_video(frames, alpha=1.0, n_iter=80)

    magnitudes = results['magnitudes']
    max_frame = results['max_frame']
    print(f'         Max motion at frame pair {max_frame} '
          f'(magnitude={magnitudes[max_frame]:.3f})')

    fig = motion_magnitude_curve(magnitudes, max_frame,
                                  save_path=fig_path('motion_curve.png'))
    import matplotlib.pyplot as plt
    plt.close(fig)
    print('         Saved -> report/figures/motion_curve.png')

    # Also visualize the flow at max-motion frame
    u_max, v_max = results['flows'][max_frame]
    fig2 = flow_visualization(frames[max_frame], frames[max_frame + 1],
                               u_max, v_max, step=6,
                               save_path=fig_path('video_flow_maxframe.png'))
    plt.close(fig2)
    print('         Saved -> report/figures/video_flow_maxframe.png')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('=' * 60)
    print('Project 09 — 2D Image Processing & Video Motion Analysis')
    print('=' * 60)

    image = get_test_image(size=256)
    print(f'Test image: {image.shape[0]}×{image.shape[1]} px, '
          f'range [{image.min():.3f}, {image.max():.3f}]')
    print()

    step1_verify_dft(image)
    print()
    step2_filter_gallery(image)
    print()
    step3_wiener(image)
    print()
    step4_texture()
    print()
    step5_optical_flow(image)
    print()
    step6_video()
    print()
    print('All figures saved to report/figures/')
    print('Run  python -m src.gui  to launch the interactive GUI.')


if __name__ == '__main__':
    main()
