"""
Visualization helpers — all functions return matplotlib Figure objects
so they can be saved or embedded in the GUI.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm


# ---------------------------------------------------------------------------
# Spectrum display
# ---------------------------------------------------------------------------

def show_spectrum(image: np.ndarray, ax=None, title='Log Magnitude Spectrum'):
    """Display the log-magnitude spectrum of an image on ax."""
    from .dft2d import fft2d, fftshift2d
    F = fft2d(image)
    F_shifted = fftshift2d(F)
    spec = np.log1p(np.abs(F_shifted))

    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(spec, cmap='inferno', aspect='equal')
    ax.set_title(title)
    ax.axis('off')
    return ax


# ---------------------------------------------------------------------------
# Filter gallery — 5 filter types × 3 columns (original / spectrum / result)
# ---------------------------------------------------------------------------

def filter_gallery(image: np.ndarray, save_path: str = None) -> plt.Figure:
    """
    5 filter types × 3 columns:
      col 0: original image
      col 1: centered log-magnitude spectrum of filtered image
      col 2: filtered image

    Filter types: Ideal LP, Butterworth LP, Ideal HP, Gaussian HP, Notch.
    """
    from .dft2d import fft2d, fftshift2d
    from .filters2d import (ideal_lowpass, butterworth_lowpass,
                             ideal_highpass, gaussian_highpass,
                             notch_filter, apply_filter)

    shape = image.shape
    D0 = min(shape) * 0.15   # cutoff at 15% of smallest dimension

    filters = {
        'Ideal Lowpass\n(D₀={:.0f}px)'.format(D0): ideal_lowpass(shape, D0),
        'Butterworth LP\n(n=4, D₀={:.0f}px)'.format(D0): butterworth_lowpass(shape, D0, n=4),
        'Ideal Highpass\n(D₀={:.0f}px)'.format(D0): ideal_highpass(shape, D0),
        'Gaussian Highpass\n(D₀={:.0f}px)'.format(D0): gaussian_highpass(shape, D0),
        'Notch Filter\n(center removed)': notch_filter(shape, [(20, 0), (0, 20)], radius=8),
    }

    n_rows = len(filters)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, n_rows * 3))
    fig.suptitle('Filter Gallery: 5 Filter Types', fontsize=14, fontweight='bold')

    col_titles = ['Original Image', 'Filtered Spectrum\n(log magnitude)', 'Filtered Image']
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=10, fontweight='bold')

    for row, (name, H) in enumerate(filters.items()):
        filtered = apply_filter(image, H)
        F_filt = fft2d(filtered)
        spec_filt = np.log1p(np.abs(fftshift2d(F_filt)))

        axes[row, 0].imshow(image, cmap='gray', vmin=0, vmax=1)
        axes[row, 0].set_ylabel(name, fontsize=8, rotation=0, labelpad=80,
                                va='center')
        axes[row, 0].axis('off')

        axes[row, 1].imshow(spec_filt, cmap='inferno')
        axes[row, 1].axis('off')

        axes[row, 2].imshow(filtered, cmap='gray',
                            vmin=filtered.min(), vmax=filtered.max())
        axes[row, 2].axis('off')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ---------------------------------------------------------------------------
# Wiener deconvolution figure
# ---------------------------------------------------------------------------

def wiener_sweep_figure(original: np.ndarray, blurred: np.ndarray,
                        sweep_results: dict, best_K,
                        save_path: str = None) -> plt.Figure:
    """
    Show: original | blurred | K=k1 | K=k2 | ... | best K restoration.
    Bottom row: SSIM and PSNR vs. K.
    """
    K_values = sorted(sweep_results.keys())
    n_K = len(K_values)

    fig = plt.figure(figsize=(4 * (n_K + 2), 8))
    gs = gridspec.GridSpec(2, n_K + 2, figure=fig)
    fig.suptitle('Wiener Deconvolution — K Sweep', fontsize=13, fontweight='bold')

    # --- Top row: images ---
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(original, cmap='gray', vmin=0, vmax=1)
    ax.set_title('Original')
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(blurred, cmap='gray', vmin=0, vmax=1)
    ax.set_title('Blurred + Noise')
    ax.axis('off')

    for i, K in enumerate(K_values):
        ax = fig.add_subplot(gs[0, i + 2])
        res = sweep_results[K]['restored']
        ssim_v = sweep_results[K]['ssim']
        psnr_v = sweep_results[K]['psnr']
        border = 'gold' if K == best_K else 'none'
        ax.imshow(res, cmap='gray', vmin=0, vmax=1)
        title = f'K={K:.0e}\nSSIM={ssim_v:.3f}\nPSNR={psnr_v:.1f}dB'
        ax.set_title(title, fontsize=7)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_edgecolor(border)
            spine.set_linewidth(3 if K == best_K else 0)

    # --- Bottom row: metrics vs K ---
    ssims = [sweep_results[K]['ssim'] for K in K_values]
    psnrs = [sweep_results[K]['psnr'] for K in K_values]
    K_log = [np.log10(K) for K in K_values]

    ax_ssim = fig.add_subplot(gs[1, :])
    ax_psnr = ax_ssim.twinx()
    ax_ssim.semilogx(K_values, ssims, 'b-o', label='SSIM')
    ax_psnr.semilogx(K_values, psnrs, 'r-s', label='PSNR (dB)')
    ax_ssim.axvline(best_K, color='gold', linestyle='--', label=f'Best K={best_K:.0e}')
    ax_ssim.set_xlabel('K (log scale)')
    ax_ssim.set_ylabel('SSIM', color='b')
    ax_psnr.set_ylabel('PSNR (dB)', color='r')
    ax_ssim.set_title('Quality Metrics vs. Regularization K')
    lines1, labels1 = ax_ssim.get_legend_handles_labels()
    lines2, labels2 = ax_psnr.get_legend_handles_labels()
    ax_ssim.legend(lines1 + lines2, labels1 + labels2, loc='lower left')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ---------------------------------------------------------------------------
# Texture analysis
# ---------------------------------------------------------------------------

def texture_gallery(images: list, labels: list,
                    class_names: list, save_path: str = None) -> plt.Figure:
    """Show one example and power spectrum per texture class."""
    from .texture import power_spectrum_2d
    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, 2, figsize=(6, n_classes * 2.5))
    fig.suptitle('Texture Classes: Sample & Power Spectrum', fontsize=12)

    for cls in range(n_classes):
        idx = labels.index(cls)
        img = images[idx]
        spec = power_spectrum_2d(img)

        axes[cls, 0].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[cls, 0].set_title(class_names[cls])
        axes[cls, 0].axis('off')

        axes[cls, 1].imshow(np.log1p(spec), cmap='inferno')
        axes[cls, 1].set_title('Log Power Spectrum')
        axes[cls, 1].axis('off')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def confusion_matrix_figure(confusion: np.ndarray, class_names: list,
                             accuracy: float, save_path: str = None) -> plt.Figure:
    """Plot KNN confusion matrix as a color grid with counts."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion, cmap='Blues')
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'KNN Confusion Matrix — LOO Accuracy: {accuracy:.1%}')
    plt.colorbar(im, ax=ax)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(confusion[i, j]),
                    ha='center', va='center', fontsize=12,
                    color='white' if confusion[i, j] > confusion.max() / 2 else 'black')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ---------------------------------------------------------------------------
# Optical flow
# ---------------------------------------------------------------------------

def flow_visualization(frame1: np.ndarray, frame2: np.ndarray,
                        u: np.ndarray, v: np.ndarray,
                        step: int = 8, save_path: str = None) -> plt.Figure:
    """
    Show frame1, frame2, quiver flow plot, and color-coded magnitude map.
    """
    from .optical_flow import flow_magnitude, flow_to_color

    mag = flow_magnitude(u, v)
    color_flow = flow_to_color(u, v)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.suptitle('Horn-Schunck Optical Flow', fontsize=13, fontweight='bold')

    axes[0, 0].imshow(frame1, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Frame 1')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(frame2, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Frame 2')
    axes[0, 1].axis('off')

    # Quiver plot — subsample by `step` for readability
    M, N = u.shape
    Y, X = np.mgrid[0:M:step, 0:N:step]
    U_sub = u[::step, ::step]
    V_sub = v[::step, ::step]

    axes[1, 0].imshow(frame1, cmap='gray', vmin=0, vmax=1, alpha=0.6)
    axes[1, 0].quiver(X, Y, U_sub, -V_sub,
                      color='yellow', scale=None, scale_units='xy',
                      angles='xy', width=0.003)
    axes[1, 0].set_title('Flow Field (Quiver)')
    axes[1, 0].axis('off')

    im = axes[1, 1].imshow(mag, cmap='hot')
    axes[1, 1].set_title('Motion Magnitude Map')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def motion_magnitude_curve(magnitudes: list, max_frame: int,
                            save_path: str = None) -> plt.Figure:
    """Plot mean optical flow magnitude per frame pair vs. frame index."""
    fig, ax = plt.subplots(figsize=(9, 4))
    frames = list(range(len(magnitudes)))
    ax.plot(frames, magnitudes, 'b-o', markersize=4, label='Mean magnitude')
    ax.axvline(max_frame, color='red', linestyle='--',
               label=f'Max motion at frame {max_frame}')
    ax.fill_between(frames, magnitudes, alpha=0.2)
    ax.set_xlabel('Frame pair index')
    ax.set_ylabel('Mean flow magnitude (pixels/frame)')
    ax.set_title('Video Motion Magnitude Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
