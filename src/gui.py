"""
Interactive GUI for 2D image processing.

Layout (2×2 grid):
  [Original Image]  |  [Frequency Spectrum]
  [Filtered Spectrum]  |  [Filtered Image]

Controls:
  - Load image button (opens file dialog)
  - Filter type selector (dropdown)
  - Parameter sliders (D0, Butterworth order)
  - Export button (saves filtered image)

Run with: python -m proj09_2d_image.src.gui
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image


class ImageProcessingGUI:
    FILTER_NAMES = [
        'Ideal Lowpass',
        'Butterworth Lowpass',
        'Ideal Highpass',
        'Gaussian Highpass',
        'Notch Filter',
    ]

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title('2D Image Processing Workbench — Project 09')
        self.root.geometry('1100x750')

        self.image: np.ndarray | None = None   # current grayscale float image

        self._build_ui()
        self._load_demo_image()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ---- Top control bar ----
        ctrl = tk.Frame(self.root, bg='#2b2b2b', pady=6)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        tk.Button(ctrl, text='Load Image', command=self._load_image,
                  bg='#4a90d9', fg='white', relief='flat',
                  padx=8).pack(side=tk.LEFT, padx=6)

        tk.Label(ctrl, text='Filter:', bg='#2b2b2b',
                 fg='white').pack(side=tk.LEFT)
        self.filter_var = tk.StringVar(value=self.FILTER_NAMES[0])
        cb = ttk.Combobox(ctrl, textvariable=self.filter_var,
                          values=self.FILTER_NAMES, width=20, state='readonly')
        cb.pack(side=tk.LEFT, padx=4)
        cb.bind('<<ComboboxSelected>>', lambda _: self._update())

        tk.Label(ctrl, text='  D₀:', bg='#2b2b2b', fg='white').pack(side=tk.LEFT)
        self.D0_var = tk.IntVar(value=30)
        tk.Scale(ctrl, from_=5, to=120, orient=tk.HORIZONTAL,
                 variable=self.D0_var, bg='#2b2b2b', fg='white',
                 troughcolor='#555', length=150,
                 command=lambda _: self._update()).pack(side=tk.LEFT)

        tk.Label(ctrl, text='  Order (BW):', bg='#2b2b2b',
                 fg='white').pack(side=tk.LEFT)
        self.order_var = tk.IntVar(value=4)
        tk.Scale(ctrl, from_=1, to=10, orient=tk.HORIZONTAL,
                 variable=self.order_var, bg='#2b2b2b', fg='white',
                 troughcolor='#555', length=100,
                 command=lambda _: self._update()).pack(side=tk.LEFT)

        tk.Button(ctrl, text='Export Result', command=self._export,
                  bg='#5a9e5a', fg='white', relief='flat',
                  padx=8).pack(side=tk.RIGHT, padx=8)

        # ---- Status bar ----
        self.status_var = tk.StringVar(value='Ready')
        tk.Label(self.root, textvariable=self.status_var,
                 bg='#1e1e1e', fg='#aaa', anchor='w',
                 relief='sunken').pack(side=tk.BOTTOM, fill=tk.X)

        # ---- Matplotlib 2×2 canvas ----
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 6.5))
        self.fig.patch.set_facecolor('#1e1e1e')
        plt.subplots_adjust(wspace=0.05, hspace=0.15, left=0.02,
                            right=0.98, top=0.95, bottom=0.05)
        for ax in self.axes.flat:
            ax.set_facecolor('#2b2b2b')
            ax.axis('off')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Set 2×2 panel titles
        titles = ['Original Image', 'Input Spectrum (log |F|)',
                  'Filtered Spectrum', 'Filtered Image']
        for ax, t in zip(self.axes.flat, titles):
            ax.set_title(t, color='white', fontsize=9)

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _load_demo_image(self):
        """Load a built-in demo image (scikit-image camera man)."""
        try:
            from skimage import data
            img = data.camera().astype(float) / 255.0
            self.image = img
            self.status_var.set('Demo image loaded (skimage.data.camera)')
            self._update()
        except Exception as e:
            self.status_var.set(f'Could not load demo: {e}')

    def _load_image(self):
        path = filedialog.askopenfilename(
            title='Select image',
            filetypes=[('Images', '*.png *.jpg *.jpeg *.bmp *.tif *.tiff'),
                       ('All files', '*.*')]
        )
        if not path:
            return
        try:
            pil = Image.open(path).convert('L')
            arr = np.array(pil, dtype=float) / 255.0
            self.image = arr
            self.status_var.set(f'Loaded: {path}  ({arr.shape[1]}×{arr.shape[0]})')
            self._update()
        except Exception as e:
            messagebox.showerror('Error', str(e))

    # ------------------------------------------------------------------
    # Filter computation & display
    # ------------------------------------------------------------------

    def _build_filter(self) -> np.ndarray:
        from .filters2d import (ideal_lowpass, butterworth_lowpass,
                                 ideal_highpass, gaussian_highpass,
                                 notch_filter)
        shape = self.image.shape
        D0 = float(self.D0_var.get())
        n = int(self.order_var.get())
        name = self.filter_var.get()

        if name == 'Ideal Lowpass':
            return ideal_lowpass(shape, D0)
        elif name == 'Butterworth Lowpass':
            return butterworth_lowpass(shape, D0, n)
        elif name == 'Ideal Highpass':
            return ideal_highpass(shape, D0)
        elif name == 'Gaussian Highpass':
            return gaussian_highpass(shape, D0)
        else:  # Notch
            return notch_filter(shape, [(D0 * 0.5, 0), (0, D0 * 0.5)], radius=8)

    def _update(self):
        if self.image is None:
            return

        from .dft2d import fft2d, fftshift2d
        from .filters2d import apply_filter

        H = self._build_filter()
        filtered = apply_filter(self.image, H)

        F_orig = fft2d(self.image)
        spec_orig = np.log1p(np.abs(fftshift2d(F_orig)))

        F_filt = fft2d(filtered)
        spec_filt = np.log1p(np.abs(fftshift2d(F_filt)))

        imgs = [self.image, spec_orig, spec_filt, filtered]
        cmaps = ['gray', 'inferno', 'inferno', 'gray']

        for ax, img, cmap in zip(self.axes.flat, imgs, cmaps):
            ax.images[0].set_data(img) if ax.images else ax.imshow(img, cmap=cmap)
            if ax.images:
                ax.images[0].set_clim(img.min(), img.max())
                ax.images[0].set_cmap(cmap)

        # Re-render if no images existed yet
        if not self.axes[0, 0].images:
            for ax, img, cmap in zip(self.axes.flat, imgs, cmaps):
                ax.imshow(img, cmap=cmap)

        self.canvas.draw_idle()

    def _full_redraw(self):
        """Full redraw (used after first load)."""
        if self.image is None:
            return

        from .dft2d import fft2d, fftshift2d
        from .filters2d import apply_filter

        H = self._build_filter()
        filtered = apply_filter(self.image, H)

        F_orig = fft2d(self.image)
        spec_orig = np.log1p(np.abs(fftshift2d(F_orig)))
        F_filt = fft2d(filtered)
        spec_filt = np.log1p(np.abs(fftshift2d(F_filt)))

        imgs = [self.image, spec_orig, spec_filt, filtered]
        cmaps = ['gray', 'inferno', 'inferno', 'gray']
        titles = ['Original Image', 'Input Spectrum (log |F|)',
                  'Filtered Spectrum', 'Filtered Image']

        for ax, img, cmap, title in zip(self.axes.flat, imgs, cmaps, titles):
            ax.clear()
            ax.imshow(img, cmap=cmap)
            ax.set_title(title, color='white', fontsize=9)
            ax.axis('off')

        self.canvas.draw_idle()
        self.filtered_image = filtered

    # Patch _update to use full redraw on first call
    def _update(self):
        self._full_redraw()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export(self):
        if self.image is None:
            messagebox.showwarning('No image', 'Load an image first.')
            return
        from .filters2d import apply_filter
        H = self._build_filter()
        filtered = apply_filter(self.image, H)

        path = filedialog.asksaveasfilename(
            defaultextension='.png',
            filetypes=[('PNG', '*.png'), ('JPEG', '*.jpg'), ('All', '*.*')]
        )
        if path:
            out = (np.clip(filtered, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(out).save(path)
            self.status_var.set(f'Exported to {path}')


def main():
    root = tk.Tk()
    app = ImageProcessingGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
