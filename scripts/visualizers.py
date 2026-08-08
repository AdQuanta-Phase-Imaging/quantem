import numpy as np
import scipy.ndimage as ndi
import cv2
from matplotlib import pyplot as plt
from typing import Literal

def plot_image_diffs(im1, im2, subtitles=None, extent=None, suptitle=''):
    diff = im2 - im1
    
    max_val = np.nanmax([im1, im2])
    min_val = np.nanmin([im1, im2])
    diff_extent = np.nanmax(np.abs(diff))

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,5))
    fig.suptitle(suptitle)

    if subtitles is not None:
        for i, title in enumerate(subtitles):
            axes[i].set_title(title)
    
    if extent is None:
        for ax in axes:
            ax.set_axis_off()
        extent = [-1,1,-1,1] # Dummy values that will not be presented

    im = axes[0].imshow(im1, vmin=min_val, vmax=max_val, extent=extent)
    im = axes[1].imshow(im2, vmin=min_val, vmax=max_val, extent=extent)
    fig.colorbar(im, ax=axes[:2], fraction=0.046, pad=0.04)

    im = axes[2].imshow(
        diff, 
        vmin=-diff_extent, 
        vmax=diff_extent, 
        cmap='seismic', 
        extent=extent)
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    plt.show()
