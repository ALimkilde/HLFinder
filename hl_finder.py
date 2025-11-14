import sys
import csv
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np
from grid import Grid
import math
from scipy.ndimage import maximum_filter, zoom

import re

def pixels_within_bounds(im, px_pr_meter, px, min_len, max_len):
    """
    Returns pixel coordinates within the distance bounds from a given pixel.

    Parameters:
        im (np.ndarray): 2D image
        px_pr_meter (float): pixels per meter
        px (tuple): (row, col) coordinate of reference pixel
        min_len (float): minimum distance in meters
        max_len (float): maximum distance in meters

    Returns:
        list of (row, col) tuples within the given distance range.
    """
    rows, cols = im.shape[:2]
    r0, c0 = px

    # Convert bounds to pixel units
    min_px = min_len / px_pr_meter
    max_px = max_len / px_pr_meter

    # Define local bounding box (avoid checking whole image)
    r_min = max(int(r0 - max_px), 0)
    r_max = min(int(r0 + max_px) + 1, rows)
    c_min = max(int(c0 - max_px), 0)
    c_max = min(int(c0 + max_px) + 1, cols)

    # Create coordinate grids
    rr, cc = np.ogrid[r_min:r_max, c_min:c_max]

    # Compute distances
    dist = np.sqrt((rr - r0)**2 + (cc - c0)**2)

    # Mask for points within range
    mask = (dist >= min_px) & (dist <= max_px) & (r0 <= rr)

    # Extract coordinates
    coords = np.argwhere(mask) + [r_min, c_min]

    return [tuple(coord) for coord in coords]

def mark_pixels_within_bounds(im, px_pr_meter, px, min_len, max_len, value=1):
    """
    Marks all pixels within the distance bounds from a given pixel.

    Parameters:
        im (np.ndarray): 2D or 3D image array
        px_pr_meter (float): pixels per meter
        px (tuple): (row, col) of the reference pixel
        min_len (float): minimum distance in meters
        max_len (float): maximum distance in meters
        value: the value or color to assign to marked pixels

    Returns:
        np.ndarray: a copy of the image with pixels marked
    """
    im_marked = im.copy()
    coords = pixels_within_bounds(im, px_pr_meter, px, min_len, max_len)

    for r, c in coords:
        im_marked[r, c] = value

    return im_marked

def brute_force_search(im, px_pr_meter, min_hl_length, max_hl_length, H):

    nx, ny = im.shape
    im_marked = im.copy()

    for r0 in range(nx):
        print(f"{r0} out of {nx}")
        for c0 in range(ny):
            h0 = im[r0, c0]
            for (r,c) in pixels_within_bounds(im, px_pr_meter, (r0,c0), min_hl_length, max_hl_length):
                h = im[r, c]
                avg_h = (h+h0)/2

                rm = int(np.round((r+r0)/2))
                cm = int(np.round((c+c0)/2))

                h_mid = im[rm, cm]
                
                if(avg_h - h_mid > H):
                    print(f"I found a highline with height {avg_h - h_mid}")
                    print(f"avg_h: {avg_h}, h0: {h0}, h: {h}, h_mid: {h_mid}")
                    im_marked[rm, cm] = 70
                    im_marked[r0, c0] = 70
                    im_marked[r, c] = 70
                    break

    return im_marked

