import sys
import csv
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np
from grid import Grid
import math
from scipy.ndimage import maximum_filter, zoom, sobel
from scipy.ndimage import maximum_filter, minimum_filter, gaussian_gradient_magnitude

import re

def pixels_within_bounds(im, px_size_m, px, min_len, max_len):
    """
    Returns pixel coordinates within the distance bounds from a given pixel.

    Parameters:
        im (np.ndarray): 2D image
        px_size_m (float): pixels per meter
        px (tuple): (row, col) coordinate of reference pixel
        min_len (float): minimum distance in meters
        max_len (float): maximum distance in meters

    Returns:
        list of (row, col) tuples within the given distance range.
    """
    rows, cols = im.shape[:2]
    r0, c0 = px

    # Convert bounds to pixel units
    min_px = min_len / px_size_m
    max_px = max_len / px_size_m

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

def mark_pixels_within_bounds(im, px_size_m, px, min_len, max_len, value=1):
    """
    Marks all pixels within the distance bounds from a given pixel.

    Parameters:
        im (np.ndarray): 2D or 3D image array
        px_size_m (float): pixels per meter
        px (tuple): (row, col) of the reference pixel
        min_len (float): minimum distance in meters
        max_len (float): maximum distance in meters
        value: the value or color to assign to marked pixels

    Returns:
        np.ndarray: a copy of the image with pixels marked
    """
    im_marked = im.copy()
    coords = pixels_within_bounds(im, px_size_m, px, min_len, max_len)

    for r, c in coords:
        im_marked[r, c] = value

    return im_marked

def get_extended_max_mask(im, px_size_m, max_hl_length, H):
     n_extended = math.ceil(max_hl_length/(2*px_size_m))
    
     extmax = maximum_filter(im, size=(n_extended, n_extended), mode='nearest')
     mask = np.greater(extmax, im + H)

     mask = maximum_filter(mask, size=(n_extended, n_extended), mode='nearest')

     return mask


def brute_force_search_masked(im, px_size_m, min_hl_length, max_hl_length, H):

    nx, ny = im.shape
    im_marked = im.copy()
    
    slope_mask = np.greater(pixel_slope_sobel_physical(im, px_size_m), 0.0)
    extmask = get_extended_max_mask(im, px_size_m, max_hl_length, H)

    mask = np.logical_and(slope_mask, extmask)

    for r0 in range(nx):
        print(f"{r0} done")
        for c0 in range(ny):
            if (not mask[r0,c0]): 
                continue

            h0 = im[r0, c0]
            for (r,c) in pixels_within_bounds(im, px_size_m, (r0,c0), min_hl_length, max_hl_length):
                if (not mask[r,c]): 
                    continue
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

def brute_force_search(im, px_size_m, min_hl_length, max_hl_length, H, rangex, rangey):

    nx, ny = im.shape
    im_marked = im.copy()
    
    slope_mask = np.greater(pixel_slope_sobel_physical(im, px_size_m), 0.0)

    for r0 in rangex:
        print(f"{r0} done")
        for c0 in rangey:
            if (not slope_mask[r0,c0]): 
                continue

            h0 = im[r0, c0]
            for (r,c) in pixels_within_bounds(im, px_size_m, (r0,c0), min_hl_length, max_hl_length):
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


def pixel_slope_gaussian(image, sigma=1.0):
    """
    Compute smooth slope using Gaussian derivative filters.
    """
    return gaussian_gradient_magnitude(image, sigma=sigma)

def pixel_slope_sobel_physical(image, meters_per_pixel):
    dx = sobel(image, axis=1) / meters_per_pixel
    dy = sobel(image, axis=0) / meters_per_pixel
    return np.hypot(dx, dy)

