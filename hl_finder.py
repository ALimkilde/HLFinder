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
import pandas as pd

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

def get_extended_max_mask(im, px_size_m, max_hl_length, H, filter_type):
     n_extended = math.ceil(max_hl_length/(2*px_size_m))
    
     if (filter_type == 'max'):
         extmax = maximum_filter(im, size=(n_extended, n_extended), mode='nearest')
         mask = np.greater(extmax, im + H)
     if (filter_type == 'min'):
         extmin = minimum_filter(im, size=(n_extended, n_extended), mode='nearest')
         mask = np.greater(im, extmin + H)

     mask = maximum_filter(mask, size=(n_extended, n_extended), mode='nearest')

     return mask

def get_highline_mask(im, px_size_m, min_hl_length, max_hl_length, H):
    extmax_mask = get_extended_max_mask(im, px_size_m, max_hl_length, H, 'max')
    extmin_mask = get_extended_max_mask(im, px_size_m, max_hl_length, H, 'min')

    # Slope mask
    gx,gy = np.gradient(im, 5)
    tmp = np.sqrt(gx**2 + gy**2)/px_size_m
    tmp = maximum_filter(tmp, size=(2, 2), mode='nearest')

    acc_slope = 0.1*float(H)/float(0.5*max_hl_length)
    slope_mask = np.greater(tmp, acc_slope)

    mask = np.logical_and(slope_mask, np.logical_and(extmin_mask, extmax_mask))

    # plt.figure()
    # plt.imshow(tmp)

    # plt.figure()
    # plt.imshow(extmax_mask)

    # plt.figure()
    # plt.imshow(extmin_mask)

    # plt.figure()
    # plt.imshow(mask)
    # plt.show()

    return mask

def search_highline(df, search_pic, px_size_m, min_hl_length, max_hl_length, H):

    nx, ny = search_pic.shape()
    im = search_pic.get_im()
    
    mask = get_highline_mask(im, px_size_m, min_hl_length, max_hl_length, H)

    n_extended = math.ceil(max_hl_length/(px_size_m))

    rows, cols = np.where(mask)

    for r0,c0 in zip(rows, cols):
         h0 = float(im[r0, c0])
         
         if (h0 < H): 
             continue
          
         # for (r,c) in pixels_within_bounds(im, px_size_m, (r0,c0), min_hl_length, max_hl_length):
         rmax = min(r0+n_extended, nx)
         cmin = max(0, c0 - n_extended)
         cmax = min(c0+n_extended, ny)
         R, C = np.where(mask[r0:rmax, cmin:cmax])

         for (r,c) in zip(R + r0,C + cmin):

             l = search_pic.get_distance_px_to_m(r0,c0,r,c)
             if (l<min_hl_length):
                 continue

             if (l>max_hl_length):
                 continue

             h = float(im[r, c])
             if (h < H): 
                 continue

             rm = round((r+r0)/2)
             cm = round((c+c0)/2)

             h_mid = float(im[rm, cm])
             h_min = min(h, h0)

             if(h_min > h_mid + H):
                 # print(f"I found a highline with height {h_min - h_mid}")
                 # print(f"min_h: {min(h,h0)}, h0: {h0}, h: {h}, h_mid: {h_mid}")
                 df = add_tile_row(df, search_pic, rm, cm, r0, c0, r, c, h_min - h_mid, l, h_mid, h0, h)
                 search_pic.mark(rm, cm)
                 search_pic.mark(r0, c0)
                 search_pic.mark(r, c)
          

    return search_pic, df

def add_tile_row(df, search_pic, pxmidx, pxmidy, pxa1x, pxa1y, pxa2x, pxa2y, height, length, hmid, ha1, ha2):

    midx, midy = search_pic.get_coords(pxmidx, pxmidy)
    a1x, a1y = search_pic.get_coords(pxa1x, pxa1y)
    a2x, a2y = search_pic.get_coords(pxa2x, pxa2y)

    new_row = {
        "midx": midx,    # pixel index (x)
        "midy": midy,    # pixel index (y)
        "a1x": a1x,    # pixel index (x)
        "a1y": a1y,    # pixel index (y)
        "a2x": a2x,    # pixel index (x)
        "a2y": a2y,    # pixel index (y)
        "length": length,
        "height": height,
        "hmid": hmid,
        "ha1" : ha1,
        "ha2" : ha2
    }

    df.loc[len(df)] = new_row
    return df

def create_hl_dataframe():
    """
    Create an empty dataframe for storing tile attributes.
    """
    df = pd.DataFrame(columns=[
        "midx", 
        "midy",  
        "a1x",  
        "a1y",   
        "a2x",   
        "a2y",   
        "length",
        "height",
        "hmid",
        "ha1",
        "ha2"
    ])
    return df


def pixel_slope_gaussian(image, sigma=1.0):
    """
    Compute smooth slope using Gaussian derivative filters.
    """
    return gaussian_gradient_magnitude(image, sigma=sigma)

def pixel_slope_sobel_physical(image, meters_per_pixel):
    dx = sobel(image, axis=1) / meters_per_pixel
    dy = sobel(image, axis=0) / meters_per_pixel
    return np.hypot(dx, dy)

