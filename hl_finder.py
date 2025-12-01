import sys
import csv
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np
from grid import Grid
import math
from scipy.ndimage import maximum_filter, minimum_filter
import pandas as pd
from search_picture import tree_in_the_way, get_distance_px_to_m
from numba import njit
from numba.typed import List

import re

@njit
def hlheight(l):
    return 0.08 * l + 8

@njit
def hlheight_over_trees(l):
    return 0.08 * l 

import numpy as np
from scipy.ndimage import maximum_filter

def max_quadrants_1(im, dist):
    """
    Compute directional maximum filters:
      Q1 = up-right
      Q2 = up-left
      Q3 = down-left
      Q4 = down-right

    Returns a tuple: (Q1, Q2, Q3, Q4)
    """

    # --- Q1: Up + Right ---
    # Shift down & left → up-right neighborhood becomes rectangle
    pad = np.pad(im, ((dist, 0), (0, dist)), mode='edge')
    shifted = pad[:-dist, dist:]
    Q1 = maximum_filter(shifted, size=(dist+1, dist+1))

    # --- Q3: Down + Left ---
    # Shift up & right
    pad = np.pad(im, ((0, dist), (dist, 0)), mode='edge')
    shifted = pad[dist:, :-dist]
    Q3 = maximum_filter(shifted, size=(dist+1, dist+1))

    return Q1, Q3

def max_quadrants_2(im, dist):
    """
    Compute directional maximum filters:
      Q1 = up-right
      Q2 = up-left
      Q3 = down-left
      Q4 = down-right

    Returns a tuple: (Q1, Q2, Q3, Q4)
    """

    # --- Q2: Up + Left ---
    # Shift down & right
    pad = np.pad(im, ((dist, 0), (dist, 0)), mode='edge')
    shifted = pad[:-dist, :-dist]
    Q2 = maximum_filter(shifted, size=(dist+1, dist+1))


    # --- Q4: Down + Right ---
    # Shift up & left
    pad = np.pad(im, ((0, dist), (0, dist)), mode='edge')
    shifted = pad[dist:, dist:]
    Q4 = maximum_filter(shifted, size=(dist+1, dist+1))

    return Q2, Q4

def improved_max_masks(im, px_size_m, min_hl_length, max_hl_length):

     num = 7

     lengths = np.linspace(min_hl_length, max_hl_length, num)

     mask = np.full(im.shape, False)

     for i,l in enumerate(lengths[:-1]):
         lmin = l
         lmax = lengths[i+1]
         hgoal = hlheight(lmin)
         n_extended = math.ceil(lmax/(2*px_size_m))
         Q1, Q3 = max_quadrants_1(im, n_extended)
         Q2, Q4 = max_quadrants_2(im, n_extended)

         extQ1 = np.greater(Q1, im+hgoal)
         extQ2 = np.greater(Q2, im+hgoal)
         extQ3 = np.greater(Q3, im+hgoal)
         extQ4 = np.greater(Q4, im+hgoal)

         and1 = np.logical_and(extQ1, extQ3)
         and2 = np.logical_and(extQ2, extQ4)

         mq1, mq3 = max_quadrants_1(and1, n_extended)
         mq2, mq4 = max_quadrants_2(and2, n_extended)

         mask1 = np.logical_or(mq1, mq3)
         mask2 = np.logical_or(mq2, mq4)

         mask = np.logical_or(mask, mask1)
         mask = np.logical_or(mask, mask2)

         # masks = {
         #       "im": im,
         #     # "extQ1": extQ1,
         #     # "extQ2": extQ2,
         #     # "extQ3": extQ3,
         #     # "extQ4": extQ4,
         #     "and1": and1,
         #     "and2": and2,
         #     "mask1": mask1,
         #     "mask2": mask2
         # }
         # plot_masks(masks)

     
     # masks = {
     #       "im": im,
     #        "mask final": mask
     # }
     # plot_masks(masks)
     # sys.exit()

     return mask



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

def get_slope_mask(im, px_size_m):
    gx,gy = np.gradient(im, 4)
    tmp = np.sqrt(gx**2 + gy**2)/px_size_m
    tmp = maximum_filter(tmp, size=(2, 2), mode='nearest')

    acc_slope = 0.04
    return np.greater(tmp, acc_slope)

def plot_masks(masks):

    n = len(masks)
    cols = 2
    rows = (n + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = axes.ravel()

    for ax, (name, data) in zip(axes, masks.items()):
        ax.imshow(data)
        ax.set_title(name)
        ax.axis("off")

    # Hide any unused subplots
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def get_highline_mask(im, px_size_m, min_hl_length, max_hl_length, H):
    extmax_mask = improved_max_masks(im, px_size_m, min_hl_length, max_hl_length)

    slope_mask = get_slope_mask(im, px_size_m)

    # mask = np.logical_and(slope_mask, np.logical_and(extmin_mask, extmax_mask))
    mask = np.logical_and(slope_mask, extmax_mask)

    # masks = {
    #     "im": im,
    #     "slope_mask": slope_mask,
    #     "extmax_mask": extmax_mask,
    # }
    # plot_masks(masks)
    # sys.exit()

    return mask

@njit
def search_highline(im, im_surf, px_size_m, min_hl_length, max_hl_length, H, mask):

    result = List()

    nx, ny = im.shape
    
    n_extended = math.ceil(max_hl_length/(px_size_m))

    for r0 in range(0, nx):
        for c0 in range(0, ny):
            if mask[r0,c0]:
                h0 = float(im[r0, c0])
                
                if (h0 < H): 
                    continue
                 
                rmax = min(r0+n_extended, nx)
                cmin = max(0, c0 - n_extended)
                cmax = min(c0+n_extended, ny)

                for r in range(r0, rmax):
                    for c in range(cmin, cmax):
                        if mask[r,c]:
                            l = get_distance_px_to_m(px_size_m,r0,c0,r,c)
                            if (l<min_hl_length):
                                continue

                            if (l>max_hl_length):
                                continue

                            hgoal = hlheight(l)

                            h = float(im[r, c])
                            if (h < hgoal): 
                                continue

                            rm = (r + r0) // 2
                            cm = (c + c0) // 2

                            h_mid = float(im[rm, cm])
                            h_min = min(h, h0)

                            if(h_min > h_mid + hgoal):
                                # print(f"I found a highline with height {h_min - h_mid}")
                                # print(f"min_h: {min(h,h0)}, h0: {h0}, h: {h}, h_mid: {h_mid}")
                                hgoal_tree = hlheight_over_trees(l)
                                tree_in_way, htree = tree_in_the_way(im, im_surf, rm, cm, r0, c0, r, c, hgoal_tree, h_min, h_mid)

                                htree = max(htree, h_mid)
                                if (not tree_in_way):
                                    result.append(( rm, cm, r0, c0, r, c, h_min, l, h_mid, h0, h, htree, hgoal))
                 

    return result


