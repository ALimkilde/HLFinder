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

import re

@njit
def hlheight(l):
    return 0.08 * l + 8

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

def get_slope_mask(im, px_size_m, max_hl_length, H):
    gx,gy = np.gradient(im, 5)
    tmp = np.sqrt(gx**2 + gy**2)/px_size_m
    tmp = maximum_filter(tmp, size=(2, 2), mode='nearest')

    acc_slope = 0.1*float(H)/float(0.5*max_hl_length)
    return np.greater(tmp, acc_slope)

def plot_masks(extmax_mask, extmin_mask, slope_mask):
    plt.figure()
    plt.imshow(slope_mask)

    plt.figure()
    plt.imshow(extmax_mask)

    plt.figure()
    plt.imshow(extmin_mask)

    plt.figure()
    plt.imshow(mask)
    plt.show()

def get_highline_mask(im, px_size_m, min_hl_length, max_hl_length, H):
    extmax_mask = get_extended_max_mask(im, px_size_m, max_hl_length, H, 'max')
    extmin_mask = get_extended_max_mask(im, px_size_m, max_hl_length, H, 'min')

    slope_mask = get_slope_mask(im, px_size_m, max_hl_length, H)

    mask = np.logical_and(slope_mask, np.logical_and(extmin_mask, extmax_mask))

    # plot_masks(extmax_mask, extmin_mask, slope_mask)

    return mask

@njit
def search_highline(im, im_surf, px_size_m, min_hl_length, max_hl_length, H, mask):

    result = []

    nx, ny = im.shape
    
    n_extended = math.ceil(max_hl_length/(px_size_m))

    rows, cols = np.where(mask)

    for r0,c0 in zip(rows, cols):
         h0 = float(im[r0, c0])
         
         if (h0 < H): 
             continue
          
         rmax = min(r0+n_extended, nx)
         cmin = max(0, c0 - n_extended)
         cmax = min(c0+n_extended, ny)
         R, C = np.where(mask[r0:rmax, cmin:cmax])

         for (r,c) in zip(R + r0,C + cmin):
             l = get_distance_px_to_m(px_size_m,r0,c0,r,c)
             if (l<min_hl_length):
                 continue

             if (l>max_hl_length):
                 continue

             hgoal = hlheight(l)

             h = float(im[r, c])
             if (h < hgoal): 
                 continue

             rm = round((r+r0)/2)
             cm = round((c+c0)/2)

             h_mid = float(im[rm, cm])
             h_min = min(h, h0)

             if(h_min > h_mid + hgoal):
                 # print(f"I found a highline with height {h_min - h_mid}")
                 # print(f"min_h: {min(h,h0)}, h0: {h0}, h: {h}, h_mid: {h_mid}")
                 tree_in_way, htree = tree_in_the_way(im, im_surf, rm, cm, r0, c0, r, c, hgoal, h_min, h_mid)

                 htree = max(htree, h_mid)
                 if (not tree_in_way):
                     result.append(( rm, cm, r0, c0, r, c, h_min, l, h_mid, h0, h, htree, hgoal))
          

    return result


