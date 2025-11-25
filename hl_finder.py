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
                 df = add_tile_row(df, search_pic, rm, cm, r0, c0, r, c, h_min - h_mid, l, h_mid, h0, h, h_min - h_mid - hgoal)
                 search_pic.mark(rm, cm)
                 search_pic.mark(r0, c0)
                 search_pic.mark(r, c)
          

    return search_pic, df

def add_tile_row(df, search_pic, pxmidx, pxmidy, pxa1x, pxa1y, pxa2x, pxa2y, height, length, hmid, ha1, ha2, score):

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
        "ha2" : ha2,
        "score" : score

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
        "ha2",
        "score"
    ])
    return df

