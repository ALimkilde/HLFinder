import matplotlib.pyplot as plt
import numpy as np
import math
import os
from PIL import Image
import cv2
import sys
from scipy.ndimage import maximum_filter, minimum_filter
from numba import njit

MAX_TREE_ANCHOR = 5
MAX_TREE_FRACTION = 0.5

def get_anchors(terr, surf):
    maxh  = np.maximum(terr, surf)
    trees = np.subtract(maxh, terr)
    treefrac = MAX_TREE_FRACTION * trees
    treemax = np.minimum(treefrac, MAX_TREE_ANCHOR)
    return np.add(terr, treemax)



class SearchPicture:
    def __init__(self, im, im_min_surf, im_max_surf, ref_px, ref_coords, px_size_m, tile_size_m):
        self.im = im

        self.im_min_surf = im_min_surf
        self.has_min_surf_data = (self.im_min_surf is not None)

        # Construct HL anchor image
        if (im_max_surf is None):
            self.im_anchor = im
        else:
            self.im_anchor = get_anchors(im, im_max_surf)

        self.ref_px = ref_px
        self.ref_coords = np.array(ref_coords)*1000
        self.px_size_m = px_size_m
        self.tile_size_m = tile_size_m

    def get_im(self):
        return self.im

    def shape(self):
        return self.im.shape

    def get_coords(self, x, y):
        diff = np.subtract((x,y), self.ref_px)
        diff[0] = -diff[0] 
        diff = np.array((diff[1],diff[0]))
        new_coor = np.add(self.ref_coords,diff*self.px_size_m)
        return tuple(new_coor)

def combine_tiles(folder_path, north_min, north_max, east_min, east_max, 
                  tile_size_km=1, tile_px=2500):
    """
    Combines grayscale tiles (DTM_1km_<north>_<east>.png) into one large
    NumPy array covering the specified grid bounds.

    Parameters:
        folder_path (str or Path): Directory containing the tiles
        north_min, north_max, east_min, east_max (int): Bounds in grid indices
        tile_size_km (int): Tile size in km (default 1)
        tile_px (int): Tile size in pixels (default 1000)
    
    Returns:
        np.ndarray or None: The combined 2D array (grayscale values)
    """

    # print(f"north_min={north_min} north_max={north_max} east_min={east_min} east_max={east_max}")

    # Calculate output dimensions
    n_rows = north_max - north_min + 1
    n_cols = east_max - east_min + 1
    out_h = n_rows * tile_px
    out_w = n_cols * tile_px

    # Initialize mosaic as float or uint8 (depending on your data)
    mosaic = np.zeros((out_h, out_w), dtype=np.uint8)

    tiles_found = False

    for north in range(north_min, north_max + 1):
        for east in range(east_min, east_max + 1):
            filename = f"DTM/DTM_{tile_size_km}km_{north}_{east}.png"
            path = f"{folder_path}/{filename}"
            if not os.path.exists(path):
                continue

            tiles_found = True

            # Read tile as grayscale NumPy array
            # img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = np.array(Image.open(path).convert("L"))


            # Compute placement in output
            col = east - east_min
            row = north_max - north  # north decreases downward
            y0 = row * tile_px
            x0 = col * tile_px

            mosaic[y0:y0 + tile_px, x0:x0 + tile_px] = img

    if not tiles_found:
        # print("⚠️ No tiles found in the given bounds.")
        return None, None

    mosaic_surface = mosaic.copy()
    tiles_found = False

    # Look for Surface data
    for north in range(north_min, north_max + 1):
        for east in range(east_min, east_max + 1):
            filename = f"DSM/DSM_{tile_size_km}km_{north}_{east}.png"
            path = f"{folder_path}/{filename}"
            # print(f"search for {path}")
            if not os.path.exists(path):
                continue

            # print(f"Found surface data: {filename}")

            tiles_found = True

            # Read tile as grayscale NumPy array
            # img = np.array(Image.open(path).convert("L"))
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            # Compute placement in output
            col = east - east_min
            row = north_max - north  # north decreases downward
            y0 = row * tile_px
            x0 = col * tile_px

            mosaic_surface[y0:y0 + tile_px, x0:x0 + tile_px] = img

    if not tiles_found:
        mosaic_surface = None

    return mosaic, mosaic_surface

def coarsen_image(mosaic, crop_px, px_size_m, px_size_m_output, filt):
    if (mosaic is None):
        return None, None 

    arr = mosaic[crop_px:-crop_px, crop_px:-crop_px]
    
    n = math.floor(px_size_m * px_size_m_output)
    # print(f"n: {n}")
    if (filt == 'max'):
        out = arr[:arr.shape[0]//n*n, :arr.shape[1]//n*n].reshape(arr.shape[0]//n, n, arr.shape[1]//n, n).max(axis=(1,3))
    else:
        arr = maximum_filter(arr, size=(4, 4), mode='nearest')
        out = arr[:arr.shape[0]//n*n, :arr.shape[1]//n*n].reshape(arr.shape[0]//n, n, arr.shape[1]//n, n).mean(axis=(1,3))

    return out, n

def get_search_picture(folder_path, north, east, max_hl_length, px_size_m_output, tile_size_meter=1000, tile_size_px=2500):

    max_hl_length_in_km = math.ceil(max_hl_length/1000)
    px_size_m = float(tile_size_px)/float(tile_size_meter)

    final_size_m = tile_size_meter + max_hl_length

    north_min = north - max_hl_length_in_km
    north_max = north + max_hl_length_in_km
    east_min = east - max_hl_length_in_km
    east_max = east + max_hl_length_in_km

    # print(f"north_min={north_min}\nnorth_max={north_max}\neast_min={east_min}\neast_max={east_max}")
    # print(f"north_min={north_min} north_max={north_max} east_min={east_min} east_max={east_max}")
 
    mosaic, mosaic_surface = combine_tiles(folder_path, north_min, north_max, east_min, east_max)

    if mosaic is None:
        return None, None

    padding = math.ceil(max_hl_length/2*px_size_m)
    crop_px = tile_size_px - padding
    # print(f"Crop: {crop_px}")

    out, n = coarsen_image(mosaic, crop_px, px_size_m, px_size_m_output, 'max')

    out_min_surface, _ = coarsen_image(mosaic_surface, crop_px, px_size_m, px_size_m_output, 'min')
    out_max_surface, _ = coarsen_image(mosaic_surface, crop_px, px_size_m, px_size_m_output, 'max')

    tile_size_m_out = float(final_size_m)
    refpx_y = math.floor(padding/n) 
    refpx_x = math.floor((padding+tile_size_px)/n) 

    nout,mout = out.shape

    true_px_size_m = tile_size_m_out/nout
    sp = SearchPicture(out, out_min_surface, out_max_surface, (refpx_x,refpx_y),(east,north),true_px_size_m,tile_size_m_out)
    
    return sp, true_px_size_m

@njit
def tree_in_the_way(im, im_surf, rm, cm, r0, c0, r, c, hgoal, h_min, h_mid):
    if (im_surf is None):
        # print("No surf data; tree not in the way...")
        return False, h_mid

    out = False

    h_tree = im_surf[rm, cm]
    out = (not h_min > h_tree + hgoal) or out
    if (out):
        # print(f"On RM, CM: Tree in the way. Height terrain: {h_min - h_mid}. Height surface: {h_min - h_tree}")
        return out, h_tree

    rm4, cm4 = (rm+r0)//2, (cm+c0)//2
    h_tree = im_surf[rm4, cm4]
    out = (not h_min > h_tree + (hgoal - 8)/2) or out
    if (out):
        # print(f"On RM41, CM41: Tree in the way. Height terrain: {h_min - h_mid}. Height surface: {h_min - h_tree}")
        return out, h_tree

    rm4, cm4 = (rm+r)//2, (cm+c)//2
    h_tree = im_surf[rm4, cm4]
    out = (not h_min > h_tree + (hgoal-8)/2) or out
    if (out):
        # print(f"On RM42, CM42: Tree in the way. Height terrain: {h_min - h_mid}. Height surface: {h_min - h_tree}")
        return out, h_tree

    return out, h_tree


@njit
def get_distance_px_to_m(px_size_m, x1, y1, x2, y2):
    dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)

    return dist*px_size_m
