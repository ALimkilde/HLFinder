import matplotlib.pyplot as plt
import numpy as np
import math
import os
from PIL import Image
import cv2
import sys
from scipy.ndimage import maximum_filter, minimum_filter
from numba import njit

from config import MAX_TREE_ANCHOR, MAX_TREE_FRACTION, MAX_HL_LENGTH, PRE_FILT_SIZE, PADDING_M, REGION, STEP_SIZE, COOR_SIZE
import re


def load_dem_png(path):
    """
    Loads a uint16 PNG DEM and applies scale factor encoded in filename.

    Filename format:
        *_s0p1.png  -> scale = 0.1
        *_s1.png    -> scale = 1.0
        no suffix   -> scale = 1.0 (fallback)
    """

    # Read image (uint16 preserved)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Could not read image: {path}")

    img = img.astype(np.float32)

    # Extract scale from filename
    match = re.search(r"_s(\d+(?:p\d+)?)", path)
    if match:
        scale_str = match.group(1).replace("p", ".")
        scale = float(scale_str)
    else:
        scale = 1.0  # fallback

    if scale <= 0 or scale > 10:
        raise ValueError(f"Suspicious scale factor: {scale}")

    img = img * scale
    return img

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
        self.ref_coords = np.array(ref_coords)*COOR_SIZE
        self.px_size_m = px_size_m
        self.tile_size_m = tile_size_m

    def get_im(self):
        return self.im

    def shape(self):
        return self.im.shape

    def get_coords(self, r, c):
        diff = np.subtract((r,c), self.ref_px)
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
    n_rows = (north_max - north_min)//STEP_SIZE + 1
    n_cols = (east_max - east_min)//STEP_SIZE + 1
    out_h = n_rows * tile_px
    out_w = n_cols * tile_px

    # Initialize mosaic as float or uint8 (depending on your data)
    mosaic = np.zeros((out_h, out_w), dtype=np.uint8)

    tiles_found = False

    for north in range(north_min, north_max + 1, STEP_SIZE):
        for east in range(east_min, east_max + 1, STEP_SIZE):
            if REGION == "Denmark":
                filename = f"DTM/DTM_{tile_size_km}km_{north}_{east}.png"
            else:
                filename = f"DTM/{north}_{east}_25.png"

            path = f"{folder_path}/{filename}"
            # print(f"path: {path}")
            if not os.path.exists(path):
                # print("DOESNT EXISTS")
                continue

            tiles_found = True

            # Read tile as grayscale NumPy array
            img = load_dem_png(path)
            # img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            # img = np.array(Image.open(path).convert("L"))


            # Compute placement in output
            col = (east - east_min)//STEP_SIZE
            row = (north_max - north)//STEP_SIZE  # north decreases downward
            y0 = row * tile_px
            x0 = col * tile_px

            mosaic[y0:y0 + tile_px, x0:x0 + tile_px] = img

    if not tiles_found:
        # print("⚠️ No tiles found in the given bounds.")
        return None, None

    mosaic_surface = mosaic.copy()
    tiles_found = False

    # Look for Surface data
    for north in range(north_min, north_max + 1, STEP_SIZE):
        for east in range(east_min, east_max + 1, STEP_SIZE):
            if REGION == "Denmark":
                filename = f"DSM/DSM_{tile_size_km}km_{north}_{east}.png"
            else:
                filename = f"DSM/{north}_{east}_25_s0p1.png"

            path = f"{folder_path}/{filename}"
            # print(f"search for {path}")
            if not os.path.exists(path):
                continue

            # print(f"Found surface data: {filename}")

            tiles_found = True

            # Read tile as grayscale NumPy array
            img = load_dem_png(path)
            # img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            # img = np.array(Image.open(path).convert("L"))

            # Compute placement in output
            col = east - east_min
            row = north_max - north  # north decreases downward
            y0 = row * tile_px
            x0 = col * tile_px

            mosaic_surface[y0:y0 + tile_px, x0:x0 + tile_px] = img

    if not tiles_found:
        mosaic_surface = None

    return mosaic, mosaic_surface

def original_to_coarse_pixel(r, c, n, crop_px):
    """
    Given a pixel (r, c) in the ORIGINAL cropped mosaic coordinates
    (i.e., 0 <= r < H0, same for c), return the coarse pixel index (i, j).

    Parameters
    ----------
    r, c : int
        Row/column in the original mosaic.
    n : int
        Block size used in coarsening (fine pixels per coarse pixel).
    crop_px : int
        Crop applied around the original mosaic before coarsening.

    Returns
    -------
    i, j : int
        Coordinates of the coarse pixel in the downsampled array.
        Returns None, None if the pixel lies inside the cropped-away border.
    """

    # Check if the pixel lies inside the usable (not-cropped-away) region
    if r < crop_px or c < crop_px:
        return None, None

    remainder_r = (r - crop_px) % n

    remainder_c = (c - crop_px) % n

    # Coarse pixel indices (floor division)
    i = (r - crop_px) // n
    j = (c - crop_px) // n

    return i, j

def coarsen_image(mosaic, crop_px, px_size_m, px_size_m_output, filt, pre_filt=False):
    """
    Crop and downsample an image by block aggregation.
    Adjusts crop_px upward so the final cropped image dimensions
    are exact multiples of the block size n.

    Returns:
        out           - coarsened image
        n             - block size (fine pixels per coarse pixel)
        crop_px_new   - adjusted crop value
        size_m        - (height_m, width_m) size of the output in meters
    """

    if mosaic is None:
        return None, None, None, None

    # ----- 1. Compute block size n correctly -----
    n = int(px_size_m_output / px_size_m)
    if n < 1:
        raise ValueError("px_size_m_output must be >= px_size_m.")

    H0, W0 = mosaic.shape

    # ----- 2. Compute how much extra cropping is needed -----
    def adjust_crop(crop_px, size):
        interior = size - 2 * crop_px
        remainder = interior % n
        if remainder == 0:
            return crop_px
    
        # Make the new interior divisible by n by reducing the total interior
        new_interior = interior - remainder + n

        if (size - new_interior) % 2 == 1:
            if (n % 2 == 1):
               new_interior = new_interior + n
            else:
               new_interior = new_interior + 2*2*n


    
        # Corresponding crop per side (must be symmetric)
        new_crop = (size - new_interior) // 2 
    
        return new_crop


    crop_px_new = min(
        adjust_crop(crop_px, H0),
        adjust_crop(crop_px, W0)
    )

    # ----- 3. Apply the adjusted crop -----
    arr = mosaic[crop_px_new : -crop_px_new, crop_px_new : -crop_px_new]
    H, W = arr.shape  # guaranteed multiples of n

    # ----- 4. Optional prefiltering -----
    if pre_filt:
        arr = maximum_filter(arr, size=(PRE_FILT_SIZE, PRE_FILT_SIZE), mode='nearest')

    # ----- 5. Block reshape and aggregation -----
    arr_blocks = arr.reshape(H // n, n, W // n, n)

    if filt == "max":
        out = arr_blocks.max(axis=(1, 3))
    elif filt == "min":
        out = arr_blocks.min(axis=(1, 3))
    elif filt == "median":
        out = np.median(arr_blocks, axis=(1, 3))
    else:
        out = arr_blocks.mean(axis=(1, 3))

    # ----- 6. Compute output physical size -----
    Hc, Wc = out.shape
    size_m = (Hc * px_size_m_output, Wc * px_size_m_output)
    true_px_size_m = n * px_size_m

    return out, n, crop_px_new, size_m

def coarse_pixel_source_range(i, j, n, crop_px):
    """
    Return (row_start, row_end, col_start, col_end) in original mosaic coordinates
    for coarse pixel (i, j).
    """

    row_start = crop_px + i * n
    row_end   = crop_px + (i + 1) * n - 1

    col_start = crop_px + j * n
    col_end   = crop_px + (j + 1) * n - 1

    return row_start, row_end, col_start, col_end


def get_search_picture(folder_path, north, east, px_size_m_output, tile_size_meter=1000, tile_size_px=2500):

    max_hl_length_in_tiles = math.ceil(MAX_HL_LENGTH/tile_size_meter)
    px_size_m = float(tile_size_meter)/float(tile_size_px)

    north_min = north - max_hl_length_in_tiles*STEP_SIZE
    north_max = north + max_hl_length_in_tiles*STEP_SIZE
    east_min = east - max_hl_length_in_tiles*STEP_SIZE
    east_max = east + max_hl_length_in_tiles*STEP_SIZE

    # print(f"north_min={north_min}\nnorth_max={north_max}\neast_min={east_min}\neast_max={east_max}")
    # print(f"north_min={north_min} north_max={north_max} east_min={east_min} east_max={east_max}")

    refpx_row_in = 2*tile_size_px 
    refpx_col_in = tile_size_px
 
    mosaic, mosaic_surface = combine_tiles(folder_path, north_min, north_max, east_min, east_max)

    if mosaic is None:
        return None

    padding = math.ceil(PADDING_M/px_size_m)
    crop_px = tile_size_px - padding

    out, n, crop_px_new, tile_size_m_out = coarsen_image(mosaic, 
                                                     crop_px, 
                                                     px_size_m, 
                                                     px_size_m_output, 
                                                     'mean')

    if mosaic_surface is not None:
        out_min_surface, _, _, _ = coarsen_image(mosaic_surface, 
                                                 crop_px, 
                                                 px_size_m, 
                                                 px_size_m_output, 
                                                 'max',
                                                 pre_filt = True)

        out_max_surface, _, _, _ = coarsen_image(mosaic_surface, 
                                                 crop_px, 
                                                 px_size_m, 
                                                 px_size_m_output, 
                                                 'max')
    else:
        out_min_surface = None
        out_max_surface = None

    refpx_row, refpx_col = original_to_coarse_pixel(refpx_row_in,
                                                refpx_col_in, 
                                                n, crop_px_new)


    nout,mout = out.shape

    sp = SearchPicture(out, out_min_surface, out_max_surface, (refpx_row,refpx_col),(east,north),px_size_m_output,tile_size_m_out)

    return sp

@njit
def get_distance_px_to_m(px_size_m, x1, y1, x2, y2):
    dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)

    return dist*px_size_m

if __name__ == "__main__":
    path = sys.argv[1]

    img = load_dem_png(path)

    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title(f"loaded DEM")
    plt.axis('off')
    plt.show()
