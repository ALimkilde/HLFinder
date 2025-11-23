import matplotlib.pyplot as plt
import numpy as np
import math
from pathlib import Path
from PIL import Image
import sys


class SearchPicture:
    def __init__(self, im, ref_px, ref_coords, px_size_m, tile_size_m):
        self.im = im
        self.ref_px = ref_px
        self.ref_coords = np.array(ref_coords)*1000
        self.px_size_m = px_size_m
        self.tile_size_m = tile_size_m

        # Internals
        self.im_marked = im.copy()
        self.mark_val = 70


    def get_im(self):
        return self.im

    def get_im_marked(self):
        return self.im_marked

    def shape(self):
        return self.im.shape

    def mark(self,x,y):
        self.im_marked[x,y] = self.mark_val
        # self.im_marked[self.ref_px[0], self.ref_px[1]] = self.mark_val

    def get_coords(self, x, y):
        diff = np.subtract((x,y), self.ref_px)
        diff[0] = -diff[0] 
        diff = np.array((diff[1],diff[0]))
        new_coor = np.add(self.ref_coords,diff*self.px_size_m)
        return tuple(new_coor)

    def get_distance_px_to_m(self, x1, y1, x2, y2):
        diff = np.subtract((x1,y1), (x2,y2))
        dist = np.linalg.norm(diff)

        return dist*self.px_size_m


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

    folder = Path(folder_path)

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
            filename = f"DTM_{tile_size_km}km_{north}_{east}.png"
            path = folder / filename
            # print(f"search for {path}")
            if not path.exists():
                continue

            tiles_found = True

            # Read tile as grayscale NumPy array
            img = np.array(Image.open(path).convert("L"))

            # Compute placement in output
            col = east - east_min
            row = north_max - north  # north decreases downward
            y0 = row * tile_px
            x0 = col * tile_px

            mosaic[y0:y0 + tile_px, x0:x0 + tile_px] = img

    if not tiles_found:
        print("⚠️ No tiles found in the given bounds.")
        return None

    return mosaic

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
 
    mosaic = combine_tiles(folder_path, north_min, north_max, east_min, east_max)

    padding = math.ceil(max_hl_length/2*px_size_m)
    crop_px = tile_size_px - padding
    # print(f"Crop: {crop_px}")

    arr = mosaic[crop_px:-crop_px, crop_px:-crop_px]
    
    n = math.floor(px_size_m * px_size_m_output)
    # print(f"n: {n}")
    out = arr[:arr.shape[0]//n*n, :arr.shape[1]//n*n].reshape(arr.shape[0]//n, n, arr.shape[1]//n, n).max(axis=(1,3))

    tile_size_m_out = float(final_size_m)
    refpx_y = math.floor(padding/n) 
    refpx_x = math.floor((padding+tile_size_px)/n) 

    nout,mout = out.shape

    true_px_size_m = tile_size_m_out/nout
    out = SearchPicture(out,(refpx_x,refpx_y),(east,north),true_px_size_m,tile_size_m_out)
    
    return out, true_px_size_m
