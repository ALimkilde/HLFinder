import sys
import csv
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np
from grid import Grid
from search_image import SearchImage
from hl_finder import brute_force_search
import math
from scipy.ndimage import maximum_filter, zoom

import re

def filter_info_files(folder_path, H):
    """
    Processes all .info files in a folder and returns a list of base names
    where (Max - Min) > H.
    """
    folder = Path(folder_path)
    selected_files = []

    # Regular expressions to extract Min and Max values
    min_pattern = re.compile(r"Min=([0-9.\-eE]+)")
    max_pattern = re.compile(r"Max=([0-9.\-eE]+)")

    for info_file in folder.glob("*.info"):
        text = info_file.read_text(errors="ignore")

        # Find first Min and Max values
        min_match = min_pattern.search(text)
        max_match = max_pattern.search(text)

        if not (min_match and max_match):
            continue  # skip files missing values

        try:
            min_val = float(min_match.group(1))
            max_val = float(max_match.group(1))
        except ValueError:
            continue

        diff = max_val - min_val
        print(f"min: {min_val}, max: {max_val}, diff: {diff}")

        if (max_val - min_val) > H:
            print(f"Found file {info_file.stem}")
            selected_files.append(info_file.stem)

    return selected_files

def dms_to_dd(dms_str):
    """
    Convert a DMS coordinate string (like 12d18'23.83"E)
    to decimal degrees (float).
    """
    # Extract parts using regex
    match = re.match(r"(\d+)d(\d+)'([\d.]+)\"?([NSEW])", dms_str.strip())
    if not match:
        raise ValueError(f"Invalid DMS format: {dms_str}")

    degrees, minutes, seconds, direction = match.groups()
    dd = float(degrees) + float(minutes)/60 + float(seconds)/3600

    # South and West are negative
    if direction in ['S', 'W']:
        dd *= -1

    return dd

def conv_gps():
    longitude_str = "12d18'21.15\"E"
    latitude_str = "55d16'17.86\"N"
    
    longitude = dms_to_dd(longitude_str)
    latitude = dms_to_dd(latitude_str)
    
    print(f"Latitude: {latitude:.6f}, Longitude: {longitude:.6f}")
    print(f"Google Maps format: {latitude:.6f}, {longitude:.6f}")

def plot_image(image_path):
    img = Image.open(image_path)
    
    # Plot the image
    plt.figure()
    plt.imshow(img)
    plt.axis("off")  # Hide the axes
    plt.title("Loaded Image")

def write_meta_data_tiles(folder_path, north_min, north_max, east_min, east_max,
                          tile_size_km=1, tile_px=2500):
    """
    Writes metadata about a grid of tiles to a CSV file.

    Columns include:
      - Parameter
      - Value
      - Units
    """

    folder = Path(folder_path)
    out_file = f"metadata_{north_min}_{north_max}_{east_min}_{east_max}.csv"

    # --- Compute metadata ---
    tile_size_m = tile_size_km * 1000  # km → m
    n_rows = north_max - north_min + 1
    n_cols = east_max - east_min + 1

    total_size_m_x = n_cols * tile_size_m
    total_size_m_y = n_rows * tile_size_m

    total_px_x = n_cols * tile_px
    total_px_y = n_rows * tile_px

    pixel_size_x = tile_size_m / tile_px
    pixel_size_y = tile_size_m / tile_px

    # --- Data rows for CSV ---
    rows = [
        ["North_min", north_min, ""],
        ["North_max", north_max, ""],
        ["East_min", east_min, ""],
        ["East_max", east_max, ""],
        ["Tile_size", tile_size_m, "m"],
        ["Tile_resolution", tile_px, "px"],
        ["Total_size_x", total_size_m_x, "m"],
        ["Total_size_y", total_size_m_y, "m"],
        ["Total_px_x", total_px_x, "px"],
        ["Total_px_y", total_px_y, "px"],
        ["Pixel_size_x", round(pixel_size_x, 3), "m/px"],
        ["Pixel_size_y", round(pixel_size_y, 3), "m/px"],
    ]

    # --- Write CSV file ---
    with open(out_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Parameter", "Value", "Units"])
        writer.writerows(rows)

    print(f"✅ Metadata written to {out_file}")

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
    px_pr_meter = float(tile_size_px)/float(tile_size_meter)

    north_min = north - max_hl_length_in_km
    north_max = north + max_hl_length_in_km
    east_min = east - max_hl_length_in_km
    east_max = east + max_hl_length_in_km

    print(f"north_min={north_min}\nnorth_max={north_max}\neast_min={east_min}\neast_max={east_max}")
 
    mosaic = combine_tiles(folder_path, north_min, north_max, east_min, east_max)

    crop_px = tile_size_px - math.ceil(max_hl_length*px_pr_meter)
    print(f"Crop: {crop_px}")

    arr = mosaic[crop_px:-crop_px, crop_px:-crop_px]
    
    n = math.floor(px_pr_meter * px_size_m_output)
    print(f"n: {n}")
    out = arr[:arr.shape[0]//n*n, :arr.shape[1]//n*n].reshape(arr.shape[0]//n, n, arr.shape[1]//n, n).max(axis=(1,3))
    
    return out

if __name__ == "__main__":
    # Check that the user provided a folder path
    if len(sys.argv) < 3:
        print("two args needed")
        sys.exit(1)

    fld = sys.argv[1]
    H = float(sys.argv[2])

    north_min=6130
    north_max=6139
    east_min=710
    east_max=719

    # mosaic = combine_tiles(fld, north_min, north_max, east_min, east_max)
    # write_meta_data_tiles(fld, north_min, north_max, east_min, east_max, tile_size_km, tile_px)
    grid = Grid.from_info_files(fld,north_min, north_max, east_min, east_max)

    coords = grid.get_highline_coords(H)

    c1_north, c1_east = coords[0]
    print(f"({c1_north}, {c1_east})")

    min_hl_length = 170
    max_hl_length = 250
    px_size_m_output = 1
    im = get_search_picture(fld, c1_north, c1_east, max_hl_length, px_size_m_output)

    plt.figure()
    plt.imshow(im)

    sim = SearchImage.coarse(im, 1, 200, 'max')
    sim.plot()

    sim2 = SearchImage.coarse(im, 1, 200, 'min')
    sim2.plot()

    x,y = 1, 1
    xold, yold = sim2.get_original_px(x,y)
    print(f"Original of ({x},{y}) is ({xold},{yold})")

    plt.show()
    sys.exit()

    # im = mark_pixels_within_bounds(im, px_size_m_output, (140, 40), min_hl_length, max_hl_length, value=H)
    im = brute_force_search(im, px_size_m_output, min_hl_length, max_hl_length, 30)

    print(im)
    plt.figure()
    plt.imshow(im)
    plt.axis("off")  # Hide the axes
    plt.title("Search image")
    plt.show()

    # print(min_grid)
    # print(max_grid)

    # out_file = f"data_{north_min}_{north_max}_{east_min}_{east_max}.png"

    # if mosaic is not None:
    #     print("Combined array shape:", mosaic.shape)
    #     Image.fromarray(mosaic).save(out_file)

    # plot_image(f".png")
    # selected_images = filter_info_files(fld, float(H))

    # for img in selected_images:
    #     plot_image(f"{fld}/{img}.png")
    
    # plt.show()




