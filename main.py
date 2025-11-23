import sys
import csv
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np
from grid import Grid
from hl_finder import pixel_slope_gaussian, pixel_slope_sobel_physical, search_highline, create_hl_dataframe
from search_picture import SearchPicture, get_search_picture
import math
from scipy.ndimage import maximum_filter, zoom
import pandas as pd
from cluster_csv import cluster_and_extract
from concurrent.futures import ProcessPoolExecutor, as_completed

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


def process_task(args):
    (df, fld, min_hl_length, max_hl_length, H, px_size_m_output,
     c1_north, c1_east) = args

    # prepare search area
    search_pic, px_size_m_output = get_search_picture(
        fld,
        c1_north,
        c1_east,
        max_hl_length,
        px_size_m_output
    )

    # run detection
    _, df_out = search_highline(
        df,
        search_pic,
        px_size_m_output,
        min_hl_length,
        max_hl_length,
        H
    )

    return df_out




if __name__ == "__main__":
    # Check that the user provided a folder path
    if len(sys.argv) < 2:
        print("two args needed")
        sys.exit(1)

    fld = sys.argv[1]

    north_min=6120
    north_max=6169
    east_min=570
    east_max=619

    # mosaic = combine_tiles(fld, north_min, north_max, east_min, east_max)
    # tile_size_km=1
    # tile_px=2500
    # write_meta_data_tiles(fld, north_min, north_max, east_min, east_max, tile_size_km, tile_px)
    grid = Grid.from_info_files(fld,north_min, north_max, east_min, east_max)

    if (grid == None):
        sys.exit()


    ranges = pd.DataFrame([
        {"min_hl_length": 0,   "max_hl_length": 50,  "H": 10, "pxsize": 5},
        {"min_hl_length": 50,  "max_hl_length": 150, "H": 15, "pxsize": 7},
        {"min_hl_length": 150, "max_hl_length": 250, "H": 20, "pxsize": 10},
        {"min_hl_length": 200, "max_hl_length": 350, "H": 25, "pxsize": 10},
        {"min_hl_length": 350, "max_hl_length": 500, "H": 30, "pxsize": 15}
    ])

    df = create_hl_dataframe()           # read-only in workers
    tasks = []
    
    for _, r in ranges.iterrows():
        min_hl_length = r["min_hl_length"]
        max_hl_length = r["max_hl_length"]
        H = r["H"]
        px_size_m_output = r["pxsize"]
    
        coords = grid.get_highline_coords(H)
    
        for (c1_north, c1_east) in coords:
            tasks.append(
                (df, fld,
                 min_hl_length, max_hl_length, H, px_size_m_output,
                 c1_north, c1_east)
            )


    all_results = []                   # we will merge results after

    with ProcessPoolExecutor() as ex:
        futures = [ex.submit(process_task, t) for t in tasks]
    
        total = len(tasks)
        done = 0

        for fut in as_completed(futures):
            done += 1
            print(f"\rProgress: {done}/{total} [{'#' * int(40*done/total):<40}] {100*done/total:5.1f}%", end="")
        
            all_results.append(cluster_and_extract(fut.result(), ranges, radius=50))

    
    df = pd.concat(all_results, ignore_index=True)
    df = cluster_and_extract(df, ranges, radius=50)
    df.to_csv("fyn_north.csv", sep=' ')

    # clustered_df = cluster_and_extract(df, ranges, radius=50)
    # clustered_df.to_csv("clustered_lines.csv", sep=' ')
    # clustered_df.to_csv("clustered_lines.csv", sep=' ', mode='a', header=False, index=False)


    # plt.show()

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




