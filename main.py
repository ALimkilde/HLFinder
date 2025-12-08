import sys
import csv
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np
from grid import Grid
from hl_finder import search_highline, hlheight, get_highline_mask
from search_picture import get_search_picture
import math
from scipy.ndimage import maximum_filter, zoom
import pandas as pd
from cluster_csv import cluster_and_extract
from create_html import save_HL_map
from numba import jit
from hl_plotter import extract_line_profiles, plot_line_profiles
import re

import config
from config import MIN_HL_LENGTH, MAX_HL_LENGTH, PX_SIZE_M_SEARCH

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


def add_tile_row(df, search_pic, res):

    rm, cm, ra1, ca1, ra2, ca2, h_min, length, hmid, ha1, ha2, hgoal, score, hmean_terr, hmean_surf, walkable = res

    height = h_min - hmid
    rigging_height_a1 = h_min - search_pic.im[ra1, ca1]
    rigging_height_a2 = h_min - search_pic.im[ra2, ca2]

    midx, midy = search_pic.get_coords(rm, cm)
    a1x, a1y = search_pic.get_coords(ra1, ca1)
    a2x, a2y = search_pic.get_coords(ra2, ca2)

    search_pic.im[rm,cm] = 100
    plt.imshow(search_pic.im)
    plt.show()

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
        "rigging_height_a1": rigging_height_a1,
        "ha2" : ha2,
        "rigging_height_a2": rigging_height_a2,
        "score" : score,
        "hmean_terr" : hmean_terr,
        "hmean_surf" : hmean_surf,
        "hmean" : min(hmean_terr, hmean_surf),
        "walkable_length" : walkable
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
        "rigging_height_a1",
        "ha2",
        "rigging_height_a2",
        "score",
        "hmean_terr",
        "hmean_surf",
        "hmean",
        "walkable_length"
    ])
    return df

def get_df_from_result(df, result, search_pic):
    for res in result:
        df = add_tile_row(df, search_pic, res)

    return df

def process_task(args):
    (df, fld, H, c_north, c_east) = args

    # prepare search area
    search_pic = get_search_picture(
        fld,
        c_north,
        c_east,
        PX_SIZE_M_SEARCH
    )
    if search_pic == None:
        return None

    mask = get_highline_mask(search_pic.im, search_pic.im_anchor)

    # run detection
    result = search_highline(
        search_pic.im,
        search_pic.im_min_surf,
        search_pic.im_anchor,
        H,
        mask
    )

    result =  cluster_and_extract(result, 
                                  PX_SIZE_M_SEARCH, 
                                  radius=config.CLUSTER_RADIUS, 
                                  keep = config.KEEP_METRICS)

    # for r in result:
    #      (
    #         rm, cm, r0, c0, r1, c1, h_min, l, h_mid, h0, h1, htree, hgoal, score 
    #      ) = r

    #      d_m, terr, surf, anch = extract_line_profiles(search_pic.im, search_pic.im_min_surf, search_pic.im_anchor, r0, c0, r1, c1, PX_SIZE_M_SEARCH)
        
    #      plot_line_profiles(d_m, terr, surf, anch, score, l, h_min, min(h0,h1))
    

    df = get_df_from_result(df, result, search_pic)

    return df


def run_tasks(tasks, use_parallel=True):
    all_results = []

    if use_parallel:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor() as ex:
            futures = [ex.submit(process_task, t) for t in tasks]

            total = len(tasks)
            done = 0

            for fut in as_completed(futures):
                done += 1
                print(f"\rProgress: {done}/{total} [{'#' * int(40*done/total):<40}] {100*done/total:5.1f}%", end="")

                result = fut.result()
                if result is not None:
                    all_results.append(result)

    else:
        # Sequential mode
        total = len(tasks)
        for i, t in enumerate(tasks, 1):
            result = process_task(t)
            print(f"\rProgress: {i}/{total} [{'#' * int(40*i/total):<40}] {100*i/total:5.1f}%", end="")

            if result is not None:
                all_results.append(result)

    print()  # finish progress line
    return all_results



if __name__ == "__main__":
    # Check that the user provided a folder path
    if len(sys.argv) < 2:
        print("two args needed")
        sys.exit(1)

    fld = sys.argv[1]

    north_min=6165
    north_max=6165
    east_min=685
    east_max=685
    outname="tmp"

    # mosaic = combine_tiles(fld, north_min, north_max, east_min, east_max)
    # tile_size_km=1
    # tile_px=2500
    # write_meta_data_tiles(fld, north_min, north_max, east_min, east_max, tile_size_km, tile_px)
    grid = Grid.from_info_files(fld,north_min, north_max, east_min, east_max)

    if (grid == None):
        sys.exit()


    df = create_hl_dataframe()           # read-only in workers
    tasks = []
    
    H = hlheight(MIN_HL_LENGTH)
    
    coords = grid.get_highline_coords(H)
    
    for (c_north, c_east) in coords:
        tasks.append(
            (df, fld, H, c_north, c_east)
        )


    all_results = run_tasks(tasks, use_parallel=True)
    if (len(all_results) == 0):
        sys.exit()

    # df = pd.concat(all_results, ignore_index=True)
    df = pd.concat(all_results, ignore_index=False)
    df = df.drop_duplicates()
    df.to_csv(f"{outname}.csv", sep=' ')
    save_HL_map(df, f"{outname}.html", score_threshold=0.0)




