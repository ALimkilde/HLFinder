import sys
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from tqdm import tqdm


def process_tif(tif_file, output_folder, region = "denmark"):
    """Process a single .tif file: gdal_translate + gdalinfo."""
    tif_file = Path(tif_file)
    base_name = tif_file.stem

    tif = Path(tif_file)
    bin_path = Path(output_folder) / f"{tif.stem}.bin"
    lz4_path = bin_path.with_suffix(".bin.lz4")
    info_path = Path(output_folder) / f"{base_name}.info"
    
    tmp_tif = Path(output_folder) / f"{base_name}_2m_tmp.tif" # temporary downsampled GeoTIFF

    cmd_warp = [
        "gdalwarp",
        "-tr", "2", "2",
        "-r", "max",
        str(tif),
        str(tmp_tif),
    ]

    cmd_translate = [
        "gdal_translate",
        "-of", "ENVI",
        "-ot", "UInt16",
        "-scale", "0", "3000", "0", "30000",  # exact 10× scaling
        str(tmp_tif),
        str(bin_path),
    ]

    cmd_lz4 = [
        "lz4",
        "-f",
        str(bin_path),
        str(lz4_path),
    ]

    subprocess.run(cmd_warp, 
                   check=True,
                   stderr=subprocess.DEVNULL,
                   stdout=subprocess.DEVNULL,
                   )

    subprocess.run(cmd_translate, 
                   check=True,
                   stderr=subprocess.DEVNULL,
                   stdout=subprocess.DEVNULL,
                   )

    subprocess.run(cmd_lz4, 
                   check=True,
                   stderr=subprocess.DEVNULL,
                   stdout=subprocess.DEVNULL,
                   )

    # cleanup large intermediate files
    bin_path.unlink()              # remove uncompressed .bin
    tmp_tif.unlink()               # remove temp GeoTIFF
    tmp_tif.with_suffix(".tif.aux.xml").unlink(missing_ok=True)

    # 2. gdalinfo
    # FOR 
    cmd2 = f"gdalinfo {tif_file} > {info_path}"
    subprocess.run(
        cmd2,
        shell=True,
        check=True,
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )

    return tif_file.name  # Return something small for logging


def main():
    if len(sys.argv) < 3:
        print("Usage: python run_on_tifs.py <folder_path> <output_folder>")
        sys.exit(1)

    input_folder = Path(sys.argv[1])
    output_folder = Path(sys.argv[2])
    region = sys.argv[3].lower() if len(sys.argv) > 3 else "denmark"

    if not input_folder.is_dir():
        print(f"Error: {input_folder} is not a valid directory.")
        sys.exit(1)

    output_folder.mkdir(parents=True, exist_ok=True)

    tifs = list(input_folder.glob("*.tif"))
    if not tifs:
        print("No .tif files found.")
        return

    # Number of workers — you can tune this
    # workers = os.cpu_count() or 4
    workers = 4
    print(f"Processing {len(tifs)} TIFF files with {workers} workers...")


    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
                executor.submit(process_tif, tif, output_folder, region): tif
                for tif in tifs
                }

    for future in tqdm(as_completed(futures), total=len(futures)):
        tif_name = futures[future].name
        try:
            future.result()
            print(f"✓ {tif_name}")
        except Exception as e:
            print(f"✗ Error processing {tif_name}: {e}")
            sys.exit()


if __name__ == "__main__":
    main()

