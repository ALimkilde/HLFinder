import sys
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


def process_tif(tif_file, output_folder):
    """Process a single .tif file: gdal_translate + gdalinfo."""
    tif_file = Path(tif_file)
    base_name = tif_file.stem

    png_path = Path(output_folder) / f"{base_name}.png"
    info_path = Path(output_folder) / f"{base_name}.info"

    # 1. gdal_translate
    cmd1 = f"gdal_translate -of PNG {tif_file} {png_path}"
    subprocess.run(
        cmd1,
        shell=True,
        check=True,
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )

    # 2. gdalinfo
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

    if not input_folder.is_dir():
        print(f"Error: {input_folder} is not a valid directory.")
        sys.exit(1)

    output_folder.mkdir(parents=True, exist_ok=True)

    tifs = list(input_folder.glob("*.tif"))
    if not tifs:
        print("No .tif files found.")
        return

    # Number of workers — you can tune this
    workers = os.cpu_count() or 4
    # print(f"Processing {len(tifs)} TIFF files with {workers} workers...")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_tif, tif, output_folder): tif
            for tif in tifs
        }

        for future in as_completed(futures):
            tif_name = futures[future].name
            try:
                future.result()
                # print(f"✓ {tif_name}")
            except Exception as e:
                sys.exit()
                # print(f"✗ Error processing {tif_name}: {e}")
                # print(f"✓ {tif_name}")


if __name__ == "__main__":
    main()

