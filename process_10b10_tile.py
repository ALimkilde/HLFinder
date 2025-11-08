import sys
import subprocess
from pathlib import Path

def main():
    # Check that the user provided a folder path
    if len(sys.argv) < 2:
        print("Usage: python run_on_tifs.py <folder_path>")
        sys.exit(1)

    fld = sys.argv[1]
    print(fld)
    fld = fld.rstrip('/')
    print(fld)
    fld_png = f"{fld}_png/"
    print(f"mkdir {fld_png}")
    subprocess.run(f"mkdir {fld_png}", shell=True)

    folder = Path(fld)

    if not folder.is_dir():
        print(f"Error: {folder} is not a valid directory.")
        sys.exit(1)

    # The bash command you want to run on each .tif file
    bash_command = f"gdal_translate -of PNG"

    # Loop through all .tif files
    for tif_file in folder.glob("*.tif"):
        base_name = tif_file.stem  # file name without extension
        png_name = f"{base_name}.png"
        print(f"Running command on {base_name}...")
        cmd = f"{bash_command} {tif_file} {fld_png}/{png_name}"
        subprocess.run(cmd, shell=True, check=True)

        cmd = f"gdalinfo {tif_file} > {fld_png}/{base_name}.info"
        subprocess.run(cmd, shell=True, check=True)

    print("✅ Done processing all .tif files.")

if __name__ == "__main__":
    main()
