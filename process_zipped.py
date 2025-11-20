#!/usr/bin/env python3
import sys
import zipfile
import tempfile
import shutil
from pathlib import Path
import subprocess
from tqdm import tqdm   # pip install tqdm


def process_zip(zip_path: Path, output_base: Path):
    """Extract a zip, run the processing script, delete temp folder."""
    # Create temporary working directory
    workdir = Path(tempfile.mkdtemp())

    # Extract zip into the temporary directory
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(workdir)

    # Run your processing command
    subprocess.run(
        [
            "python",
            "process_10b10_tile.py",
            str(workdir),
            str(output_base),
        ],
        check=True
    )

    # Remove temporary directory
    shutil.rmtree(workdir)


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <DIR>")
        sys.exit(1)

    DIR = Path(sys.argv[1])
    ZIPDIR = DIR / "zipped"

    if not ZIPDIR.is_dir():
        print(f"Error: {ZIPDIR} does not exist.")
        sys.exit(1)

    zipfiles = sorted(ZIPDIR.glob("*.zip"))
    total = len(zipfiles)

    if total == 0:
        print("No zip files found.")
        sys.exit(0)

    print(f"Starting processing of {total} zip files...")

    OUTPUT_DIR = Path("/home/asger/Documents/Privat/slackline/hlfinder/all_denmark/")

    for zipfile_path in tqdm(zipfiles, desc="Processing zips", unit="zip"):
        process_zip(zipfile_path, OUTPUT_DIR)

    print("Done.")


if __name__ == "__main__":
    main()

