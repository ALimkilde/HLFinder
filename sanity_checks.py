import sys
from pathlib import Path


def collect_png_basenames(folder):
    folder = Path(folder)
    return {f.stem for f in folder.glob("*.png")}


def collect_lz4_basenames(folder):
    folder = Path(folder)
    basenames = set()
    for f in folder.glob("*.bin.lz4"):
        name = f.name.replace(".bin.lz4", "")
        basenames.add(name)
    return basenames


def sanity_check(png_folder, lz4_folder):
    png_files = collect_png_basenames(png_folder)
    lz4_files = collect_lz4_basenames(lz4_folder)

    common = png_files & lz4_files
    missing_png = lz4_files - png_files
    missing_lz4 = png_files - lz4_files

    print(f"Total PNG files: {len(png_files)}")
    print(f"Total LZ4 files: {len(lz4_files)}")
    print(f"Matching pairs: {len(common)}")

    if missing_png:
        print(f"\nMissing {len(missing_png)} PNG files for:")
        with open("missing_png.txt", "w") as f:
            for name in sorted(missing_png):
                f.write(name + "\n")
        print("  → Written to missing_png.txt")
    
    if missing_lz4:
        print(f"\nMissing {len(missing_lz4)} LZ4 files for:")
        with open("missing_lz4.txt", "w") as f:
            for name in sorted(missing_lz4):
                f.write(name + "\n")
        print("  → Written to missing_lz4.txt")

    if not missing_png and not missing_lz4:
        print("\n✅ All files match perfectly!")
        return 0
    else:
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sanity_check.py <png_folder> <lz4_folder>")
        sys.exit(1)

    png_folder = sys.argv[1]
    lz4_folder = sys.argv[2]

    exit_code = sanity_check(png_folder, lz4_folder)
    sys.exit(exit_code)
