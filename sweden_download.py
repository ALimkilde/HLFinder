import time
import os
import requests
from urllib.parse import urlparse
import sys

# -------------------------------------
# LOAD AUTH FROM ENVIRONMENT VARIABLES
# -------------------------------------
USERNAME = os.getenv("LM_USERNAME")
PASSWORD = os.getenv("LM_PASSWORD")

if not USERNAME or not PASSWORD:
    print("❌ ERROR: Environment variables LM_USERNAME and/or LM_PASSWORD not set.")
    print("Set them first:")
    print("  export LM_USERNAME=\"your_username\"")
    print("  export LM_PASSWORD=\"your_password\"")
    sys.exit(1)

AUTH = (USERNAME, PASSWORD)

# -------------------------------------
# CONFIGURATION
# -------------------------------------
OUTPUT_FOLDER = "../Southern_Sweden_TIF/"

# Skåne bounding box in WGS84 (EPSG:4326)
# BBOX = [11.0, 55.3, 14.3, 56.6]
# Blekinge (EPSG:4326)
# BBOX = [
#     14.2,   # min lon (west)
#     55.9,   # min lat (south)
#     16.0,   # max lon (east)
#     56.5    # max lat (north)
# ]
# Halland (EPSG:4326)
# BBOX = [
#     11.3,   # min lon (west)
#     56.3,   # min lat (south)
#     13.4,   # max lon (east)
#     57.6    # max lat (north)
# ]
BBOX = [12.6, 56.9, 12.9, 57.1]

STAC_URL = "https://api.lantmateriet.se/stac-hojd/v1"

# -------------------------------------
# HELPER FUNCTIONS
# -------------------------------------
def download_file(url, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    filename = os.path.basename(urlparse(url).path)
    filepath = os.path.join(output_folder, filename)

    # Skip if tile already exists
    if os.path.exists(filepath):
        print(f"✔ Already exists: {filename}")
        return

    print(f"⬇ Downloading: {filename}")
    r = requests.get(url, auth=AUTH, stream=True)
    r.raise_for_status()

    with open(filepath, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"✔ Saved: {filepath}")


def search_collection(collection_id):
    query = {
        "collections": [collection_id],
        "bbox": BBOX,
        "limit": 2000
    }

    r = requests.post(f"{STAC_URL}/search", json=query, auth=AUTH)

    if r.status_code == 429:
        print("⏳ Rate limited. Sleeping 60 seconds…")
        time.sleep(60)
        return search_collection(collection_id)

    r.raise_for_status()
    time.sleep(1)  # polite delay
    return r.json().get("features", [])


def get_collections():
    print("Fetching collections…")
    r = requests.get(f"{STAC_URL}/collections", auth=AUTH)
    r.raise_for_status()
    data = r.json()
    return [c["id"] for c in data["collections"]]


# -------------------------------------
# MAIN ROUTINE
# -------------------------------------
def download_skane_dem():
    collections = get_collections()
    print(f"Found {len(collections)} collections total.")

    for col in collections:
        print(f"\n🔍 Searching in collection: {col}")
        items = search_collection(col)

        if not items:
            print("   No tiles found.")
            continue

        print(f"   Found {len(items)} tiles in {col}")

        for item in items:
            url = item["assets"]["data"]["href"]
            download_file(url, OUTPUT_FOLDER)

    print("\n🎉 Download complete!")


if __name__ == "__main__":
    download_skane_dem()
