import os
import re
import json
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from pyproj import Transformer
import folium
import sys
from tqdm import tqdm

# ------------------------------
# Configuration
# ------------------------------
FOLDER = sys.argv[1]
OUTPUT_HTML = sys.argv[2]
UTM_ZONE = 32
TILE_SIZE = 1000  # 1×1 km

# Regex to match DTM_1km_6101_511.png or DSM_...
pattern = re.compile(r"D[TS]M_1km_(\d+)_(\d+)\.png")

# ------------------------------
# 1. Read tiles
# ------------------------------
tiles = []
for filename in os.listdir(FOLDER):
    match = pattern.match(filename)
    if match:
        northing_km = int(match.group(1))
        easting_km = int(match.group(2))

        e = easting_km * 1000
        n = northing_km * 1000

        tiles.append((e, n))

print(f"Found {len(tiles)} tiles.")

# ------------------------------
# 2. UTM → WGS84 transformer
# ------------------------------
epsg_code = 32600 + UTM_ZONE
transformer = Transformer.from_crs(epsg_code, 4326, always_xy=True)

# ------------------------------
# 3. Build polygons
# ------------------------------
polys = []
for e, n in tqdm(tiles, desc="Building tile polygons"):
    corners = [
        (e, n),
        (e + TILE_SIZE, n),
        (e + TILE_SIZE, n + TILE_SIZE),
        (e, n + TILE_SIZE),
        (e, n),
    ]
    lonlat = [transformer.transform(x, y) for x, y in corners]
    polys.append(Polygon(lonlat))

# ------------------------------
# 4. Merge polygons
# ------------------------------
print("Merging polygons...")
merged = unary_union(polys)

# ------------------------------
# 5. Convert merged geometry to GeoJSON
# ------------------------------
features = []

# merged may be Polygon or MultiPolygon
if merged.geom_type == "Polygon":
    geoms = [merged]
else:
    geoms = list(merged.geoms)

for g in geoms:
    features.append({
        "type": "Feature",
        "properties": {},
        "geometry": mapping(g)
    })

geojson_merged = {
    "type": "FeatureCollection",
    "features": features
}

# ------------------------------
# 6. Determine map center
# ------------------------------
# Extract all coordinates in merged geometry
all_lons = []
all_lats = []
for f in features:
    coords = f["geometry"]["coordinates"]
    if isinstance(coords[0][0], (float, int)):
        # simple polygon
        ring = coords[0]
        for x, y in ring:
            all_lons.append(x)
            all_lats.append(y)
    else:
        # multipolygon
        for poly in coords:
            for x, y in poly:
                all_lons.append(x)
                all_lats.append(y)

center = (sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons))

# ------------------------------
# 7. Build map
# ------------------------------
m = folium.Map(location=center, zoom_start=11)

folium.GeoJson(
    geojson_merged,
    name="Merged Areas",
    style_function=lambda x: {
        "fillOpacity": 0.1,
        "color": "red",
        "weight": 2,
    }
).add_to(m)

folium.LayerControl().add_to(m)

# ------------------------------
# 8. Save map
# ------------------------------
m.save(OUTPUT_HTML)
print(f"Map saved to {OUTPUT_HTML}")

