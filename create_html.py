import pandas as pd
import folium
from pyproj import Transformer
import json

# -------------------------------------------------------
# SETTINGS
# -------------------------------------------------------
CSV_FILE = "test.csv"
OUTPUT_HTML = "map.html"
UTM_ZONE = 32
HEMISPHERE = "north"
# -------------------------------------------------------

# Prepare UTM→WGS84 transformer
utm_crs = f"+proj=utm +zone={UTM_ZONE} +{HEMISPHERE} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
transformer = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

def utm_to_latlon(x, y):
    lon, lat = transformer.transform(x, y)
    return lat, lon

# -------------------------------------------------------
# LOAD CSV
# -------------------------------------------------------
df = pd.read_csv(CSV_FILE, sep=" ")

# Precompute slider ranges
min_len, max_len = df["length"].min(), df["length"].max()
min_h, max_h     = df["height"].min(), df["height"].max()

# -------------------------------------------------------
# FUNCTION: color by height
# -------------------------------------------------------
def height_color(h):
    # blue → orange → red
    if h < min_h + (max_h-min_h)*0.33:
        return "#2a78db"   # blue
    elif h < min_h + (max_h-min_h)*0.66:
        return "#ff8c00"   # orange
    else:
        return "#d73027"   # red

# -------------------------------------------------------
# BUILD GEOJSON FEATURES
# -------------------------------------------------------
features = []

for _, row in df.iterrows():

    lat1, lon1 = utm_to_latlon(row["a1x"], row["a1y"])
    lat2, lon2 = utm_to_latlon(row["a2x"], row["a2y"])

    popup_html = f"""
    <b>ID:</b> {row['midx']}<br>
    <b>Length:</b> {row['length']:.2f}<br>
    <b>Height:</b> {row['height']:.2f}<br>
    <b>hmid:</b> {row['hmid']}<br>
    <b>ha1:</b> {row['ha1']}<br>
    <b>ha2:</b> {row['ha2']}<br>
    """

    color = height_color(row["height"])

    feature = {
        "type": "Feature",
        "properties": {
            "id": int(row["midx"]),
            "length": float(row["length"]),
            "height": float(row["height"]),
            "popup": popup_html,
            "color": color
        },
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [lon1, lat1],
                [lon2, lat2]
            ]
        }
    }

    features.append(feature)

geojson = {"type": "FeatureCollection", "features": features}

# -------------------------------------------------------
# CREATE THE MAP
# -------------------------------------------------------
center_lat, center_lon = utm_to_latlon(df["midx"].mean(), df["midy"].mean())
m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="cartodbpositron")

gj = folium.GeoJson(
    geojson,
    name="Lines",
    style_function=lambda feat: {
        "color": feat["properties"]["color"],
        "weight": 3,
        "opacity": 0.9
    },
    tooltip=folium.GeoJsonTooltip(fields=["id"]),
    popup=folium.GeoJsonPopup(fields=["popup"])
).add_to(m)

# -------------------------------------------------------
# ADD RANGE FILTER PANELS
# -------------------------------------------------------
filter_script = f"""
<script>
var gjLayer = {gj.get_name()};
var features = [];

// Collect all individual line layers once initialized
gjLayer.on('add', function() {{
    gjLayer.eachLayer(function(layer) {{
        features.push(layer);
    }});
}});

function applyFilters() {{
    var lenMin = parseFloat(document.getElementById('lengthMin').value);
    var lenMax = parseFloat(document.getElementById('lengthMax').value);

    var hMin = parseFloat(document.getElementById('heightMin').value);
    var hMax = parseFloat(document.getElementById('heightMax').value);

    features.forEach(function(line) {{
        var p = line.feature.properties;

        var inLength = (p.length >= lenMin && p.length <= lenMax);
        var inHeight = (p.height >= hMin && p.height <= hMax);

        if (inLength && inHeight) {{
            if (!line._map) line.addTo(gjLayer._map);
        }} else {{
            if (line._map) gjLayer._map.removeLayer(line);
        }}
    }});
}}

</script>

<div style="
    position: fixed;
    top: 10px;
    left: 10px;
    z-index: 999999;
    background: white;
    padding: 12px;
    border: 1px solid #999;
    border-radius: 4px;
    width: 240px;
">

<b>Filter by Length</b><br>
Min: <span id="lenMinVal">{min_len:.1f}</span>,
Max: <span id="lenMaxVal">{max_len:.1f}</span><br>

<input type="range" id="lengthMin" min="{min_len}" max="{max_len}" value="{min_len}" step="1"
  oninput="document.getElementById('lenMinVal').innerHTML=this.value; applyFilters();">

<input type="range" id="lengthMax" min="{min_len}" max="{max_len}" value="{max_len}" step="1"
  oninput="document.getElementById('lenMaxVal').innerHTML=this.value; applyFilters();">

<br><br>

<b>Filter by Height</b><br>
Min: <span id="hMinVal">{min_h:.1f}</span>,
Max: <span id="hMaxVal">{max_h:.1f}</span><br>

<input type="range" id="heightMin" min="{min_h}" max="{max_h}" value="{min_h}" step="1"
  oninput="document.getElementById('hMinVal').innerHTML=this.value; applyFilters();">

<input type="range" id="heightMax" min="{min_h}" max="{max_h}" value="{max_h}" step="1"
  oninput="document.getElementById('hMaxVal').innerHTML=this.value; applyFilters();">

</div>
"""

m.get_root().html.add_child(folium.Element(filter_script))

# -------------------------------------------------------
# SAVE
# -------------------------------------------------------
m.save(OUTPUT_HTML)
print("Saved:", OUTPUT_HTML)

