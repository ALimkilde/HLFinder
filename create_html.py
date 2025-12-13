import pandas as pd
import folium
from pyproj import Transformer
import sys

def save_HL_map(df, output_html, score_threshold=0, utm_zone=32, hemisphere="north", hmean_threshold=0):
    # Prepare UTM→WGS84 transformer
    utm_crs = f"+proj=utm +zone={utm_zone} +{hemisphere} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    transformer = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
    
    def utm_to_latlon(x, y):
        lon, lat = transformer.transform(x, y)
        return lat, lon
    
    
    # Create color scale by height
    min_h, max_h = df["hmean"].min(), df["hmean"].max()
    max_h = min(max_h, 15)

    min_w, max_w = df["walkable_length"].min(), df["walkable_length"].max()
    # max_w = min(max_h, 15)
    
    min_score, max_score = df["score"].min(), df["score"].max()
    min_score = max(max(min_score, 0), score_threshold)
    max_score = min(max_score, 20)
    
    i = 0
    for _, row in df.iterrows():
        if (row["score"] < score_threshold):
            continue
        if (row["hmean"] < hmean_threshold):
            continue
        i += 1
    
    print(f"N lines found: {i}")
    if (i == 0):
        sys.exit()
    
    # print(f"slope_max: {slope_max}")
    # print(f"slope_min: {slope_min}")
    
    def slope_color(h,l):
        s = h/l
        t = (s - slope_min) / (slope_max - slope_min + 1e-9)
        # print(f"s: {s}")
        # print(f"t: {t}")
        # blue → red gradient
        r = int(255 * t)
        b = int(255 * (1 - t))
        # print(f"rgb({r},0,{b})")
        return f"rgb({r},0,{b})"
    
    def hmean_color(h):
        t = (h - min_h) / (max_h - min_h + 1e-9)
        # blue → red gradient
        r = int(255 * t)
        b = int(255 * (1 - t))
        return f"rgb({r},0,{b})"

    def walkable_color(w):
        t = (w - min_w) / (max_w - min_w + 1e-9)
        # blue → red gradient
        r = int(255 * t)
        b = int(255 * (1 - t))
        return f"rgb({r},0,{b})"
    
    def score_color(s):
        t = (s - min_score) / (max_score - min_score + 1e-9)
        # blue → red gradient
        r = int(255 * t)
        b = int(255 * (1 - t))
        return f"rgb({r},0,{b})"
    
    # -------------------------------------------------------
    # BUILD GEOJSON LINES
    # -------------------------------------------------------
    features = []

    
    for _, row in df.iterrows():
        if (row["score"] < score_threshold):
            continue
        if (row["hmean"] < hmean_threshold):
            continue
    
        # Convert endpoints
        lat1, lon1 = utm_to_latlon(row["a1x"], row["a1y"])
        lat2, lon2 = utm_to_latlon(row["a2x"], row["a2y"])
    
        # Convert midpoint in UTM -> WGS84 for Google Maps link
        center_lat, center_lon = utm_to_latlon(row["midx"], row["midy"])

        a1_lat, a1_lon = utm_to_latlon(row["a1x"], row["a1y"])
        a2_lat, a2_lon = utm_to_latlon(row["a2x"], row["a2y"])
    
        google_link = f"https://www.google.com/maps?q={center_lat},{center_lon}"
        google_link_a1 = f"https://www.google.com/maps?q={a1_lat},{a1_lon}"
        google_link_a2 = f"https://www.google.com/maps?q={a2_lat},{a2_lon}"

        skrfoto_link = f'https://skraafoto.dataforsyningen.dk/?center={row["midx"]}%2C{row["midy"]}'
        skrfoto_link_a1 = f'https://skraafoto.dataforsyningen.dk/?center={row["a1x"]}%2C{row["a1y"]}'
        skrfoto_link_a2 = f'https://skraafoto.dataforsyningen.dk/?center={row["a2x"]}%2C{row["a2y"]}'
    
        popup_html = f"""
        <b>Center UTM:</b><br> {row['midx']} {row['midy']}<br>
        <b>Anchor 1 UTM:</b><br> {row['a1x']} {row['a1y']}<br>
        <b>Anchor 2 UTM:</b><br> {row['a2x']} {row['a2y']}<br><br>

        <b>hmid:</b> {row['hmid']:.1f}<br>
        <b>ha1:</b> {row['ha1']:.1f}<br>
        <b>ha2:</b> {row['ha2']:.1f}<br><br>
    
        <b>Length:</b> {row['length']:.1f}<br>
        <b>Height:</b> {row['height']:.1f}<br>
        <b>Mean height:</b> {row['hmean']:.1f}<br>
        <b>Rigging height a1:</b> {row['rigging_height_a1']:.1f}<br>
        <b>Rigging height a2:</b> {row['rigging_height_a2']:.1f}<br>
        <b>Walkable length:</b> {row['walkable_length']:.1f}<br>
        <b>score:</b> {row['score']:.3f}<br><br>
    
        <a href='{google_link}' target='_blank'>Middle in Google Maps</a><br>
        <a href='{google_link_a1}' target='_blank'>Anchor 1 in Google Maps</a><br>
        <a href='{google_link_a2}' target='_blank'>Anchor 2 in Google Maps</a><br><br>

        <a href='{skrfoto_link}' target='_blank'>Middle skråfoto</a><br>
        <a href='{skrfoto_link_a1}' target='_blank'>Anchor 1 skråfoto</a><br>
        <a href='{skrfoto_link_a2}' target='_blank'>Anchor 2 skråfoto</a><br>
        """
    
        features.append({
            "type": "Feature",
            "properties": {
                "midx": float(row["midx"]),
                "midy": float(row["midy"]),
                "length": float(row["length"]),
                "height": float(row["height"]),
                "color": walkable_color(row["walkable_length"]),
                # "color": score_color(row["score"]),
                "popup": popup_html
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [lon1, lat1],
                    [lon2, lat2]
                ]
            }
        })
    
    
    geojson = {"type": "FeatureCollection", "features": features}
    
    # -------------------------------------------------------
    # CREATE MAP
    # -------------------------------------------------------
    center_lat, center_lon = utm_to_latlon(df["midx"].mean(), df["midy"].mean())
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="cartodbpositron")
    
    gj = folium.GeoJson(
        geojson,
        name="Lines",
        style_function=lambda f: {
            "color": f["properties"]["color"],
            "weight": 7,
            "opacity": 0.9,
        },
        tooltip=folium.GeoJsonTooltip(fields=["midx", "midy"]),
        popup=folium.GeoJsonPopup(fields=["popup"])
    ).add_to(m)
    
    map_id = m.get_name()
    gj_id = gj.get_name()
    
    # -------------------------------------------------------
    # ADD LEAFLET DRAW FOR RECTANGLE SELECTION
    # -------------------------------------------------------
    draw_code = f"""
    <script>
    
    // Real map object created by Folium
    var map = {map_id};
    var gjLayer = {gj_id};
    
    // Store references to line layers
    var lineLayers = [];
    
    gjLayer.eachLayer(function(layer) {{
        lineLayers.push(layer);
    }});
    
    // Add Leaflet Draw control
    var drawnItems = new L.FeatureGroup().addTo(map);
    var drawControl = new L.Control.Draw({{
        draw: {{
            marker: false,
            circle: false,
            polyline: false,
            polygon: false,
            circlemarker: false,
            rectangle: true
        }},
        edit: false
    }});
    map.addControl(drawControl);
    
    // Helper: check if line intersects rectangle
    function lineIntersectsRectangle(line, bounds) {{
        var coords = line.feature.geometry.coordinates;
    
        var p1 = L.latLng(coords[0][1], coords[0][0]);
        var p2 = L.latLng(coords[1][1], coords[1][0]);
    
        // If either endpoint is inside rectangle
        if (bounds.contains(p1) || bounds.contains(p2)) {{
            return true;
        }}
    
        // Check if line crosses any rectangle edge
        var rectPoints = [
            bounds.getNorthWest(),
            bounds.getNorthEast(),
            bounds.getSouthEast(),
            bounds.getSouthWest(),
        ];
    
        // Edges of rectangle
        var edges = [
            [rectPoints[0], rectPoints[1]],
            [rectPoints[1], rectPoints[2]],
            [rectPoints[2], rectPoints[3]],
            [rectPoints[3], rectPoints[0]]
        ];
    
        function segIntersect(a,b,c,d){{
            function ccw(p1,p2,p3){{
                return (p3.lat - p1.lat)*(p2.lng - p1.lng) > (p2.lat - p1.lat)*(p3.lng - p1.lng);
            }}
            return (ccw(a,c,d) != ccw(b,c,d)) && (ccw(a,b,c) != ccw(a,b,d));
        }}
    
        for (var i=0; i<edges.length; i++) {{
            if (segIntersect(p1,p2,edges[i][0],edges[i][1])) {{
                return true;
            }}
        }}
    
        return false;
    }}
    
    // When rectangle is created
    map.on(L.Draw.Event.CREATED, function (e) {{
        var layer = e.layer;
        drawnItems.addLayer(layer);
    
        var bounds = layer.getBounds();
    
        var selected = [];
    
        lineLayers.forEach(function(line) {{
            if (lineIntersectsRectangle(line, bounds)) {{
                selected.push(line.feature.properties);
            }}
        }});
    
        // Compute stats
        if (selected.length > 0) {{
            var heights = selected.map(s => s.height);
            var lengths = selected.map(s => s.length);
    
            var msg = 
                "<b>Selected lines:</b> " + selected.length + "<br>" +
                "<b>Height min:</b> " + Math.min(...heights).toFixed(1) + "<br>" +
                "<b>Height max:</b> " + Math.max(...heights).toFixed(1) + "<br>" +
                "<b>Length min:</b> " + Math.min(...lengths).toFixed(1) + "<br>" +
                "<b>Length max:</b> " + Math.max(...lengths).toFixed(1) + "<br>";
    
            L.popup()
                .setLatLng(bounds.getCenter())
                .setContent(msg)
                .openOn(map);
        }} else {{
            L.popup()
                .setLatLng(bounds.getCenter())
                .setContent("No lines in selection.")
                .openOn(map);
        }}
    }});
    </script>
    """
    
    m.get_root().html.add_child(folium.Element(draw_code))

    # -------------------------------------------------------
    # SAVE OUTPUT
    # -------------------------------------------------------
    m.save(output_html)
    print("Saved:", output_html)


if __name__ == "__main__":
    # -------------------------------------------------------
    # SETTINGS
    # -------------------------------------------------------
    csv_file = sys.argv[1]
    output_html = sys.argv[2]
    score_threshold = float(sys.argv[3])
    hmean_threshold = float(sys.argv[4])
    utm_zone = 32
    hemisphere = "north"
    # -------------------------------------------------------
    
    # -------------------------------------------------------
    # LOAD CSV
    # -------------------------------------------------------
    df = pd.read_csv(csv_file, sep=" ")
    
    save_HL_map(df, output_html, score_threshold, utm_zone, hemisphere, hmean_threshold)

