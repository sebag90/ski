import glob
import io
import json
import os
import sys
from datetime import date, datetime
from flask import Flask, jsonify, request, send_file, send_from_directory
import meteostat as ms
import pandas as pd

# Set Agg backend before importing plotter
import matplotlib

matplotlib.use("Agg")

from plotter import generate_plot

app = Flask(__name__, static_folder=".")

# In-memory cache for generated plots: key -> bytes
PLOT_CACHE = {}

# Directory containing place JSON files
PLACES_DIR = os.path.join(os.path.dirname(__file__), "places")


def get_max_station_distance():
    """Reads STATION_DISTANCE from environment (in km, default 50.0)."""
    raw = os.environ.get("STATION_DISTANCE", "50").strip()
    try:
        val = float(raw)
        # If user supplied meters (> 500), convert to km
        if val > 500:
            val = val / 1000.0
        return val
    except ValueError:
        return 50.0


def load_places():
    """
    Reads all *.json files from PLACES_DIR.
    Each file represents one ski vacation option/location.
    """
    places = []
    if not os.path.exists(PLACES_DIR):
        return places

    for fpath in sorted(glob.glob(os.path.join(PLACES_DIR, "*.json"))):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["_id"] = os.path.splitext(os.path.basename(fpath))[0]
                places.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {fpath}: {e}", file=sys.stderr)

    return places


def compute_stations_for_places(places, max_dist_km):
    """
    Dynamically searches Meteostat for stations within max_dist_km
    of any of the given places. Deduplicates and ranks them.
    """
    stations_dict = {}
    radius_m = int(max_dist_km * 1000)

    for p in places:
        coords = p.get("coords")
        if not coords or len(coords) < 2:
            continue
        lat, lon = coords[0], coords[1]
        pt = ms.Point(lat, lon)

        try:
            nearby = ms.stations.nearby(pt, radius=radius_m, limit=100)
        except Exception as e:
            print(
                f"Warning: nearby query failed for {p.get('name')}: {e}",
                file=sys.stderr,
            )
            continue

        for st_id, row in nearby.iterrows():
            dist_km = round(float(row["distance"]) / 1000.0, 1)
            if dist_km > max_dist_km:
                continue

            sid = str(st_id)
            elev = int(row["elevation"]) if pd.notna(row.get("elevation")) else None
            p_name = p.get("name", "Unknown")
            dist_label = f"{p_name} ({dist_km}km)"

            if sid not in stations_dict:
                stations_dict[sid] = {
                    "id": sid,
                    "name": str(row["name"]),
                    "country": str(row.get("country", "")),
                    "coords": [float(row["latitude"]), float(row["longitude"])],
                    "elevation": elev,
                    "serves": [dist_label],
                    "closest_place": p_name,
                    "min_distance_km": dist_km,
                }
            else:
                if dist_label not in stations_dict[sid]["serves"]:
                    stations_dict[sid]["serves"].append(dist_label)
                if dist_km < stations_dict[sid]["min_distance_km"]:
                    stations_dict[sid]["min_distance_km"] = dist_km
                    stations_dict[sid]["closest_place"] = p_name

    return sorted(stations_dict.values(), key=lambda s: s["min_distance_km"])


# Cache of computed stations: (max_dist_km, places_hash) -> stations
STATION_CACHE = {}


def get_cached_stations(places, max_dist_km):
    places_key = tuple((p.get("name"), tuple(p.get("coords", []))) for p in places)
    cache_key = (max_dist_km, places_key)
    if cache_key not in STATION_CACHE:
        STATION_CACHE[cache_key] = compute_stations_for_places(places, max_dist_km)
    return STATION_CACHE[cache_key]


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/places")
def api_places():
    """Returns all places loaded from places/*.json."""
    places = load_places()
    return jsonify(
        {
            "places": places,
            "count": len(places),
        }
    )


@app.route("/api/stations")
def api_stations():
    """
    Returns weather stations dynamically discovered within STATION_DISTANCE
    (or query param 'distance') from any loaded place.
    """
    custom_dist = request.args.get("distance", type=float)
    max_dist_km = custom_dist if custom_dist is not None else get_max_station_distance()

    places = load_places()
    stations = get_cached_stations(places, max_dist_km)

    return jsonify(
        {
            "stations": stations,
            "count": len(stations),
            "station_distance_km": max_dist_km,
            "places_count": len(places),
        }
    )


@app.route("/api/plot")
def api_plot():
    """
    Generates and returns the 3-panel ski weather overview plot as PNG.
    Query parameters:
      station: station ID (e.g. '07497')
      lat: latitude float (optional)
      lon: longitude float (optional)
      start: YYYY-MM-DD
      end: YYYY-MM-DD
      name: optional display name
    """
    station_id = request.args.get("station")
    start_str = request.args.get("start", "2025-12-01")
    end_str = request.args.get("end", "2026-01-31")
    name = request.args.get("name")

    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)

    if lat is None or lon is None:
        places = load_places()
        stations = get_cached_stations(places, get_max_station_distance())
        for st in stations:
            if st["id"] == station_id:
                lat, lon = st["coords"]
                if not name:
                    name = st["name"]
                break

    if lat is None or lon is None:
        lat, lon = 45.6167, 6.7667

    try:
        start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_str, "%Y-%m-%d").date()
    except ValueError as e:
        return jsonify({"error": f"Invalid date format (use YYYY-MM-DD): {e}"}), 400

    if start_date > end_date:
        return jsonify({"error": "Start date must be before or equal to end date"}), 400

    cache_key = (station_id, lat, lon, start_str, end_str, name or "")
    if cache_key in PLOT_CACHE:
        return send_file(
            io.BytesIO(PLOT_CACHE[cache_key]),
            mimetype="image/png",
            max_age=3600,
        )

    try:
        img_bytes = generate_plot(
            station_id=station_id,
            lat=lat,
            lon=lon,
            start_date=start_date,
            end_date=end_date,
            name=name,
            dpi=140,
        )
        PLOT_CACHE[cache_key] = img_bytes
        return send_file(
            io.BytesIO(img_bytes),
            mimetype="image/png",
            max_age=3600,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    dist = get_max_station_distance()
    print(
        f"Ski Weather Backend running on http://0.0.0.0:{port} (STATION_DISTANCE = {dist} km)"
    )
    app.run(host="0.0.0.0", port=port, debug=False)
