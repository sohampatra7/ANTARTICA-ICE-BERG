"""
predict_iceberg_drift_2024.py

WHAT THIS DOES
---------------
Your 5 selected icebergs (from iceberg_download.py) all stop being tracked
before 2024, but your sea-ice and ERA5 datasets are 2024-only. This script
fills that gap by PREDICTING where each iceberg likely was in 2024, using:

  1. BACKGROUND DRIFT: each iceberg's own average daily lat/lon movement,
     estimated from its last N real days of tracked motion (captures
     ocean currents + its own inertia/momentum).

  2. WIND PUSH: using real 2024 ERA5 wind (u10, v10) at the iceberg's
     current predicted position each day, added on top of the background
     drift. Icebergs are commonly modeled as drifting at ~2% of surface
     wind speed, deflected to the LEFT of the wind direction in the
     Southern Hemisphere (a simplified Ekman/Coriolis effect). This is a
     standard simplification used in iceberg drift literature -- not an
     exact physical simulation.

IMPORTANT HONESTY NOTE FOR YOUR WRITEUP
-----------------------------------------
These are MODELED / PREDICTED positions, not real satellite observations.
Real tracking for these icebergs stopped in 2020/2023 -- they may have
broken apart, melted, or drifted somewhere totally different by 2024.
Every predicted row is tagged predicted=True in the output so it's never
confused with real data. Say this plainly in your presentation.

OUTPUT
------
data/iceberg_tracks_5_with_predictions.csv
  - Same columns as the original file, plus a `predicted` column
    (False = real observation, True = modeled).
"""

import os
import numpy as np
import pandas as pd
import xarray as xr

# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = "data"

ICEBERG_FILE = os.path.join(DATA_DIR, "iceberg_tracks_5.csv")
ERA5_FILE = os.path.join(DATA_DIR, "era5_wind_mslp_2024_weddell.nc")

OUTPUT_FILE = os.path.join(DATA_DIR, "iceberg_tracks_5_with_predictions.csv")

# How many of the iceberg's last real days to use when estimating its
# "background" drift velocity (currents + momentum).
BACKGROUND_WINDOW_DAYS = 60

# Fraction of wind speed that becomes iceberg drift speed.
# Literature range is roughly 1-3%; 2% is a common default.
WIND_FACTOR = 0.02

# Degrees to rotate the wind vector when converting it to a drift
# direction. Southern Hemisphere surface drift is deflected LEFT of the
# wind (Ekman effect), commonly 0-40 degrees; 20 is a reasonable middle.
DEFLECTION_DEG = 20.0

# Meters per degree of latitude (approx, used for unit conversion).
METERS_PER_DEG_LAT = 111_320.0

SECONDS_PER_DAY = 86_400.0


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

icebergs = pd.read_csv(ICEBERG_FILE, parse_dates=["date"])
era5 = xr.open_dataset(ERA5_FILE)

era5_lat_min = float(era5["latitude"].min())
era5_lat_max = float(era5["latitude"].max())
era5_lon_min = float(era5["longitude"].min())
era5_lon_max = float(era5["longitude"].max())

era5_time_min = pd.Timestamp(era5["valid_time"].min().values)
era5_time_max = pd.Timestamp(era5["valid_time"].max().values)

print(f"ERA5 covers lat [{era5_lat_min}, {era5_lat_max}], "
      f"lon [{era5_lon_min}, {era5_lon_max}]")
print(f"ERA5 time range: {era5_time_min.date()} to {era5_time_max.date()}")


def get_wind_at(lat, lon, date):
    """Nearest ERA5 wind vector (u10, v10) at given lat/lon/date."""
    point = era5.sel(
        latitude=lat,
        longitude=lon,
        valid_time=date,
        method="nearest"
    )
    return float(point["u10"].values), float(point["v10"].values)


def in_era5_bounds(lat, lon):
    return (era5_lat_min <= lat <= era5_lat_max) and \
           (era5_lon_min <= lon <= era5_lon_max)


def rotate_vector(u, v, degrees):
    """Rotate wind vector (u=east, v=north) counterclockwise (=left)
    by `degrees`, to approximate Southern Hemisphere drift deflection."""
    theta = np.radians(degrees)
    u_rot = u * np.cos(theta) - v * np.sin(theta)
    v_rot = u * np.sin(theta) + v * np.cos(theta)
    return u_rot, v_rot


# ============================================================
# PREDICT FORWARD FOR EACH ICEBERG
# ============================================================

print()
print("=" * 60)
print("PREDICTING 2024 POSITIONS")
print("=" * 60)

predicted_frames = []

for name, group in icebergs.groupby("iceberg"):

    group = group.sort_values("date").reset_index(drop=True)

    last_row = group.iloc[-1]
    last_date = last_row["date"]
    lat = float(last_row["latitude"])
    lon = float(last_row["longitude"])
    last_size = last_row["size"] if "size" in group.columns else np.nan

    # ---- Estimate background drift velocity (deg/day) ----
    recent = group[
        group["date"] >= last_date - pd.Timedelta(days=BACKGROUND_WINDOW_DAYS)
    ].sort_values("date")

    if len(recent) >= 2:
        days_span = (recent["date"].iloc[-1] - recent["date"].iloc[0]).days
        days_span = max(days_span, 1)
        bg_dlat = (recent["latitude"].iloc[-1] - recent["latitude"].iloc[0]) / days_span
        bg_dlon = (recent["longitude"].iloc[-1] - recent["longitude"].iloc[0]) / days_span
    else:
        bg_dlat, bg_dlon = 0.0, 0.0

    print(f"\n[{name}] last real position: {last_date.date()} "
          f"({lat:.3f}, {lon:.3f})")
    print(f"  background drift: {bg_dlat:.5f} deg lat/day, "
          f"{bg_dlon:.5f} deg lon/day")

    # ---- Step 1: bridge the gap with background drift only ----
    # (no ERA5 wind exists before era5_time_min)
    sim_date = last_date + pd.Timedelta(days=1)
    rows = []
    out_of_bounds_flagged = False

    while sim_date < era5_time_min:
        lat += bg_dlat
        lon += bg_dlon
        sim_date += pd.Timedelta(days=1)
        # not saving these bridge-gap rows: they're outside your 2024
        # analysis window anyway and carry no wind information

    # ---- Step 2: walk through 2024 using background + wind drift ----
    n_predicted = 0
    while sim_date <= era5_time_max:

        if in_era5_bounds(lat, lon):
            u, v = get_wind_at(lat, lon, sim_date)
            u_def, v_def = rotate_vector(u, v, DEFLECTION_DEG)

            drift_u_ms = u_def * WIND_FACTOR
            drift_v_ms = v_def * WIND_FACTOR

            meters_per_deg_lon = METERS_PER_DEG_LAT * np.cos(np.radians(lat))
            wind_dlat = (drift_v_ms * SECONDS_PER_DAY) / METERS_PER_DEG_LAT
            wind_dlon = (drift_u_ms * SECONDS_PER_DAY) / meters_per_deg_lon
        else:
            if not out_of_bounds_flagged:
                print(f"  -> drifted outside ERA5 grid on {sim_date.date()}, "
                      f"continuing with background drift only")
                out_of_bounds_flagged = True
            wind_dlat, wind_dlon = 0.0, 0.0

        lat += bg_dlat + wind_dlat
        lon += bg_dlon + wind_dlon

        rows.append({
            "iceberg": name,
            "date": sim_date,
            "latitude": lat,
            "longitude": lon,
            "size": last_size,
            "predicted": True
        })

        n_predicted += 1
        sim_date += pd.Timedelta(days=1)

    print(f"  -> generated {n_predicted} predicted days in 2024")

    predicted_frames.append(pd.DataFrame(rows))


# ============================================================
# COMBINE REAL + PREDICTED, SAVE
# ============================================================

icebergs["predicted"] = False

all_predicted = pd.concat(predicted_frames, ignore_index=True)

result = pd.concat([icebergs, all_predicted], ignore_index=True)
result = result.sort_values(["iceberg", "date"]).reset_index(drop=True)

result.to_csv(OUTPUT_FILE, index=False)

print()
print("=" * 60)
print("DONE")
print("=" * 60)
print(f"Saved: {OUTPUT_FILE}")
print(f"Total rows: {len(result)} "
      f"({(~result['predicted']).sum()} real, {result['predicted'].sum()} predicted)")
