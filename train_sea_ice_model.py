"""
train_sea_ice_model.py

Predicts sea-ice concentration from ERA5 wind (u10, v10, wind_speed) and
mean sea level pressure.

WHY THIS NEEDS CARE (read before changing anything):

1. GRID MISMATCH: ERA5 is a regular lat/lon grid (21 x 41). The sea-ice
   data is on a polar-stereographic x/y grid (27 x 27) with 2D lat/lon
   fields -- the grids do NOT line up cell-for-cell. This script matches
   each sea-ice grid cell to its nearest ERA5 grid cell using actual
   lat/lon distance (with longitude scaled by cos(latitude), since
   longitude lines converge near the poles) -- not naive index alignment,
   which would silently pair up the wrong locations.

2. MISSING VALUES: sea-ice cells that are NaN (land mask / no data) are
   dropped BEFORE training, never filled with 0 -- filling with 0 would
   falsely tell the model "no ice" for places that are actually land or
   unobserved.

3. TRAIN/TEST SPLIT: this data is a full year of daily, spatially
   autocorrelated grid cells. A random row-wise split would leak
   information (nearby days/cells are highly similar), inflating your
   reported accuracy. This script splits by TIME instead: trains on
   Jan-Oct, tests on Nov-Dec, so the model is evaluated on genuinely
   unseen future days.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

DATA_DIR = "data"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

ERA5_FILE = os.path.join(DATA_DIR, "era5_wind_mslp_2024_weddell.nc")
SEA_ICE_FILE = os.path.join(DATA_DIR, "sea_ice_2024_weddell.nc")

MODEL_OUTPUT = os.path.join(MODELS_DIR, "sea_ice_model.pkl")

# Time-based split: train on everything before this date, test on/after it
SPLIT_DATE = "2024-11-01"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

era5 = xr.open_dataset(ERA5_FILE)
sea_ice = xr.open_dataset(SEA_ICE_FILE)

sea_ice_var = list(sea_ice.data_vars)[0]
print(f"Sea-ice variable: {sea_ice_var}")
print(f"ERA5 variables: {list(era5.data_vars)}")


# ============================================================
# BUILD NEAREST-NEIGHBOR SPATIAL MATCH
# (sea-ice grid cell -> nearest ERA5 grid cell)
# ============================================================

print()
print("=" * 60)
print("MATCHING GRIDS (sea-ice -> nearest ERA5 cell)")
print("=" * 60)

era5_lat = era5["latitude"].values      # 1D
era5_lon = era5["longitude"].values     # 1D
era5_lat_grid, era5_lon_grid = np.meshgrid(era5_lat, era5_lon, indexing="ij")

# Scale longitude by cos(latitude) so Euclidean distance approximates
# real distance despite longitude lines converging near the poles.
mean_lat_rad = np.radians(era5_lat.mean())
cos_mean_lat = np.cos(mean_lat_rad)

era5_points = np.column_stack([
    era5_lat_grid.ravel(),
    era5_lon_grid.ravel() * cos_mean_lat
])

tree = cKDTree(era5_points)

sea_ice_lat = sea_ice["latitude"].values   # 2D (y, x)
sea_ice_lon = sea_ice["longitude"].values  # 2D (y, x)

sea_ice_points = np.column_stack([
    sea_ice_lat.ravel(),
    sea_ice_lon.ravel() * cos_mean_lat
])

# For each sea-ice grid cell, index of nearest ERA5 grid cell (flattened)
_, nearest_era5_flat_idx = tree.query(sea_ice_points)

# Convert flat ERA5 index back to (lat_idx, lon_idx)
n_era5_lat = len(era5_lat)
n_era5_lon = len(era5_lon)
nearest_lat_idx = nearest_era5_flat_idx // n_era5_lon
nearest_lon_idx = nearest_era5_flat_idx % n_era5_lon

print(f"Sea-ice grid: {sea_ice_lat.shape}, ERA5 grid: {era5_lat_grid.shape}")
print(f"Matched {len(sea_ice_points)} sea-ice cells to nearest ERA5 cells")


# ============================================================
# BUILD THE TRAINING TABLE (loop over days)
# ============================================================

print()
print("=" * 60)
print("BUILDING FEATURE TABLE")
print("=" * 60)

# .normalize() strips the time-of-day component (ERA5 uses 12:00,
# sea-ice uses midnight) so we match by calendar date only.
sea_ice_dates = pd.to_datetime(sea_ice["time"].values).normalize()
era5_dates = pd.to_datetime(era5["valid_time"].values).normalize()

# Sanity check: dates should match 1:1 since both are daily 2024 data
common_dates = sea_ice_dates.intersection(era5_dates)
print(f"Sea-ice days: {len(sea_ice_dates)}, ERA5 days: {len(era5_dates)}, "
      f"common days: {len(common_dates)}")

rows = []

u10_all = era5["u10"].values            # (time, lat, lon)
v10_all = era5["v10"].values
wind_speed_all = era5["wind_speed"].values
mslp_all = era5["mean_sea_level_pressure"].values

sea_ice_all = sea_ice[sea_ice_var].values  # (time, y, x)

for t_idx, date in enumerate(sea_ice_dates):

    if date not in era5_dates:
        continue

    era5_t_idx = era5_dates.get_loc(date)

    ice_slice = sea_ice_all[t_idx].ravel()  # (y*x,)

    valid = ~np.isnan(ice_slice)
    if valid.sum() == 0:
        continue

    lat_idx_valid = nearest_lat_idx[valid]
    lon_idx_valid = nearest_lon_idx[valid]

    u10_vals = u10_all[era5_t_idx][lat_idx_valid, lon_idx_valid]
    v10_vals = v10_all[era5_t_idx][lat_idx_valid, lon_idx_valid]
    wind_speed_vals = wind_speed_all[era5_t_idx][lat_idx_valid, lon_idx_valid]
    mslp_vals = mslp_all[era5_t_idx][lat_idx_valid, lon_idx_valid]

    rows.append(pd.DataFrame({
        "date": date,
        "day_of_year": date.dayofyear,
        "latitude": sea_ice_lat.ravel()[valid],
        "longitude": sea_ice_lon.ravel()[valid],
        "u10": u10_vals,
        "v10": v10_vals,
        "wind_speed": wind_speed_vals,
        "mean_sea_level_pressure": mslp_vals,
        "sea_ice_concentration": ice_slice[valid]
    }))

table = pd.concat(rows, ignore_index=True)

print(f"\nFinal training table: {len(table)} rows")
print(table.head())
print(f"\nAny remaining NaNs?\n{table.isna().sum()}")


# ============================================================
# TIME-BASED TRAIN/TEST SPLIT
# ============================================================

print()
print("=" * 60)
print("SPLITTING TRAIN/TEST BY TIME")
print("=" * 60)

train = table[table["date"] < SPLIT_DATE]
test = table[table["date"] >= SPLIT_DATE]

print(f"Train: {len(train)} rows ({train['date'].min().date()} to "
      f"{train['date'].max().date()})")
print(f"Test:  {len(test)} rows ({test['date'].min().date()} to "
      f"{test['date'].max().date()})")

feature_cols = [
    "u10", "v10", "wind_speed", "mean_sea_level_pressure",
    "latitude", "longitude", "day_of_year"
]

X_train = train[feature_cols]
y_train = train["sea_ice_concentration"]

X_test = test[feature_cols]
y_test = test["sea_ice_concentration"]


# ============================================================
# TRAIN MODEL
# ============================================================

print()
print("=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

pred_test = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred_test))
r2 = r2_score(y_test, pred_test)

print(f"\nTest RMSE: {rmse:.4f}")
print(f"Test R^2:  {r2:.4f}")

print("\nFeature importances:")
for name, importance in sorted(
    zip(feature_cols, model.feature_importances_),
    key=lambda x: -x[1]
):
    print(f"  {name}: {importance:.3f}")


# ============================================================
# SAVE
# ============================================================

joblib.dump(model, MODEL_OUTPUT)

print()
print("=" * 60)
print("DONE")
print("=" * 60)
print(f"Saved model: {MODEL_OUTPUT}")