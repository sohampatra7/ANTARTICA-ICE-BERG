"""
train_sea_ice_weekly_delta_model.py

Predicts WEEK-TO-WEEK change in sea-ice concentration from that week's
average wind/pressure, instead of noisy day-to-day change.

WHY THIS CHANGE:
The daily-delta model (train_sea_ice_delta_model.py) showed median and
75th-percentile delta of exactly 0.0 -- most single-day changes are
dominated by retrieval noise, not real physical signal. The model ended
up performing WORSE than a naive "no change" baseline (R^2 = -0.57).

Averaging to weekly resolution smooths out day-to-day sensor noise while
still capturing genuine wind-driven trends (compaction, drift, melt
onset) -- this is standard practice in sea-ice/atmosphere research,
where daily reanalysis-vs-satellite comparisons are known to be noisy.
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
MODEL_OUTPUT = os.path.join(MODELS_DIR, "sea_ice_weekly_delta_model.pkl")

SPLIT_DATE = "2024-11-01"


# ============================================================
# LOAD + SPATIAL MATCH (identical to previous scripts)
# ============================================================

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

era5 = xr.open_dataset(ERA5_FILE)
sea_ice = xr.open_dataset(SEA_ICE_FILE)
sea_ice_var = list(sea_ice.data_vars)[0]

era5_lat = era5["latitude"].values
era5_lon = era5["longitude"].values
era5_lat_grid, era5_lon_grid = np.meshgrid(era5_lat, era5_lon, indexing="ij")

cos_mean_lat = np.cos(np.radians(era5_lat.mean()))

era5_points = np.column_stack([
    era5_lat_grid.ravel(), era5_lon_grid.ravel() * cos_mean_lat
])
tree = cKDTree(era5_points)

sea_ice_lat = sea_ice["latitude"].values
sea_ice_lon = sea_ice["longitude"].values

sea_ice_points = np.column_stack([
    sea_ice_lat.ravel(), sea_ice_lon.ravel() * cos_mean_lat
])
_, nearest_flat_idx = tree.query(sea_ice_points)

n_era5_lon = len(era5_lon)
nearest_lat_idx = nearest_flat_idx // n_era5_lon
nearest_lon_idx = nearest_flat_idx % n_era5_lon

sea_ice_dates = pd.to_datetime(sea_ice["time"].values).normalize()
era5_dates = pd.to_datetime(era5["valid_time"].values).normalize()
print(f"Sea-ice days: {len(sea_ice_dates)}, ERA5 days: {len(era5_dates)}, "
      f"common: {len(sea_ice_dates.intersection(era5_dates))}")


# ============================================================
# RESAMPLE TO WEEKLY RESOLUTION, PER CELL
# ============================================================

print()
print("=" * 60)
print("AGGREGATING TO WEEKLY RESOLUTION")
print("=" * 60)

sea_ice_all = sea_ice[sea_ice_var].values
u10_all = era5["u10"].values
v10_all = era5["v10"].values
wind_speed_all = era5["wind_speed"].values
mslp_all = era5["mean_sea_level_pressure"].values

n_days, ny, nx = sea_ice_all.shape
sea_ice_flat = sea_ice_all.reshape(n_days, ny * nx)

valid_mask_flat = ~np.isnan(sea_ice_flat[0])
valid_indices = np.where(valid_mask_flat)[0]
print(f"Valid ocean cells: {len(valid_indices)} / {ny * nx}")

cell_lat_idx = nearest_lat_idx[valid_indices]
cell_lon_idx = nearest_lon_idx[valid_indices]
cell_lats = sea_ice_lat.ravel()[valid_indices]
cell_lons = sea_ice_lon.ravel()[valid_indices]

conc_ts = sea_ice_flat[:, valid_indices]  # (366, n_cells)

# Assign each day to a week number (0, 1, 2, ...)
week_number = ((sea_ice_dates - sea_ice_dates[0]).days // 7)
n_weeks = week_number.max() + 1

era5_idx_by_day = np.array([era5_dates.get_loc(d) for d in sea_ice_dates])

weekly_conc = np.zeros((n_weeks, conc_ts.shape[1]))
weekly_u10 = np.zeros((n_weeks, len(cell_lat_idx)))
weekly_v10 = np.zeros((n_weeks, len(cell_lat_idx)))
weekly_wind_speed = np.zeros((n_weeks, len(cell_lat_idx)))
weekly_mslp = np.zeros((n_weeks, len(cell_lat_idx)))
weekly_mid_date = []

for w in range(n_weeks):
    day_mask = (week_number == w)
    day_positions = np.where(day_mask)[0]

    weekly_conc[w] = conc_ts[day_positions].mean(axis=0)

    era5_positions = era5_idx_by_day[day_positions]
    weekly_u10[w] = u10_all[era5_positions][:, cell_lat_idx, cell_lon_idx].mean(axis=0)
    weekly_v10[w] = v10_all[era5_positions][:, cell_lat_idx, cell_lon_idx].mean(axis=0)
    weekly_wind_speed[w] = wind_speed_all[era5_positions][:, cell_lat_idx, cell_lon_idx].mean(axis=0)
    weekly_mslp[w] = mslp_all[era5_positions][:, cell_lat_idx, cell_lon_idx].mean(axis=0)

    weekly_mid_date.append(sea_ice_dates[day_positions[len(day_positions) // 2]])

weekly_mid_date = pd.DatetimeIndex(weekly_mid_date)

print(f"Weeks: {n_weeks}")


# ============================================================
# BUILD WEEK-TO-WEEK DELTA TABLE
# ============================================================

weekly_delta = weekly_conc[1:] - weekly_conc[:-1]
weekly_conc_prev = weekly_conc[:-1]

rows = []
for w in range(weekly_delta.shape[0]):

    date = weekly_mid_date[w + 1]
    doy = date.dayofyear
    doy_sin = np.sin(2 * np.pi * doy / 365.25)
    doy_cos = np.cos(2 * np.pi * doy / 365.25)

    rows.append(pd.DataFrame({
        "date": date,
        "latitude": cell_lats,
        "longitude": cell_lons,
        "u10": weekly_u10[w + 1],
        "v10": weekly_v10[w + 1],
        "wind_speed": weekly_wind_speed[w + 1],
        "mean_sea_level_pressure": weekly_mslp[w + 1],
        "day_of_year_sin": doy_sin,
        "day_of_year_cos": doy_cos,
        "concentration_prev_week": weekly_conc_prev[w],
        "delta_concentration": weekly_delta[w]
    }))

table = pd.concat(rows, ignore_index=True)

print(f"\nFinal weekly training table: {len(table)} rows")
print(f"Weekly delta stats:\n{table['delta_concentration'].describe()}")


# ============================================================
# TIME-BASED TRAIN/TEST SPLIT
# ============================================================

print()
print("=" * 60)
print("SPLITTING TRAIN/TEST BY TIME")
print("=" * 60)

train = table[table["date"] < SPLIT_DATE]
test = table[table["date"] >= SPLIT_DATE]

print(f"Train: {len(train)} rows, Test: {len(test)} rows")

feature_cols = [
    "u10", "v10", "wind_speed", "mean_sea_level_pressure",
    "latitude", "longitude", "day_of_year_sin", "day_of_year_cos",
    "concentration_prev_week"
]

X_train, y_train = train[feature_cols], train["delta_concentration"]
X_test, y_test = test[feature_cols], test["delta_concentration"]


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 60)
print("TRAINING RANDOM FOREST (weekly delta)")
print("=" * 60)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
pred_test = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred_test))
r2 = r2_score(y_test, pred_test)
baseline_rmse = np.sqrt(mean_squared_error(y_test, np.zeros_like(y_test)))

print(f"\nTest RMSE (model):    {rmse:.5f}")
print(f"Test RMSE (baseline): {baseline_rmse:.5f}")
print(f"Test R^2:             {r2:.4f}")

print("\nFeature importances:")
for name, importance in sorted(
    zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]
):
    print(f"  {name}: {importance:.3f}")

joblib.dump(model, MODEL_OUTPUT)
print(f"\nSaved model: {MODEL_OUTPUT}")