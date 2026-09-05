import os
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# FILE LOCATIONS
# ============================================================

DATA_DIR = "data"

SEA_ICE_FILE = os.path.join(
    DATA_DIR,
    "sea_ice_2024_weddell.nc"
)

ICEBERG_FILE = os.path.join(
    DATA_DIR,
    "iceberg_tracks_5.csv"
)

ERA5_FILE = os.path.join(
    DATA_DIR,
    "era5_wind_mslp_2024_weddell.nc"
)


# ============================================================
# 1. LOAD THE FILES
# ============================================================

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

# Sea ice
sea_ice = xr.open_dataset(SEA_ICE_FILE)

# Icebergs
icebergs = pd.read_csv(ICEBERG_FILE)

# ERA5
era5 = xr.open_dataset(ERA5_FILE)


print("\nSea-ice dataset:")
print(sea_ice)

print("\nIceberg dataset:")
print(icebergs.head())

print("\nERA5 dataset:")
print(era5)


# ============================================================
# 2. BASIC STATISTICS
# ============================================================

print("\n")
print("=" * 60)
print("BASIC STATISTICS")
print("=" * 60)


# ---------- SEA ICE ----------

print("\nSEA ICE")

# Find the first data variable
sea_ice_variables = list(sea_ice.data_vars)

print("Variables:", sea_ice_variables)

sea_ice_var = sea_ice_variables[0]

values = sea_ice[sea_ice_var].values

print("Minimum:", np.nanmin(values))
print("Maximum:", np.nanmax(values))
print("Average:", np.nanmean(values))
print("Missing values:", np.isnan(values).sum())


# ---------- ICEBERGS ----------

print("\nICEBERGS")

print("Number of rows:", len(icebergs))
print("Columns:", list(icebergs.columns))

numeric_columns = icebergs.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:

    values = icebergs[column]

    print("\n", column)

    print("Minimum:", values.min())
    print("Maximum:", values.max())
    print("Average:", values.mean())
    print("Missing values:", values.isna().sum())


# ---------- ERA5 ----------

print("\nERA5")

print("Variables:", list(era5.data_vars))

for variable in era5.data_vars:

    values = era5[variable].values

    print("\n", variable)

    print("Minimum:", np.nanmin(values))
    print("Maximum:", np.nanmax(values))
    print("Average:", np.nanmean(values))
    print("Missing values:", np.isnan(values).sum())


# ============================================================
# 3. SEA-ICE CONCENTRATION OVER TIME
# ============================================================

print("\n")
print("=" * 60)
print("CREATING SEA-ICE TIME SERIES")
print("=" * 60)

# Find the sea-ice variable
sea_ice_var = list(sea_ice.data_vars)[0]

data = sea_ice[sea_ice_var]

# Find the middle grid cell
# This avoids needing to know the exact coordinate names.
sizes = data.sizes

spatial_dims = [
    dim for dim in sizes
    if dim != "time"
]

if len(spatial_dims) >= 2:

    y_dim = spatial_dims[-2]
    x_dim = spatial_dims[-1]

    y_index = sizes[y_dim] // 2
    x_index = sizes[x_dim] // 2

    one_cell = data.isel(
        {
            y_dim: y_index,
            x_dim: x_index
        }
    )

else:

    one_cell = data


plt.figure(figsize=(10, 5))

one_cell.plot()

plt.title(
    "Sea-Ice Concentration Over Time\n"
    "One Grid Cell"
)

plt.xlabel("Date")
plt.ylabel("Sea-Ice Concentration")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        DATA_DIR,
        "sea_ice_time_series.png"
    )
)

plt.show()


# ============================================================
# 4. ICEBERG TRACK PLOT
# ============================================================

print("\n")
print("=" * 60)
print("CREATING ICEBERG TRACK")
print("=" * 60)

# Make sure date is interpreted as a date
icebergs["date"] = pd.to_datetime(
    icebergs["date"],
    errors="coerce"
)

# Select the first iceberg
iceberg_name = icebergs["iceberg"].iloc[0]

track = icebergs[
    icebergs["iceberg"] == iceberg_name
].copy()

track = track.sort_values("date")

print(
    "Plotting iceberg:",
    iceberg_name
)

print(
    "Number of positions:",
    len(track)
)


plt.figure(figsize=(8, 8))

plt.plot(
    track["longitude"],
    track["latitude"],
    marker="o",
    markersize=3
)

plt.scatter(
    track["longitude"].iloc[0],
    track["latitude"].iloc[0],
    s=100,
    label="Start"
)

plt.scatter(
    track["longitude"].iloc[-1],
    track["latitude"].iloc[-1],
    s=100,
    label="End"
)

plt.title(
    "Iceberg Track: " +
    str(iceberg_name)
)

plt.xlabel("Longitude (degrees)")
plt.ylabel("Latitude (degrees)")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        DATA_DIR,
        "iceberg_track.png"
    )
)

plt.show()


# ============================================================
# 5. FINISHED
# ============================================================

print("\n")
print("=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)

print("\nCreated:")
print("1. data/sea_ice_time_series.png")
print("2. data/iceberg_track.png")