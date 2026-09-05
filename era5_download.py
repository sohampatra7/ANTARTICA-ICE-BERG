import os
import cdsapi
import xarray as xr
import numpy as np

# -----------------------------
# SETTINGS
# -----------------------------

YEAR = 2024

# Weddell Sea
NORTH = -65
WEST = -50
SOUTH = -70
EAST = -40

DATA_DIR = "data"

RAW_FILE = os.path.join(
    DATA_DIR,
    "era5_2024_weddell.nc"
)

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "era5_wind_mslp_2024_weddell.nc"
)

os.makedirs(DATA_DIR, exist_ok=True)


# -----------------------------
# CREATE VALID DATE LIST
# -----------------------------

months = [
    f"{m:02d}"
    for m in range(1, 13)
]

days = [
    f"{d:02d}"
    for d in range(1, 32)
]


# -----------------------------
# DOWNLOAD ERA5
# -----------------------------

if not os.path.exists(RAW_FILE):

    print("Downloading ERA5...")

    client = cdsapi.Client()

    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",

            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "mean_sea_level_pressure"
            ],

            "year": str(YEAR),

            "month": months,

            "day": days,

            # One observation per day
            "time": "12:00",

            # North, West, South, East
            "area": [
                NORTH,
                WEST,
                SOUTH,
                EAST
            ],

            "data_format": "netcdf",

            "download_format": "unarchived"
        },
        RAW_FILE
    )

    print("ERA5 download complete.")

else:

    print("Using existing ERA5 file.")


# -----------------------------
# OPEN DATA
# -----------------------------

print("Opening ERA5 file...")

ds = xr.open_dataset(RAW_FILE)


# -----------------------------
# CALCULATE WIND SPEED
# -----------------------------

u = ds["u10"]
v = ds["v10"]

wind_speed = np.sqrt(
    u**2 + v**2
)

wind_speed.name = "wind_speed"

wind_speed.attrs["units"] = "m s-1"

wind_speed.attrs["description"] = (
    "10 metre wind speed calculated from "
    "ERA5 u10 and v10 components"
)


# -----------------------------
# KEEP USEFUL VARIABLES
# -----------------------------

small = xr.Dataset(
    {
        "u10": u,
        "v10": v,
        "wind_speed": wind_speed,
        "mean_sea_level_pressure": ds["msl"]
    }
)


# -----------------------------
# SAVE
# -----------------------------

small.to_netcdf(
    OUTPUT_FILE,
    format="NETCDF4"
)

print()
print("DONE!")
print("Saved:")
print(OUTPUT_FILE)