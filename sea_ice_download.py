import os
import re
import requests
from bs4 import BeautifulSoup
import xarray as xr

# -----------------------------
# SETTINGS
# -----------------------------

YEAR = 2024

# Weddell Sea region
NORTH = -65
SOUTH = -70
WEST = -50
EAST = -40

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")

os.makedirs(RAW_DIR, exist_ok=True)

NSIDC_ROOT = "https://noaadata.apps.nsidc.org/NOAA/G02202_V6/south/aggregate/"

RAW_FILE = os.path.join(
    RAW_DIR,
    f"sea_ice_{YEAR}_full.nc"
)

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    f"sea_ice_{YEAR}_weddell.nc"
)


# -----------------------------
# FIND THE 2024 FILE
# -----------------------------

def find_file(url, pattern, depth=0, max_depth=3):

    if depth > max_depth:
        return None

    print("Checking:", url)

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for link in soup.find_all("a", href=True):

        href = link["href"]

        if href.startswith("../"):
            continue

        full_url = requests.compat.urljoin(url, href)

        # Is this the file we want?
        if re.search(pattern, href):
            return full_url

        # Search inside directories
        if href.endswith("/"):
            result = find_file(
                full_url,
                pattern,
                depth + 1,
                max_depth
            )

            if result:
                return result

    return None


# -----------------------------
# DOWNLOAD FILE
# -----------------------------

if os.path.exists(RAW_FILE):

    print("Using existing downloaded NSIDC file.")

else:

    print("Searching NSIDC for the 2024 Antarctic file...")

    pattern = rf"sic_pss25_.*2024.*v06r00.*\.nc"

    file_url = find_file(
        NSIDC_ROOT,
        pattern
    )

    if file_url is None:
        raise RuntimeError(
            "Could not automatically find the NSIDC 2024 file.\n"
            "Download the 2024 Southern Hemisphere yearly file manually "
            "from the NSIDC archive and place it in data/raw/sea_ice_2024_full.nc"
        )

    print("Downloading:")
    print(file_url)

    with requests.get(file_url, stream=True, timeout=120) as r:

        r.raise_for_status()

        with open(RAW_FILE, "wb") as f:

            for chunk in r.iter_content(chunk_size=1024 * 1024):

                if chunk:
                    f.write(chunk)

    print("Download complete.")


# -----------------------------
# OPEN DATA
# -----------------------------

print("Opening NetCDF file...")

ds = xr.open_dataset(
    RAW_FILE,
    group=None
)

# The latitude/longitude coordinates in the aggregated files
# are stored in the cdr_supplementary group.

coords = xr.open_dataset(
    RAW_FILE,
    group="cdr_supplementary"
)

lat = coords["latitude"]
lon = coords["longitude"]

# Add geographic coordinates to the main dataset
ds = ds.assign_coords(
    latitude=(("y", "x"), lat.values),
    longitude=(("y", "x"), lon.values)
)

# -----------------------------
# SELECT WEDDELL SEA
# -----------------------------

mask = (
    (ds.latitude >= SOUTH) &
    (ds.latitude <= NORTH) &
    (ds.longitude >= WEST) &
    (ds.longitude <= EAST)
)

small = ds["cdr_seaice_conc"].where(
    mask,
    drop=True
)

# -----------------------------
# SAVE SMALL DATASET
# -----------------------------

small = small.to_dataset(name="sea_ice_concentration")

small.to_netcdf(
    OUTPUT_FILE,
    format="NETCDF4"
)

print()
print("DONE!")
print("Output file:")
print(OUTPUT_FILE)