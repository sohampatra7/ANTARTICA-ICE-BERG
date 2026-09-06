import os
import zipfile
import pandas as pd

# -----------------------------
# SETTINGS
# -----------------------------

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "iceberg_tracks_5.csv"
)

MIN_DAYS = 180
NUMBER_OF_ICEBERGS = 5

# Must match the region covered by your sea-ice / ERA5 downloads
# (Weddell Sea box).
REGION_LAT_MIN = -70.0
REGION_LAT_MAX = -65.0
REGION_LON_MIN = -50.0
REGION_LON_MAX = -40.0

os.makedirs(RAW_DIR, exist_ok=True)


# -----------------------------
# FIND BYU ZIP FILE
# -----------------------------

zip_files = [
    os.path.join(RAW_DIR, f)
    for f in os.listdir(RAW_DIR)
    if f.lower().endswith(".zip")
]

if not zip_files:

    raise RuntimeError(
        "No ZIP file found.\n\n"
        "Download the BYU/NIC Statistical Database ZIP and "
        "put it inside data/raw/"
    )

zip_file = zip_files[0]

print("Using:")
print(zip_file)


# -----------------------------
# READ ALL CSV FILES
# -----------------------------

candidate_tracks = []

with zipfile.ZipFile(zip_file, "r") as z:

    csv_files = [
        name for name in z.namelist()
        if name.lower().endswith(".csv")
    ]

    print("CSV files found:", len(csv_files))

    for filename in csv_files:

        try:

            with z.open(filename) as f:
                df = pd.read_csv(f)

            # Find date column
            date_columns = [
                c for c in df.columns
                if "date" in c.lower()
            ]

            # Find latitude column
            lat_columns = [
                c for c in df.columns
                if "lat" in c.lower()
            ]

            # Find longitude column
            lon_columns = [
                c for c in df.columns
                if "lon" in c.lower()
            ]

            if not date_columns or not lat_columns or not lon_columns:
                continue

            date_col = date_columns[0]
            lat_col = lat_columns[0]
            lon_col = lon_columns[0]

            # Convert YYYYDDD into normal dates
            dates = pd.to_datetime(
                df[date_col].astype(str),
                format="%Y%j",
                errors="coerce"
            )

            df["date_clean"] = dates

            df["latitude_clean"] = pd.to_numeric(
                df[lat_col],
                errors="coerce"
            )

            df["longitude_clean"] = pd.to_numeric(
                df[lon_col],
                errors="coerce"
            )

            df = df.dropna(
                subset=[
                    "date_clean",
                    "latitude_clean",
                    "longitude_clean"
                ]
            )

            if len(df) < 2:
                continue

            # ---------------------------------------------------
            # SPATIAL FILTER (NEW)
            # Only keep this iceberg if at least one of its
            # tracked positions ever falls inside the Weddell Sea
            # box that your sea-ice / ERA5 data covers.
            # ---------------------------------------------------

            in_region = df[
                (df["latitude_clean"] >= REGION_LAT_MIN) &
                (df["latitude_clean"] <= REGION_LAT_MAX) &
                (df["longitude_clean"] >= REGION_LON_MIN) &
                (df["longitude_clean"] <= REGION_LON_MAX)
            ]

            if len(in_region) == 0:
                continue

            start = df["date_clean"].min()
            end = df["date_clean"].max()

            duration = (end - start).days

            if duration >= MIN_DAYS:

                iceberg_name = os.path.basename(
                    filename
                ).replace(".csv", "")

                candidate_tracks.append(
                    {
                        "iceberg": iceberg_name,
                        "duration_days": duration,
                        "data": df,
                        "date_col": date_col,
                        "days_in_region": len(in_region)
                    }
                )

        except Exception:
            continue


# -----------------------------
# SELECT LONGEST TRACKS
# (among icebergs that actually pass through the region)
# -----------------------------

candidate_tracks.sort(
    key=lambda x: x["duration_days"],
    reverse=True
)

selected = candidate_tracks[
    :NUMBER_OF_ICEBERGS
]

if len(selected) == 0:

    raise RuntimeError(
        "No icebergs found that pass through the Weddell Sea "
        "region at all. Check REGION_LAT_MIN/MAX and "
        "REGION_LON_MIN/MAX against your sea-ice/ERA5 bounds."
    )

if len(selected) < NUMBER_OF_ICEBERGS:

    print(
        f"WARNING: only {len(selected)} icebergs found that pass "
        f"through the region with at least {MIN_DAYS} days of "
        "total track length. Proceeding with fewer icebergs than "
        "requested."
    )


print()
print("Selected icebergs (all pass through the Weddell Sea box):")

for item in selected:

    print(
        item["iceberg"],
        "-",
        item["duration_days"],
        "days total, ",
        item["days_in_region"],
        "positions inside the region"
    )


# -----------------------------
# COMBINE TRACKS
# -----------------------------

output_frames = []

for item in selected:

    df = item["data"].copy()

    output = pd.DataFrame()

    # Assign row-length columns FIRST so the scalar iceberg name
    # broadcasts correctly (see earlier bug fix).
    output["date"] = df["date_clean"].values
    output["latitude"] = df["latitude_clean"].values
    output["longitude"] = df["longitude_clean"].values

    output["iceberg"] = item["iceberg"]

    size_col_found = None
    for c in df.columns:
        name = c.lower()
        if "size" in name:
            size_col_found = c
            output["size"] = pd.to_numeric(
                df[c],
                errors="coerce"
            )
            break

    rotation_col_found = None
    for c in df.columns:
        name = c.lower()
        if "rotation" in name:
            rotation_col_found = c
            output["rotation"] = pd.to_numeric(
                df[c],
                errors="coerce"
            )
            break

    print(
        f"[{item['iceberg']}] "
        f"size_col='{size_col_found}' "
        f"rotation_col='{rotation_col_found}'"
    )

    output_frames.append(output)


result = pd.concat(
    output_frames,
    ignore_index=True
)

result = result.sort_values(
    ["iceberg", "date"]
)

result.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("DONE!")
print("Saved:")
print(OUTPUT_FILE)

# -----------------------------
# QUICK VERIFICATION
# -----------------------------
print()
print("VERIFICATION")
print("iceberg NaN count:", result["iceberg"].isna().sum())
print("iceberg unique values:", result["iceberg"].unique())
if "size" in result.columns:
    print("size min/max:", result["size"].min(), result["size"].max())
    print("negative size rows:", (result["size"] < 0).sum())

for name, group in result.groupby("iceberg"):
    in_region = group[
        (group["latitude"] >= REGION_LAT_MIN) &
        (group["latitude"] <= REGION_LAT_MAX) &
        (group["longitude"] >= REGION_LON_MIN) &
        (group["longitude"] <= REGION_LON_MAX)
    ]
    print(
        f"{name}: {len(group)} total rows, "
        f"{len(in_region)} rows inside the Weddell Sea box, "
        f"date range {group['date'].min().date()} to {group['date'].max().date()}"
    )