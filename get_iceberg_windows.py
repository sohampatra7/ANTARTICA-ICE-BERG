import pandas as pd

REGION_LAT_MIN = -70.0
REGION_LAT_MAX = -65.0
REGION_LON_MIN = -50.0
REGION_LON_MAX = -40.0

# Extra days padded on each side of the real in-region window, so we
# capture the approach/exit too, not just the exact box-crossing days.
PADDING_DAYS = 14

df = pd.read_csv("data/iceberg_tracks_5.csv", parse_dates=["date"])

in_region = df[
    (df["latitude"] >= REGION_LAT_MIN) &
    (df["latitude"] <= REGION_LAT_MAX) &
    (df["longitude"] >= REGION_LON_MIN) &
    (df["longitude"] <= REGION_LON_MAX)
].copy()

print("Exact in-region windows per iceberg (padded by "
      f"{PADDING_DAYS} days on each side):\n")

windows = []

for name, group in in_region.groupby("iceberg"):
    start = group["date"].min() - pd.Timedelta(days=PADDING_DAYS)
    end = group["date"].max() + pd.Timedelta(days=PADDING_DAYS)
    n_days = len(group)

    print(f"{name}: real in-region {group['date'].min().date()} to "
          f"{group['date'].max().date()} ({n_days} real days)")
    print(f"  -> padded download window: {start.date()} to {end.date()}")

    windows.append({
        "iceberg": name,
        "download_start": start.date(),
        "download_end": end.date(),
        "real_in_region_days": n_days
    })

windows_df = pd.DataFrame(windows)
windows_df.to_csv("data/iceberg_download_windows.csv", index=False)

print("\nSaved: data/iceberg_download_windows.csv")
print(windows_df)
