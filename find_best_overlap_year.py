import pandas as pd

# Must match the box used in iceberg_download.py, era5_download.py,
# and sea_ice_download.py
REGION_LAT_MIN = -70.0
REGION_LAT_MAX = -65.0
REGION_LON_MIN = -50.0
REGION_LON_MAX = -40.0

df = pd.read_csv("data/iceberg_tracks_5.csv", parse_dates=["date"])

in_region = df[
    (df["latitude"] >= REGION_LAT_MIN) &
    (df["latitude"] <= REGION_LAT_MAX) &
    (df["longitude"] >= REGION_LON_MIN) &
    (df["longitude"] <= REGION_LON_MAX)
].copy()

in_region["year"] = in_region["date"].dt.year

# Rows per iceberg per year
pivot = in_region.groupby(["year", "iceberg"]).size().unstack(fill_value=0)

# How many of the 5 icebergs were in-region at all that year
pivot["icebergs_present"] = (pivot > 0).sum(axis=1)

# Total in-region days across all icebergs that year
pivot["total_in_region_days"] = pivot.drop(columns="icebergs_present").sum(axis=1)

pivot = pivot.sort_values(
    ["icebergs_present", "total_in_region_days"],
    ascending=False
)

print("Per-year breakdown (rows = years, columns = icebergs):\n")
print(pivot)

best_year = pivot.index[0]
print(f"\nBest year by icebergs_present, then total in-region days: {best_year}")
print(f"  -> {pivot.loc[best_year, 'icebergs_present']} icebergs present, "
      f"{pivot.loc[best_year, 'total_in_region_days']} total in-region days")

print("\nTop 5 candidate years:")
print(pivot.head(5)[["icebergs_present", "total_in_region_days"]])
