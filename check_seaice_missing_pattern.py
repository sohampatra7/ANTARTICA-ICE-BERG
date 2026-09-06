import xarray as xr
import numpy as np

sea_ice = xr.open_dataset("data/sea_ice_2024_weddell.nc")
var = list(sea_ice.data_vars)[0]
data = sea_ice[var]

print("Shape (time, y, x):", data.shape)

# For each grid cell, is it NaN on EVERY day, NEVER, or SOMETIMES?
is_nan = np.isnan(data.values)  # (time, y, x)

always_nan = is_nan.all(axis=0)      # cell missing on all 366 days
never_nan = (~is_nan).all(axis=0)    # cell present on all 366 days
sometimes_nan = ~always_nan & ~never_nan  # inconsistent

n_cells = always_nan.size

print(f"\nTotal grid cells: {n_cells}")
print(f"Always NaN (likely land mask): {always_nan.sum()} "
      f"({100*always_nan.sum()/n_cells:.1f}%)")
print(f"Never NaN (fully valid ocean cell): {never_nan.sum()} "
      f"({100*never_nan.sum()/n_cells:.1f}%)")
print(f"Sometimes NaN (inconsistent -> data gaps): {sometimes_nan.sum()} "
      f"({100*sometimes_nan.sum()/n_cells:.1f}%)")

if sometimes_nan.sum() > 0:
    # For inconsistent cells, how often are they missing?
    frac_missing = is_nan[:, sometimes_nan].mean(axis=0)
    print(f"\nFor inconsistent cells, missing on average "
          f"{frac_missing.mean()*100:.1f}% of days "
          f"(range {frac_missing.min()*100:.1f}% to {frac_missing.max()*100:.1f}%)")