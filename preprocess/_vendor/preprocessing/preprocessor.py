import pandas as pd
import re
import numpy as np
import geopandas as gpd
from skbio.stats.composition import clr, ilr

def get_df_element(df: pd.DataFrame, elements: list = None, include_coords: bool = True) -> list:
    """
    Extract geochemical concentration-related columns from a DataFrame.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        elements (list, optional): List of element symbols to filter (e.g. ["Cu", "Pb", "Zn"]).
        include_coords (bool, optional): Whether to include coordinate columns (X, Y, Longitude, Latitude, etc.). Default True.
    
    Returns:
        list: Selected column names.
    """
    cols = list(df.columns)

    # --- Identify coordinate columns (optional) ---
    coord_keywords = ["x", "y", "longitude", "latitude", "easting", "northing"]
    coord_cols = [c for c in cols if c.lower() in coord_keywords] if include_coords else []

    # --- Identify concentration columns ---
    if elements:
        # Match given elements with typical suffixes (ppm, pct, oxide forms, etc.)
        pattern = re.compile(
            rf"^({'|'.join(elements)})((_?\w*)|(_O\d*)|(_O\d*_pct)|(_O\d*_ppm)|(_O\d*_ppb)|(_O\d*_wt))*$",
            re.IGNORECASE
        )
        conc_cols = [c for c in cols if pattern.match(c)]
    else:
        # Auto-detect concentration columns
        pattern = re.compile(
            r'(_ppm|_ppb|_pct|_wt|_O\d*$|_oxide|_O\d*_pct|_O\d*_wt|_O\d*_ppth)$',
            re.IGNORECASE
        )
        conc_cols = [c for c in cols if pattern.search(c)]

    # --- Merge and remove duplicates (preserve order) ---
    selected_cols = list(dict.fromkeys(coord_cols + conc_cols))
    return selected_cols


def preprocess_geochem_data(
    data,
    x_col="X",
    y_col="Y",
    elem_suffixes=("_ppm", "_pct"),
    transform="clr",
    crs="EPSG:4326",
    save_path=None,
    verbose=True
):
    """
    Safe geochemical preprocessing:
      • Forces numeric dtype for element columns
      • Replaces <=0, NaN, or inf with half detection limit (min positive / 2)
      • Ensures finite positive matrix before CLR/ILR
    """
    # --- Load CSV or DataFrame
    if isinstance(data, str):
        df = pd.read_csv(data)
        if verbose: print(f"Loaded: {data} ({len(df)} rows)")
    else:
        df = data.copy()

    # --- GeoDataFrame with coordinates
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[x_col], df[y_col]),
        crs=crs
    )
    gdf["Longitude"] = gdf.geometry.x
    gdf["Latitude"] = gdf.geometry.y

    # --- Identify element columns
    elem_cols = [c for c in df.columns if c.endswith(elem_suffixes)]
    if verbose: print(f"Detected {len(elem_cols)} element columns.")

    # --- Convert everything to numeric (forcefully)
    for c in elem_cols:
        gdf[c] = (
            pd.to_numeric(gdf[c], errors="coerce")
            .astype("float64")
        )

    # --- Replace invalid/negative/zero with half detection limit
    for c in elem_cols:
        col = gdf[c].replace([np.inf, -np.inf], np.nan)
        min_pos = col[col > 0].min()
        half_limit = min_pos / 2 if pd.notna(min_pos) and min_pos > 0 else 1e-6

        # Replace all bad values with half limit
        gdf[c] = col.apply(lambda x: x if (isinstance(x, (int, float)) and x > 0 and np.isfinite(x)) else half_limit)
        if verbose:
            print(f"{c}: using half detection limit = {half_limit:.4g}")

    # --- Final numeric cleanup: ensure float64 & finite
    gdf[elem_cols] = gdf[elem_cols].astype("float64")
    gdf[elem_cols] = gdf[elem_cols].replace([np.inf, -np.inf], np.nan).fillna(1e-6)

    # --- CLR / ILR transform
    X = gdf[elem_cols].to_numpy(dtype=float)
    if not np.all(np.isfinite(X)):
        if verbose: print("Warning: replacing remaining non-finite with 1e-6")
        X[~np.isfinite(X)] = 1e-6

    if transform == "clr":
        values = clr(X)
        suffix = "_clr"
    elif transform == "ilr":
        values = ilr(X)
        suffix = "_ilr"
    else:
        values = None
        suffix = ""

    # --- Merge results
    if values is not None:
        trans_df = pd.DataFrame(values, columns=[f"{c}{suffix}" for c in elem_cols])
        gdf = pd.concat([gdf.reset_index(drop=True), trans_df.reset_index(drop=True)], axis=1)
        if verbose:
            print(f"{transform.upper()} transform applied on {len(gdf)} rows successfully.")

    # --- Save (optional)
    if save_path:
        gdf.to_csv(save_path, index=False)
        if verbose:
            print(f"Saved processed file to {save_path}")

    return gdf

