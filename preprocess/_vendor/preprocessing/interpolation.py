import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from pyproj import CRS, Transformer
from scipy.interpolate import griddata, Rbf
from pykrige.ok import OrdinaryKriging
import os
import json,pickle
import random
from PIL import Image
from preprocessing.preprocessor import get_df_element
from skbio.stats.composition import clr, ilr
import re
from tqdm import tqdm
# from PIL import Image
# Image.MAX_IMAGE_PIXELS = None
def get_safe_crs(lat, lon, input_epsg=7844):
    """Safely detect whether coords are degrees or projected, and return appropriate CRS"""
    # If coordinates look like degrees (abs(lat) < 90, abs(lon) < 180)
    if abs(lat) < 90 and abs(lon) < 180:
        zone = int((lon + 180) / 6) + 1
        epsg = 7800 + zone  # GDA2020 MGA (EPSG:7849–7856)
        print(f"🧭 Detected geographic coords — using GDA2020 MGA zone {zone} (EPSG:{epsg})")
        return CRS.from_epsg(epsg)
    else:
        print(f"📍 Detected projected coords — using GDA2020 (EPSG:{input_epsg})")
        return CRS.from_epsg(input_epsg)

def interpolate_and_save_image_gda2020(
    df, column, output_image='interpolation_map.png',
    resolution_m=500, patch_size=16, method='idw', input_epsg=7844
):
    # --- Step 1: Compute CRS safely
    lon_center, lat_center = df['X'].mean(), df['Y'].mean()
    input_crs = CRS.from_epsg(input_epsg)
    utm_crs = get_safe_crs(lat_center, lon_center, input_epsg=input_epsg)
    transformer = Transformer.from_crs(input_crs, utm_crs, always_xy=True)

    # --- Step 2: Clean and transform
    df = df.dropna(subset=[column])
    x_proj, y_proj = transformer.transform(df['X'].values, df['Y'].values)
    z = df[column].values

    # --- Step 3: Define grid extent
    x_min_raw, x_max_raw = np.min(x_proj), np.max(x_proj)
    y_min_raw, y_max_raw = np.min(y_proj), np.max(y_proj)
    width_raw = x_max_raw - x_min_raw
    height_raw = y_max_raw - y_min_raw

    num_cols = int(np.ceil(width_raw / resolution_m))
    num_rows = int(np.ceil(height_raw / resolution_m))

    # --- Step 4: Round up to patch size
    num_cols = ((num_cols // patch_size) + 1) * patch_size
    num_rows = ((num_rows // patch_size) + 1) * patch_size

    width = num_cols * resolution_m
    height = num_rows * resolution_m

    # --- Step 5: Center grid
    center_x = (x_min_raw + x_max_raw) / 2
    center_y = (y_min_raw + y_max_raw) / 2
    x_min = center_x - width / 2
    x_max = center_x + width / 2
    y_min = center_y - height / 2
    y_max = center_y + height / 2

    # --- Step 6: Create grid
    x_grid = np.linspace(x_min, x_max, num=num_cols)
    y_grid = np.linspace(y_min, y_max, num=num_rows)
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)

    # --- Step 7: Interpolation
    if method == 'idw':
        interp_points = np.vstack((grid_x.flatten(), grid_y.flatten())).T
        tree = cKDTree(np.vstack((x_proj, y_proj)).T)
        dists, idxs = tree.query(interp_points, k=6)
        weights = 1 / (dists ** 2 + 1e-12)
        interpolated = np.sum(weights * z[idxs], axis=1) / np.sum(weights, axis=1)
        interpolated = interpolated.reshape(grid_x.shape)
    elif method == 'kriging':
        OK = OrdinaryKriging(x_proj, y_proj, z, variogram_model='linear',
                             verbose=False, enable_plotting=False)
        interpolated, _ = OK.execute('grid', x_grid, y_grid)
    elif method in ['linear', 'cubic', 'nearest']:
        interpolated = griddata(np.column_stack((x_proj, y_proj)), z, (grid_x, grid_y), method=method)
    elif method == 'rbf':
        rbf = Rbf(x_proj, y_proj, z)
        interpolated = rbf(grid_x, grid_y)
    else:
        raise ValueError(f"Unsupported interpolation method: {method}")

    # --- Step 8: Save image
    fig_width = num_cols / 100
    fig_height = num_rows / 100

    print(f"✅ Image grid: {num_cols} × {num_rows} pixels (resolution = {resolution_m} m)")
    print(f"📏 Area: {width/1000:.2f} × {height/1000:.2f} km")
    print(f"🧩 Patch size: {patch_size}, divisible ✓")

    plt.figure(figsize=(fig_width, fig_height), dpi=100)
    plt.contourf(grid_x, grid_y, interpolated, levels=20, cmap='viridis')
    plt.axis('off')
    plt.gca().set_aspect('equal')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_image, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.show()
    plt.close()

    print(f"📷 Image saved to {output_image}")
    
def get_utm_crs_from_latlon(lat, lon):
    """Determine GDA2020 MGA zone from lat/lon (returns pyproj CRS)"""
    zone = int((lon + 180) / 6) + 1
    epsg = 7800 + zone  # GDA2020 MGA (EPSG:7849–7856)
    return CRS.from_epsg(epsg)

def idw_interpolation(x, y, z, grid_x, grid_y, power=2, k=6):
    interp_points = np.vstack((grid_x.flatten(), grid_y.flatten())).T
    tree = cKDTree(np.vstack((x, y)).T)
    dists, idxs = tree.query(interp_points, k=k)
    weights = 1 / (dists ** power + 1e-12)
    weighted_vals = np.sum(weights * z[idxs], axis=1) / np.sum(weights, axis=1)
    return weighted_vals.reshape(grid_x.shape)

def interpolate_and_annotate_with_patches(
    df, gdf, column,
    patch_size=16,
    resolution_m=500,
    method='idw',
    output_image='interpolation_map.png',
    output_json='interpolation_patch_labels.pickle',
    input_epsg=7844,
    dpi=1,
):
    # 1. Project CRS
    lon_center, lat_center = df['X'].mean(), df['Y'].mean()
    input_crs = CRS.from_epsg(input_epsg)
    utm_crs = get_utm_crs_from_latlon(lat_center, lon_center)
    transformer = Transformer.from_crs(input_crs, utm_crs, always_xy=True)

    df = df.dropna(subset=[column])
    x_proj, y_proj = transformer.transform(df['X'].values, df['Y'].values)
    z = df[column].values

    # Remove duplicate projected coordinates (keeps first occurrence)
    # Duplicates cause a singular covariance matrix in kriging and degrade IDW
    coords = np.vstack((x_proj, y_proj)).T
    _, unique_idx = np.unique(coords, axis=0, return_index=True)
    if len(unique_idx) < len(x_proj):
        print(f"  ⚠️  Removed {len(x_proj) - len(unique_idx)} duplicate coordinate rows")
        x_proj = x_proj[unique_idx]
        y_proj = y_proj[unique_idx]
        z = z[unique_idx]

    if not {"X", "Y"}.issubset(gdf.columns):
        raise ValueError("CSV must contain 'X' and 'Y' columns.")

    # Create GeoDataFrame (assume input is geographic GDA2020: EPSG 7844)
    gdf = gpd.GeoDataFrame(
        gdf,
        geometry=gpd.points_from_xy(gdf["X"], gdf["Y"]),
        crs="EPSG:7844"  # GDA2020 geographic (degrees)
    )
    gdf = gdf.to_crs(utm_crs)
    gold_x = gdf.geometry.x.values
    print(gold_x)
    gold_y = gdf.geometry.y.values

    # 2. Bounding box from data
    x_min_raw, x_max_raw = np.min(x_proj), np.max(x_proj)
    y_min_raw, y_max_raw = np.min(y_proj), np.max(y_proj)
    center_x = (x_min_raw + x_max_raw) / 2
    center_y = (y_min_raw + y_max_raw) / 2

    width_raw = x_max_raw - x_min_raw
    height_raw = y_max_raw - y_min_raw
    num_cols = int(np.ceil(width_raw / resolution_m))
    num_rows = int(np.ceil(height_raw / resolution_m))

    if num_cols % patch_size != 0:
        num_cols = ((num_cols // patch_size) + 1) * patch_size
    if num_rows % patch_size != 0:
        num_rows = ((num_rows // patch_size) + 1) * patch_size

    width = num_cols * resolution_m
    height = num_rows * resolution_m

    x_min = center_x - width / 2
    x_max = center_x + width / 2
    y_min = center_y - height / 2
    y_max = center_y + height / 2

    # 3. Generate interpolation grid
    x_grid = np.linspace(x_min, x_max, num=num_cols)
    y_grid = np.linspace(y_min, y_max, num=num_rows)
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)

    # 4. Interpolate
    if method == 'idw':
        interpolated = idw_interpolation(x_proj, y_proj, z, grid_x, grid_y)
    elif method == 'kriging':
        OK = OrdinaryKriging(x_proj, y_proj, z, variogram_model='linear', verbose=False, enable_plotting=False)
        interpolated, _ = OK.execute('grid', x_grid, y_grid)
    elif method in ['linear', 'cubic', 'nearest']:
        interpolated = griddata(np.column_stack((x_proj, y_proj)), z, (grid_x, grid_y), method=method)
    elif method == 'rbf':
        rbf = Rbf(x_proj, y_proj, z)
        interpolated = rbf(grid_x, grid_y)
    else:
        raise ValueError(f"Unsupported interpolation method: {method}")

    # 5. Save image
    fig_width = num_cols / dpi
    fig_height = num_rows / dpi
    plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    plt.contourf(grid_x, grid_y, interpolated, levels=20, cmap='viridis')
    plt.axis('off')
    plt.gca().set_aspect('equal')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_image, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close()
    print(f"📷 Saved image: {output_image}")

    # 6. Patch-level annotation
    patch_labels = []
    pixels_per_patch = patch_size  # Fixed: each patch = 16×16 pixels
    patch_w = pixels_per_patch * resolution_m
    patch_h = pixels_per_patch * resolution_m
    
    num_patch_cols = num_cols // pixels_per_patch
    num_patch_rows = num_rows // pixels_per_patch
    print("num_patch_rows:",num_patch_rows)
    print("num_patch_cols:",num_patch_cols)
    for row in range(num_patch_rows):
        for col in range(num_patch_cols):

            # --- Patch 边界（UTM）---
            x0 = x_min + col * patch_w
            x1 = x0 + patch_w
            y0 = y_min + row * patch_h
            y1 = y0 + patch_h

            patch_id = row * num_patch_cols + col

            # --- 判断这个 patch 是否包含 gold 点 ---
            inside = ((gold_x >= x0) & (gold_x < x1) &
                    (gold_y >= y0) & (gold_y < y1))
            has_gold = np.any(inside)
            label = 1 if has_gold else -1

            # --- 保存 patch-level 信息 ---
            patch_labels.append({
                "patch_id": patch_id,
                "row": row,
                "col": col,
                "bbox_utm": [x0, y0, x1, y1],
                "label": label
            })


    # 7. Pixel-level labels (GLOBAL, 20 groups, balanced)
    # -1: background
    # +1: gold
    #  0: random negative (same count as gold, >=10 pixels away)

    num_groups = 10
    min_dist = 10  # pixels
    max_trials = 10000

    # collect gold pixel coordinates (deduplicate: multiple samples may fall in same pixel)
    gold_pixels_raw = []
    for gx, gy in zip(gold_x, gold_y):
        px = int((gx - x_min) // resolution_m)
        py = int((y_max - gy) // resolution_m)   # north-first: row 0 = y_max (top of image)
        if 0 <= px < num_cols and 0 <= py < num_rows:
            gold_pixels_raw.append((py, px))

    # Remove duplicates so num_pos matches actual unique positive pixels
    gold_pixels = list(dict.fromkeys(gold_pixels_raw))
    num_pos = len(gold_pixels)
    if num_pos == 0:
        raise ValueError("No positive (gold) pixels found.")

    pixel_labels_all = []

    for group_id in range(num_groups):
        pixel_labels = np.full((num_rows, num_cols), -1, dtype=int)

        # mark gold pixels
        for py, px in gold_pixels:
            pixel_labels[py, px] = 1

        neg_pixels = []
        trials = 0

        while len(neg_pixels) < num_pos and trials < max_trials:
            trials += 1
            ry = np.random.randint(0, num_rows)
            rx = np.random.randint(0, num_cols)

            if pixel_labels[ry, rx] != -1:
                continue

            # distance constraint
            if min(
                np.hypot(ry - gy, rx - gx)
                for gy, gx in gold_pixels
            ) < min_dist:
                continue

            pixel_labels[ry, rx] = 0
            neg_pixels.append((ry, rx))

        if len(neg_pixels) < num_pos:
            raise RuntimeError(
                f"Group {group_id}: could not sample enough negative pixels."
            )

        pixel_labels_all.append(pixel_labels.tolist())

    print(f"✅ Generated {len(pixel_labels_all)} pixel-label groups "
        f"(each with {num_pos} positives and {num_pos} negatives)")

    # 8. Save JSON
    output = {
        "utm_crs": utm_crs.to_string(),
        "bounding_box_utm": [x_min, y_min, x_max, y_max],
        "interpolation_image": os.path.abspath(output_image),
        "patch_grid": {
            "grid_size": [patch_size, patch_size],
            "patches": patch_labels
        },
        "pixel_grid": {
            "resolution_m": resolution_m,
            "size": [num_cols, num_rows],
            "labels": pixel_labels_all
        }
    }

    with open(output_json, 'wb') as f:
        pickle.dump(output, f)
    print(f"✅ Saved annotation JSON: {output_json}")

    return output_image, output_json, False


def split_image_into_patches_and_prepare_dataset(
    image_path,
    patch_label_json,        # output from interpolate_and_annotate_with_patches()
    output_dir='dataset_patches',
    test_ratio=1.0,          # 1:1 positive/negative in test set
    patch_size=16,           # PATCH SIZE IN PIXELS (e.g., 16x16)
    save_images=True,
    seed=42
):
    # Create output folders
    os.makedirs(output_dir, exist_ok=True)
    patch_dir = os.path.join(output_dir, 'patches')
    os.makedirs(patch_dir, exist_ok=True)

    # Load image
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    img_h, img_w = img_np.shape[:2]
    print(img.size)

    assert img_h % patch_size == 0 and img_w % patch_size == 0, \
        f"Image size must be divisible by patch size ({patch_size})"

    num_rows = img_h // patch_size
    num_cols = img_w // patch_size
    total_patches = num_rows * num_cols

    # Load patch labels
    with open(patch_label_json,'rb') as f:
        data = pickle.load(f)
        patches = data['patch_grid']['patches']

    pixel_labels = list(data['pixel_grid']['labels'])

    assert len(patches) == total_patches, \
        f"Label JSON contains {len(patches)} patches, but expected {total_patches}"

    # Split patches into pos/neg
    pos_patches = []
    neg_patches = []

    for patch in patches:

        pid = patch['patch_id']
        row = patch['row']   # south-first (row=0 = southernmost patch in annotation)
        col = patch['col']
        label = patch['label']

        # The saved image has NORTH at row 0 (plt.contourf y-axis convention).
        # The annotation uses south-first indexing (row=0 = SOUTH).
        # Flip to north-first so the image region matches the geographic patch.
        img_row = num_rows - 1 - row

        y0 = img_row * patch_size
        y1 = y0 + patch_size
        x0 = col * patch_size
        x1 = x0 + patch_size

        patch_img = img_np[y0:y1, x0:x1]
        patch_name = f"patch_{img_row:02d}_{col:02d}.png"
        patch_path = os.path.join(patch_dir, patch_name)

        if save_images:
            Image.fromarray(patch_img).save(patch_path)

        patch_entry = {
            "patch_id": pid,
            "row": img_row,   # store north-first image row (matches pixel_labels convention)
            "col": col,
            "image": patch_name,
            "label": label
        }

        if label == 1:
            pos_patches.append(patch_entry)
        else:
            neg_patches.append(patch_entry)

    # Random test sampling
    train_set = pos_patches + neg_patches
    
    # Output dataset split
    split_info = {
        "patch_size_pixels": patch_size,
        "image_size": [img_h, img_w],
        "num_patches": [num_rows, num_cols],
        "total_patches": total_patches,
        "train": train_set,
        "pixel_labels": pixel_labels
    }

    with open(os.path.join(output_dir, 'dataset_split.pkl'), 'wb') as f:
        pickle.dump(split_info, f)

    #print(f"✅ Saved {len(train_set)} training and {len(test_set)} testing patches to {output_dir}")
    return split_info


def overlay_gold_sites_on_image(image_path, gdf, annotation_json, output_path="interpolation_with_gold_sites.png"):
    """
    Overlay gold site points from gdf on the interpolated image using bounding box info from annotation JSON.
    """
    # Load annotation metadata
    with open(annotation_json, 'r') as f:
        meta = json.load(f)
    x_min, y_min, x_max, y_max = meta["bounding_box_utm"]

    # Ensure gdf is in the same UTM CRS
    utm_crs = meta["utm_crs"]
    gdf = gdf.to_crs(utm_crs)

    # Load image
    image = Image.open(image_path)
    img_w, img_h = image.size

    # Plot image with gold sites
    fig, ax = plt.subplots(figsize=(img_w / 100, img_h / 100), dpi=100)
    ax.imshow(image, extent=[x_min, x_max, y_min, y_max], origin='upper')
    gdf.plot(ax=ax, marker='*', color='red', markersize=20, label='Gold Site')

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.axis('off')
    ax.legend(loc='upper right')

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()
    print(f"✅ Saved visualization with gold sites: {output_path}")
    


def run_interpolation_pipeline(
    df_path: str,
    gdf_path: str,
    element_list: list,
    methods: list = ["idw"],
    patch_size: int = 16,
    resolution_m: int = 500,
    base_output_dir: str = "outputs/interpolated",
    transform: str = "clr"
):
    """
    Run interpolation and patch dataset generation for selected elements.

    Args:
        df_path: Path to raw geochemical CSV file.
        gdf_path: Path to site coordinates CSV or GeoPackage/shapefile.
        element_list: List of element names (e.g., ["Cu", "Pb", "Zn"]). Empty = all detected.
        methods: List of interpolation methods, e.g. ["idw", "kriging"].
        patch_size: Patch size in pixels for patch-based dataset generation.
        resolution_m: Interpolation grid resolution (meters).
        base_output_dir: Root directory for storing all interpolation results.
        transform: Compositional transform applied before interpolation: "clr", "ilr", or None.
    """

    # --- Load data ---
    print(f"📂 Loading geochemical data from: {df_path}")
    df = pd.read_csv(df_path)
    df = df.replace(-9999, np.nan)  # Replace sentinel missing values with NaN
    df.attrs["source_file"] = df_path

    print(f"📂 Loading site data from: {gdf_path}")
    if gdf_path.endswith('.csv'):
        gdf = pd.read_csv(gdf_path)
    else:
        gdf = gpd.read_file(gdf_path)

    # --- Detect available element columns ---
    available_cols = get_df_element(df, elements=element_list, include_coords=True)
    coord_keywords = ["x", "y", "longitude", "latitude", "easting", "northing", "X", "Y"]
    coord_cols = [c for c in available_cols if c.lower() in coord_keywords]
    chem_cols = [c for c in available_cols if c not in coord_cols]

    if not chem_cols:
        print("⚠️ No matching element concentration columns found.")
        return

    print(f"✅ Found {len(chem_cols)} chemical columns: {chem_cols}")

    # --- Apply CLR / ILR transform ---
    transform_label = transform if transform else "raw"
    if transform in ("clr", "ilr"):
        print(f"🔄 Applying {transform.upper()} transform to {len(chem_cols)} columns ...")

        # Build a complete positive matrix: impute NaN / ≤0 with half detection limit per column
        X = df[chem_cols].copy().astype(float)
        for col in chem_cols:
            series = X[col]
            min_pos = series[series > 0].min()
            half_dl = min_pos / 2 if pd.notna(min_pos) and min_pos > 0 else 1e-6
            X[col] = series.apply(lambda v: v if (pd.notna(v) and v > 0) else half_dl)

        X_matrix = X.to_numpy(dtype=float)

        if transform == "clr":
            transformed = clr(X_matrix)
            suffix = "_clr"
            trans_col_names = [f"{c}{suffix}" for c in chem_cols]
        else:  # ilr
            transformed = ilr(X_matrix)
            suffix = "_ilr"
            trans_col_names = [f"ilr_{i}" for i in range(transformed.shape[1])]

        trans_df = pd.DataFrame(transformed, columns=trans_col_names, index=df.index)
        df = pd.concat([df, trans_df], axis=1)
        chem_cols = trans_col_names
        print(f"✅ {transform.upper()} transform done → {len(chem_cols)} columns: {chem_cols}")
    else:
        print("ℹ️  No compositional transform applied (raw values).")

    # --- Derive area name from filename ---
    filename = os.path.basename(df_path)
    area_name_match = re.search(r"(area\d+)", filename, re.IGNORECASE)
    area_name = area_name_match.group(1) if area_name_match else "area_unknown"

    # --- Main loop ---
    for method in methods:
        # --- Prepare output directory (area / transform / interp method) ---
        area_output_dir = os.path.join(
            base_output_dir,
            f"{area_name}_resolution_{resolution_m}_patch_size_{patch_size}_{transform_label}_{method}"
        )
        os.makedirs(area_output_dir, exist_ok=True)

        for element_col in chem_cols:
            # Extract element symbol: split on '_' or capital 'O', take first non-empty part
            parts = re.split(r'[_O]', element_col)
            element_name = next((p for p in parts if p), element_col)
            prefix = f"{element_name}_{transform_label}_{method}"

            # Define structured output paths
            # Root folder already encodes transform+method; subfolders use element_name only
            output_image = os.path.join(area_output_dir, f"{element_name}_interp.png")
            output_json  = os.path.join(area_output_dir, f"{element_name}_annotation.json")
            output_dir   = os.path.join(area_output_dir, f"patches_{element_name}")

            # Skip columns with too few valid (non-NaN) values
            valid_count = df[element_col].notna().sum()
            if valid_count < 6:
                print(f"⚠️  Skipping {element_col}: only {valid_count} valid values (need ≥ 6)")
                continue

            print(f"🔹 Processing {prefix} ...")

            try:
                # Step 1: Interpolation + Annotation
                interpolate_and_annotate_with_patches(
                    df=df,
                    gdf=gdf,
                    column=element_col,
                    patch_size=patch_size,
                    resolution_m=resolution_m,
                    method=method,
                    output_image=output_image,
                    output_json=output_json,
                )

                # Step 2: Patch generation (create output_dir only on success)
                os.makedirs(output_dir, exist_ok=True)
                split_image_into_patches_and_prepare_dataset(
                    image_path=output_image,
                    patch_label_json=output_json,
                    output_dir=output_dir,
                    patch_size=patch_size,
                    test_ratio=1.0,
                    save_images=True
                )

                print(f"✅ Completed: {prefix} → {output_dir}")

            except Exception as e:
                print(f"❌ Skipping {prefix}: {type(e).__name__}: {e}")