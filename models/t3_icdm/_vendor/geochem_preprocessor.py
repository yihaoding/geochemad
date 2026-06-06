import numpy as np
import os
import pandas as pd
import re
import math

import os
import numpy as np
import pandas as pd
from scipy.linalg import helmert
import torch
from tqdm import tqdm
from our_model.feature_extractor import extract_hidden_state
ADD_EPS = 1e-9

def helmert_submatrix(D: int) -> np.ndarray:
    H = np.zeros((D, D))
    for i in range(1, D):
        a = 1.0 / math.sqrt(i*(i+1))
        for j in range(i):
            H[i, j] = a
        H[i, i] = -i * a
    V = H[1:, :].T  # D x (D-1)
    return V

def clr(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    g = np.exp(np.mean(np.log(x)))
    return np.log(x / g)

def clr_inv(z: np.ndarray) -> np.ndarray:
    x = np.exp(z)
    return x / np.sum(x)

def ilr(x: np.ndarray, V: np.ndarray) -> np.ndarray:
    return clr(x) @ V

def ensure_composition(X: np.ndarray) -> np.ndarray:
    X = np.maximum(X, 0.0) + ADD_EPS
    X = X / X.sum(axis=1, keepdims=True)
    return X

def candidate_columns(df, elems):
    """
    Find valid element columns in the DataFrame.

    Priority:
      1. Columns ending with '_ppm'
      2. Direct element name (e.g. 'Cu')
    Returns:
      cols: list of valid column names
      missing: list of elements not found
    """
    cols, missing = [], []
    for e in elems:
        ppm = f"{e}_ppm"
        if ppm in df.columns:
            cols.append(ppm)
        elif e in df.columns:
            cols.append(e)
        else:
            missing.append(e)
    return cols, missing

# ==========================================================
# --- Coordinate & Element Column Utilities ---
# ==========================================================
def standardize_coordinates(df):
    """
    Detect coordinate columns and rename to x,y.
    Returns DataFrame with standardized coordinate names.
    """
    coord_pairs = [
        ("LONGITUDE", "LATITUDE"),
        ("longitude", "latitude"),
        ("lon", "lat"),
        ("EASTING", "NORTHING"),
        ("x", "y"),
        ("X", "Y")
    ]
    for cand_x, cand_y in coord_pairs:
        if cand_x in df.columns and cand_y in df.columns:
            df = df.rename(columns={cand_x: "x", cand_y: "y"})
            print(f"📍 Using coordinate columns: {cand_x}, {cand_y} → x, y")
            return df
    raise AssertionError(
        "❌ Could not find coordinate columns. Expect LONGITUDE/LATITUDE or EASTING/NORTHING or x/y."
    )


def candidate_columns(df: pd.DataFrame, elems: list = None) -> tuple[list, list]:
    """
    Identify geochemical element-related columns in the DataFrame.

    Strictly matches element columns like Cu_ppm, Pb_ppb, Zn_pct.
    Does NOT match partial prefixes (e.g., 'Y' will NOT match 'Y_ppm' or 'Yb_ppm').

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        elems (list, optional): List of element symbols (e.g., ["Cu", "Pb", "Zn"]).

    Returns:
        tuple:
            - matched_cols (list): Found matching column names.
            - missing (list): Elements not found in the DataFrame.
    """
    cols = list(df.columns)
    matched_cols, missing = [], []

    # --- Case 0: literal column-name passthrough ---
    # If every requested elem is already an exact column name in df, use them
    # directly. Covers datasets like the Canadian-style sed3/sed4 where columns
    # are bare element symbols ('AU', 'AG', 'AS_O', ...) with no _ppm/_pct unit
    # suffix the regex below would otherwise require.
    if elems and len(elems) > 0 and all(e in cols for e in elems):
        return list(elems), []

    # --- Case 1: specific elements provided ---
    if elems and len(elems) > 0:
        for e in elems:
            # Match exact element name followed by _ppm, _ppb, or _pct
            subpattern = re.compile(
                rf"^{re.escape(e)}_(ppm|ppb|pct)$", re.IGNORECASE
            )
            # Ensure full exact element match, not substring (e.g. 'Y' ≠ 'Yb')
            element_cols = [
                c for c in cols
                if subpattern.fullmatch(c)
                and c.lower().startswith(f"{e.lower()}_")
                and len(c.split('_')[0]) == len(e)
            ]
            if element_cols:
                matched_cols.extend(element_cols)
            else:
                missing.append(e)

    # --- Case 2: auto-detect all *_ppm, *_ppb, *_pct columns ---
    else:
        pattern = re.compile(r'^[A-Za-z]+_(ppm|ppb|pct)$', re.IGNORECASE)
        matched_cols = [c for c in cols if pattern.fullmatch(c)]

    # --- Preserve order & uniqueness ---
    matched_cols = list(dict.fromkeys(matched_cols))

    return matched_cols, missing

# ==========================================================
# --- Cleaning & Imputation ---
# ==========================================================
def clean_values(df_elem):
    """
    Replace -9999 and invalid values (<=0) with NaN.
    """
    df_elem = df_elem.replace(-9999, np.nan).astype(float)
    df_elem[df_elem <= 0] = np.nan
    return df_elem


def remove_abnormal_rows(df_elem, verbose=True):
    """Drop rows containing any NaN (i.e. below-detection/invalid values)."""
    before = len(df_elem)
    df_clean = df_elem.dropna()
    after = len(df_clean)
    if verbose:
        print(f"[Remove] Dropped {before - after} rows with abnormal values "
              f"({(before - after) / before * 100:.1f}%). Remaining: {after}")
    return df_clean


def half_detection_impute(df_elem, verbose=True):
    """
    Replace NaNs or invalid values by half of median detection limit per element.
    Rows with too few valid entries are flagged.
    """
    ROW_MIN_VALID = max(3, len(df_elem.columns) // 2)
    valid_counts = df_elem.notna().sum(axis=1)
    low_valid_mask = valid_counts < ROW_MIN_VALID
    low_valid_ratio = low_valid_mask.mean()

    if verbose:
        print(f"[Half Detection Check] Rows with < {ROW_MIN_VALID} valid elements: "
              f"{low_valid_mask.sum()} / {len(df_elem)} ({low_valid_ratio*100:.2f}%)")
        if low_valid_ratio > 0.1:
            print("⚠️ Warning: More than 10% of rows have fewer than half valid element values.")

    # --- Imputation per element ---
    df_imputed = df_elem.copy()
    for c in df_elem.columns:
        col = df_elem[c]
        detected = col[~col.isna() & (col > 0)]
        if len(detected) == 0:
            half_detection = 1e-6
            if verbose:
                print(f"⚠️ No detected values for {c}, filling with {half_detection}")
        else:
            detection_limit = detected.median(skipna=True)
            half_detection = detection_limit / 2.0
        df_imputed[c] = col.fillna(half_detection)
        df_imputed.loc[col <= 0, c] = half_detection
    return df_imputed


# ==========================================================
# --- Main Preprocessing Pipeline ---
# ==========================================================
def preprocess_geochem_data(path, target_elems, verbose=True):
    """
    Full preprocessing pipeline for geochemical data.

    Steps:
      1. Load CSV
      2. Detect and standardize coordinates
      3. Select target elements
      4. Clean invalid values
      5. Replace missing by half detection limit
      6. Apply ILR transform

    Returns:
      work_df (pd.DataFrame): Cleaned dataset (x, y + elements)
      X_ilr (np.ndarray): ILR-transformed matrix (N, D-1)
      coords (np.ndarray): Coordinates (N, 2)
      elem_cols (list): Selected element names
    """
    assert os.path.exists(path), f"File not found: {path}"
    df = pd.read_csv(path)
    if verbose:
        print(f"✅ Loaded {os.path.basename(path)}: {df.shape}")

    # --- Coordinates ---
    df = standardize_coordinates(df) # Unify the Column Name to X and Y 

    # --- Elements ---
    elem_cols, missing = candidate_columns(df, target_elems)
    if verbose:
        print(f"🧪 Found {len(elem_cols)} element columns; Missing: {missing}")

    # --- Clean and Impute --- 
    ## Abnormal Value Processing
    df_elem = clean_values(df[elem_cols])
    if verbose:
        print("🔍 Top-10 NaN rates:\n", df_elem.isna().mean().sort_values(ascending=False).head(10))

    df_elem_imputed = half_detection_impute(df_elem, verbose=verbose)

    # --- Compose final dataset ---
    work_df = pd.concat([df[["x", "y"]].reset_index(drop=True),
                         df_elem_imputed.reset_index(drop=True)], axis=1)
    if verbose:
        print(f"✅ Working DF: {work_df.shape}")

    # --- ILR transform ---
    X = work_df[elem_cols].to_numpy(float)
    X = ensure_composition(X)
    D = X.shape[1]
    V = helmert_submatrix(D)
    X_ilr = np.stack([ilr(row, V) for row in X], axis=0)
    print(X_ilr)
    coords = work_df[["x", "y"]].to_numpy(float)

    if verbose:
        print(f"📈 ILR shape: {X_ilr.shape} | Coords shape: {coords.shape}")

    return work_df, X_ilr, coords, elem_cols

# ======================================================
# 🔹 1️⃣ Hidden Feature Extraction
# ======================================================
def extract_hidden_features_from_model(
    model_ckpt,
    csv_path,
    work_df,
    coords,
    elem_cols,
    target_element,
    k_neighbors=128,
    hidden_dim=128,
    num_layers=2,
    num_elements=20,
    device=None,
    verbose=True,
):
    """
    Extract hidden representations (target token embeddings) for all samples.

    Args:
        model_ckpt (str): Path to trained GeoTransformer checkpoint (.pt)
        csv_path (str): Original CSV (used by extract_hidden_state)
        work_df (pd.DataFrame): Cleaned geochemical dataset
        coords (np.ndarray): (N, 2) coordinate array
        elem_cols (list[str]): Selected element features
        target_element (str): e.g., 'Au_ppm'
        k_neighbors (int): Context neighbors per sample
        hidden_dim (int): Transformer hidden dimension
        num_layers (int): Number of encoder layers
        num_elements (int): Number of element embeddings
        device (str): 'cuda' or 'cpu'
        verbose (bool): Print progress

    Returns:
        hidden_mat (np.ndarray): (N, hidden_dim) hidden representations
        preds (np.ndarray): (N,) predicted element values
    """
    DEVICE = device or ("cuda" if torch.cuda.is_available() else "cpu")
    hidden_vecs, preds = [], []

    if verbose:
        print(f"🔍 Loading model checkpoint: {model_ckpt}")
        print(f"💻 Device: {DEVICE}")

    for i in tqdm(range(len(work_df)), desc="Extracting hidden features"):
        coord = tuple(coords[i])
        pred_val, hidden_vec = extract_hidden_state(
            model_ckpt=model_ckpt,
            csv_path=csv_path,
            target_element=target_element,
            target_coord=coord,
            feature_cols=elem_cols,
            k_neighbors=k_neighbors,
            device=DEVICE,
        )
        preds.append(pred_val)
        hidden_vecs.append(hidden_vec)

    hidden_mat = np.stack(hidden_vecs, axis=0)
    preds = np.array(preds)
    if verbose:
        print(f"✅ Hidden matrix shape: {hidden_mat.shape}")
    return hidden_mat, preds


# ======================================================
# 🔹 2️⃣ Data Preprocessing + Optional Feature Extraction
# ======================================================
def preprocess_geochem_data_new(
    path,
    target_elems,
    model_ckpt=None,
    device=None,
    verbose=True,
    target_element="Au_ppm",
    transform="ilr",             # 'ilr' | 'clr' | 'none'
    n_elements=None,             # int or None → keep top-N elements by variance
    abnormal_method='half_detection',  # 'half_detection' | 'remove'
):
    """
    Full preprocessing pipeline for geochemical data.
    If model_ckpt is provided, calls extract_hidden_features_from_model() automatically.

    Args:
        transform (str): Compositional transform applied to element concentrations.
            'ilr'  – Isometric Log-Ratio (default, distance-preserving in simplex).
            'clr'  – Centred Log-Ratio (keeps D dimensions, interpretable).
            'none' – Log-transform only (no closure / compositional constraint).
        n_elements (int|None): Keep only the top-N elements by inter-sample
            variance after imputation.  None = keep all found columns.

    Returns:
        work_df (pd.DataFrame): Cleaned dataset (x, y + elements)
        X_trans (np.ndarray): Transformed feature matrix (N, D or N, D-1 for ILR)
        coords (np.ndarray): Coordinates (N, 2)
        elem_cols (list): Selected element column names
        hidden_mat (np.ndarray or None): Hidden states if model_ckpt given
    """
    VALID_TRANSFORMS = ('ilr', 'clr', 'none')
    assert transform in VALID_TRANSFORMS, \
        f"transform must be one of {VALID_TRANSFORMS}, got '{transform}'"
    assert os.path.exists(path), f"File not found: {path}"
    DEVICE = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load data ---
    df = pd.read_csv(path)
    if verbose:
        print(f"✅ Loaded {os.path.basename(path)}: {df.shape}")

    # --- Coordinate and feature cleaning ---
    df = standardize_coordinates(df)
    elem_cols, missing = candidate_columns(df, target_elems)
    if verbose:
        print(f"🧪 Found {len(elem_cols)} element columns; Missing: {missing}")

    df_elem = clean_values(df[elem_cols])
    if abnormal_method == 'half_detection':
        df_elem = half_detection_impute(df_elem, verbose=verbose)
    elif abnormal_method == 'remove':
        df_elem = remove_abnormal_rows(df_elem, verbose=verbose)
        df = df.loc[df_elem.index]
    else:
        raise ValueError(f"Unknown abnormal_method: '{abnormal_method}'. "
                         f"Choose 'half_detection' or 'remove'.")

    # --- Optional: trim to top-N elements by variance ---
    if n_elements is not None and n_elements < len(elem_cols):
        variances = df_elem.var(axis=0)
        top_cols  = variances.nlargest(n_elements).index.tolist()
        dropped   = [c for c in elem_cols if c not in top_cols]
        elem_cols = top_cols
        df_elem   = df_elem[elem_cols]
        if verbose:
            print(f"🔢 Kept top-{n_elements} elements by variance. Dropped: {dropped}")

    work_df = pd.concat(
        [df[["x", "y"]].reset_index(drop=True), df_elem.reset_index(drop=True)], axis=1
    )
    if verbose:
        print(f"✅ Working DF: {work_df.shape}")

    # --- Compositional transform ---
    X = work_df[elem_cols].to_numpy(float)
    if transform in ('ilr', 'clr'):
        X = ensure_composition(X)    # closure onto simplex (required for CLR/ILR)
    if transform == 'ilr':
        D = X.shape[1]
        V = helmert_submatrix(D)
        X_trans = np.stack([ilr(row, V) for row in X], axis=0)
    elif transform == 'clr':
        X_trans = np.stack([clr(row) for row in X], axis=0)
    else:  # 'none'
        X_trans = np.log(np.maximum(X, ADD_EPS))    # plain log, no closure

    coords = work_df[["x", "y"]].to_numpy(float)
    if verbose:
        print(f"📈 transform='{transform}' | feature shape: {X_trans.shape} | coords: {coords.shape}")

    # --- Hidden feature extraction (optional) ---
    hidden_mat = None
    if model_ckpt:
        hidden_mat, preds = extract_hidden_features_from_model(
            model_ckpt=model_ckpt,
            csv_path=path,
            work_df=work_df,
            coords=coords,
            elem_cols=elem_cols,
            target_element=target_element,
            k_neighbors=128,
            hidden_dim=128,
            num_layers=2,
            num_elements=20,
            device=DEVICE,
            verbose=verbose,
        )
        if verbose:
            print(f"✅ Hidden features extracted: {hidden_mat.shape}")

    return work_df, X_trans, coords, elem_cols, hidden_mat
