"""
Data loading + preprocessing for ICDM-v2 single-area pipeline.

Configurable preprocessing (all per-area, no state-level mixing):

  --transform        none | clr | ilr | log
      none   StandardScaler on half-DL raw values (paper main-table default)
      log    log1p then StandardScaler (better for elements spanning 6+ orders)
      clr    Centred Log-Ratio after half-DL imputation
      ilr    Isometric Log-Ratio (D → D-1 features)

  --input_elements  comma-separated list (e.g. 'Au_ppm,As_ppm,Sb_ppm')
      empty / None → use ALL detected element columns
      'auto:pca-15' → top-15 PCA-loaded elements (computed on the area)

  --bdl_strategy    flag | mask | drop_col
      flag      keep BDL value as half-DL and emit a binary BDL flag (default)
      mask      treat BDL as missing token (zero) inside attention; flag emitted
      drop_col  drop any element column with >X% BDL fraction

  --bdl_threshold   float in [0,1] (only when bdl_strategy='drop_col')
      default 0.95 — drop elements where ≥95% rows are BDL

These give the per-area knobs you need to test paper's clr / ilr / pca configs
or simply run the cleanest baseline.
"""

import re
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


SENTINELS = (-9999, -1, -0.5, -0.1)

# Paper element regex: pure-letter symbol + suffix (no digits → drops oxides
# like Al2O3_pct, Fe2O3_pct, ...).  Mirrors regex used in
# geo_transformer/geochem_preprocessor.py so backbone sees the SAME element
# set as paper T1 does.
PAPER_ELEMENT_REGEX = re.compile(r"^[A-Za-z]+_(ppm|ppb|pct)$", re.IGNORECASE)

# Fallback for the Canadian (SIGEOM/Quebec) schema where element columns are
# bare element symbols in all-caps with no unit suffix (e.g. AU, AG, AS_O,
# IN_O — the *_O variant marks the aqua-regia digest). Oxides (SIO2, AL2O3,
# CR2O3, ...), totals (TOT, C_TOT, CO2_IN, PAF, C_ORG) and non-element codes
# (CODE_*, NUMR_*, DATE_*, PROJ_*) must NOT be included here.
# Source: standard periodic-table symbols 1-92 with stable analytical
# coverage in SIGEOM tables.
CANADIAN_ELEMENT_SYMBOLS = {
    "H", "LI", "BE", "B", "C", "N", "O", "F", "NA", "MG", "AL", "SI", "P",
    "S", "CL", "K", "CA", "SC", "TI", "V", "CR", "MN", "FE", "CO", "NI",
    "CU", "ZN", "GA", "GE", "AS", "SE", "BR", "RB", "SR", "Y", "ZR", "NB",
    "MO", "RU", "RH", "PD", "AG", "CD", "IN", "SN", "SB", "TE", "I", "CS",
    "BA", "LA", "CE", "PR", "ND", "SM", "EU", "GD", "TB", "DY", "HO", "ER",
    "TM", "YB", "LU", "HF", "TA", "W", "RE", "OS", "IR", "PT", "AU", "HG",
    "TL", "PB", "BI", "PO", "TH", "U",
}


def _is_canadian_element(col: str) -> bool:
    """True if `col` looks like a SIGEOM element name (AU, AG, AS_O, ...).

    Accepts: a bare element symbol, or symbol + "_O" (aqua-regia variant).
    Rejects:
      - coord column headers (X / Y / LAT / LON, both cases). In SIGEOM
        CSVs the latitude column is literally named "Y" — the yttrium
        column is the duplicate-suffix "Y.1" — so a naive symbol match
        leaks the coord into the feature set.
      - oxide formulas (SIO2, AL2O3, CR2O3, MOS2, TR2O3), totals
        (TOT, PAF, MOIST), non-element codes (NUMR_*, CODE_*), and any
        column with a "." (duplicate-rename, e.g. "Y.1" handled by the
        original column slot already if present elsewhere).
    """
    if col in {"X", "Y", "x", "y", "LAT", "LON", "LATITUDE", "LONGITUDE"}:
        return False
    if "." in col:
        return False
    base = col[:-2] if col.endswith("_O") else col
    return base in CANADIAN_ELEMENT_SYMBOLS


# ─── coordinate / column helpers ──────────────────────────────────────────────
def _xy_cols(df: pd.DataFrame):
    for a, b in [("X", "Y"), ("x", "y"),
                 ("LONGITUDE", "LATITUDE"), ("longitude", "latitude"),
                 ("lon", "lat")]:
        if a in df.columns and b in df.columns:
            return a, b
    raise KeyError("no x/y columns found")


def _element_columns(df: pd.DataFrame, paper_compatible: bool = True):
    """
    paper_compatible=True (default): use paper's regex — pure-letter symbol +
    suffix. Drops oxide / non-elemental columns (Al2O3_pct, Fe2O3_pct, ...).

    paper_compatible=False: loose suffix match (legacy behaviour, includes
    oxides).

    Fallback: if the paper regex matches 0 columns, try the Canadian
    SIGEOM/Quebec schema (bare element symbols, optionally with `_O` for
    the aqua-regia digest). This kicks in for area15-16_geochem.csv
    (sed3/sed4) and never fires on the standard Australian GSWA CSVs.
    """
    if paper_compatible:
        hits = [c for c in df.columns if PAPER_ELEMENT_REGEX.fullmatch(c)]
        if hits:
            return hits
        # Fallback: Canadian SIGEOM schema
        canadian = [c for c in df.columns if _is_canadian_element(c)]
        if canadian:
            print(f"[data.py] paper-regex matched 0 columns; falling back to "
                  f"Canadian SIGEOM schema → {len(canadian)} elements detected.")
            return canadian
        return []
    return [c for c in df.columns
            if any(c.endswith(s) for s in ("_ppm", "_pct", "_ppb"))]


# ─── BDL handling ─────────────────────────────────────────────────────────────
def _detect_bdl(raw: np.ndarray) -> np.ndarray:
    bdl = np.zeros_like(raw, dtype=np.float32)
    for s in SENTINELS:
        bdl[raw == s] = 1.0
    bdl[np.isnan(raw)] = 1.0
    bdl[raw <= 0]      = 1.0
    return bdl


def _half_dl(raw: np.ndarray, bdl: np.ndarray) -> np.ndarray:
    """
    Paper-compatible half-detection imputation: each BDL value is replaced by
    half of the column's MEDIAN detected value (1e-6 if no values detected).

    This matches geochem_preprocessor.half_detection_impute().
    """
    out = raw.copy().astype(np.float64)
    for j in range(out.shape[1]):
        col = out[:, j]
        pos = col[(col > 0) & np.isfinite(col)]
        half = float(np.median(pos) / 2.0) if len(pos) else 1e-6
        col[bdl[:, j] == 1.0] = half
        out[:, j] = col
    return out


# ─── compositional transforms ─────────────────────────────────────────────────
def _clr(x: np.ndarray) -> np.ndarray:
    # x assumed positive
    g = np.exp(np.log(x).mean(axis=1, keepdims=True))
    return np.log(x / g)


def _ilr(x: np.ndarray) -> np.ndarray:
    """Isometric log-ratio using a sequential basis (Aitchison)."""
    D = x.shape[1]
    out = np.zeros((x.shape[0], D - 1), dtype=np.float64)
    for k in range(D - 1):
        a = np.log(x[:, k]) - np.log(x[:, k + 1:]).mean(axis=1)
        out[:, k] = np.sqrt((D - k - 1) / (D - k)) * a
    return out


# ─── PCA / FS helpers ─────────────────────────────────────────────────────────
def _pca_topk(x: np.ndarray, elem_cols, k=15, target_element=None):
    """Pick top-k elements by sum of |loadings| over first 3 PCs."""
    pca = PCA(n_components=3).fit(StandardScaler().fit_transform(np.log1p(x)))
    score = np.abs(pca.components_).sum(axis=0)
    order = np.argsort(score)[::-1]
    if target_element and target_element in elem_cols:
        t_idx = elem_cols.index(target_element)
        if t_idx not in order[:k]:
            order = np.concatenate([[t_idx], order[order != t_idx]])
    keep = sorted(order[:k].tolist())
    return [elem_cols[i] for i in keep]


# ─── main loader ──────────────────────────────────────────────────────────────
def load_area(
    csv_path: str,
    target_element: str,
    transform: str = "none",                # none / log / clr / ilr
    input_elements=None,                    # None (all) | list[str] | 'auto:pca-15'
    bdl_strategy: str = "flag",             # flag / mask / drop_col / none
    bdl_threshold: float = 0.95,
    paper_compatible: bool = True,           # use paper element regex + median imputation
):
    df = pd.read_csv(csv_path)
    xcol, ycol = _xy_cols(df)
    df = df.dropna(subset=[xcol, ycol]).reset_index(drop=True)
    coords = df[[xcol, ycol]].to_numpy(dtype=np.float32)

    elem_cols_all = _element_columns(df, paper_compatible=paper_compatible)
    raw_all = df[elem_cols_all].to_numpy(dtype=np.float64)

    # ── element subset selection ────────────────────────────────────────────
    if input_elements is None or input_elements == "":
        sel = list(elem_cols_all)
    elif isinstance(input_elements, str):
        if input_elements.startswith("auto:pca-"):
            k = int(input_elements.split("-")[1])
            tmp_bdl = _detect_bdl(raw_all)
            tmp_x   = _half_dl(raw_all, tmp_bdl)
            sel = _pca_topk(tmp_x, elem_cols_all, k=k, target_element=target_element)
        else:
            sel = [s.strip() for s in input_elements.split(",") if s.strip()]
            if not sel:
                sel = list(elem_cols_all)
    else:
        sel = list(input_elements)

    if target_element not in sel:
        sel = sel + [target_element]                       # always keep target

    elem_cols = sel
    raw = df[elem_cols].to_numpy(dtype=np.float64)

    # ── BDL handling ────────────────────────────────────────────────────────
    # Always detect BDL for imputation — bdl_strategy controls only whether
    # the flag is EXPOSED downstream as input.
    bdl = _detect_bdl(raw)
    hide_bdl_flag = (bdl_strategy == "none")
    if bdl_strategy == "drop_col":
        keep = bdl.mean(axis=0) < bdl_threshold
        if not keep.all():
            dropped = [c for c, k in zip(elem_cols, keep) if not k]
            print(f"[load_area] dropping {len(dropped)} cols with ≥{bdl_threshold:.0%} BDL: {dropped[:8]}...")
            elem_cols = [c for c, k in zip(elem_cols, keep) if k]
            raw = raw[:, keep]; bdl = bdl[:, keep]
            if target_element not in elem_cols:
                raise ValueError(f"target {target_element} dropped by --bdl_strategy drop_col")
    x = _half_dl(raw, bdl)

    # ── compositional / numeric transform ──────────────────────────────────
    if transform == "none":
        feat = x.astype(np.float32)
    elif transform == "log":
        feat = np.log1p(x).astype(np.float32)
    elif transform == "clr":
        feat = _clr(x).astype(np.float32)
    elif transform == "ilr":
        feat = _ilr(x).astype(np.float32)
    else:
        raise ValueError(f"unknown transform: {transform}")

    scaler = StandardScaler().fit(feat)
    feat   = scaler.transform(feat).astype(np.float32)

    # ── target column ───────────────────────────────────────────────────────
    if transform == "ilr":
        # ILR drops one dim — target value gets handled separately via raw column
        # we keep target as standardised log1p of raw concentration for SSL task C
        tgt_raw = _half_dl(df[[target_element]].to_numpy(dtype=np.float64),
                           _detect_bdl(df[[target_element]].to_numpy(dtype=np.float64)))
        tgt_log = np.log1p(tgt_raw).astype(np.float32).ravel()
        tgt = ((tgt_log - tgt_log.mean()) / (tgt_log.std() + 1e-6)).astype(np.float32)
        t_idx = -1
    else:
        t_idx = elem_cols.index(target_element)
        tgt   = feat[:, t_idx].copy()

    # ── density (5 km neighbour count) ─────────────────────────────────────
    tree = KDTree(coords)
    DENSITY_RADIUS_DEG = 5.0 / 111.0
    density = np.array(
        [len(tree.query_radius(coords[i:i + 1], r=DENSITY_RADIUS_DEG)[0])
         for i in range(len(coords))], dtype=np.float32,
    )

    bdl_out = bdl.astype(np.float32)
    if transform == "ilr":
        # ILR rotates the simplex; BDL flags no longer align with the D-1 coordinates
        bdl_out = np.zeros((bdl.shape[0], feat.shape[1]), dtype=np.float32)
    elif hide_bdl_flag:
        bdl_out = np.zeros_like(bdl_out)                  # hide flag from model

    return {
        "coords":       coords,                          # [N, 2]
        "elem":         feat,                            # [N, C']
        "bdl":          bdl_out,                         # [N, C'] (zero if hidden)
        "target":       tgt,                             # [N]
        "density":      density,                         # [N]
        "element_cols": elem_cols,
        "target_idx":   t_idx,
        "transform":    transform,
        "scaler":       scaler,
    }


# ─── neighbour index + collate ────────────────────────────────────────────────
def build_neighbour_index(coords: np.ndarray, K: int):
    tree = KDTree(coords)
    _, idx = tree.query(coords, k=K + 1)
    return idx[:, 1:]                                    # drop self


def collate(ds: dict, nbr_idx: np.ndarray, query_ids: np.ndarray, device):
    coords  = ds["coords"]; elem = ds["elem"]; bdl = ds["bdl"]; tgt = ds["target"]
    dens    = ds["density"]
    nbr     = nbr_idx[query_ids]

    q_xy   = torch.tensor(coords[query_ids], device=device)
    q_elem = torch.tensor(elem [query_ids], device=device)
    q_bdl  = torch.tensor(bdl  [query_ids], device=device)
    q_tgt  = torch.tensor(tgt  [query_ids], device=device)

    n_xy   = torch.tensor(coords[nbr], device=device)
    n_elem = torch.tensor(elem [nbr], device=device)
    n_bdl  = torch.tensor(bdl  [nbr], device=device)
    n_dens = torch.tensor(dens [nbr], device=device)
    rel_xy = n_xy - q_xy.unsqueeze(1)
    return dict(
        q_xy=q_xy, q_elem=q_elem, q_bdl=q_bdl, q_target=q_tgt,
        n_xy=n_xy, n_elem=n_elem, n_bdl=n_bdl, n_density=n_dens,
        rel_xy=rel_xy,
    )


# ─── simple batch iterator over a fixed id list ──────────────────────────────
def iter_batches(ids: np.ndarray, batch_size: int, shuffle: bool, seed=None):
    if shuffle:
        rng = np.random.default_rng(seed)
        ids = ids[rng.permutation(len(ids))]
    for s in range(0, len(ids), batch_size):
        yield ids[s:s + batch_size]
