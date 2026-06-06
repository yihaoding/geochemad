"""
area_config.py  –  All 16 area configurations for GeoChemADv2.

Each entry defines:
  area_id        : short identifier used for output directories
  csv_path       : geochemical sample CSV
  deposits       : mineral site CSV (positive labels)
  target_element : column name for the target element
  input_elements : explicit element list, or None (auto-detect by non-exclude cols)
  data_type      : "sediment" | "drillhole" | "rockchip" | "soil"

Derived keys added at import time:
  base_elements  : the original input_elements spec (None → auto-detect all cols)
  llm_elements   : LLM-curated element subset for this area (None if not covered)
  hparams        : per-area training/eval hyper-parameters (see derive_hparams)

Feature-selection modes (--feature_selection in the VAE pipeline):
  none → base_elements (all cols)   llm → llm_elements   pca → top-k vs target
"""

import os as _os_paths

# Data root resolution (in priority order):
#   1. $GEOCHEM_DATA_ROOT  (point at any copy of data/gswa)
#   2. repo-bundled data/gswa  (this file lives in <repo>/shared/)
_DEFAULT_DATA_ROOT = _os_paths.path.normpath(
    _os_paths.path.join(_os_paths.path.dirname(__file__), "..", "data", "gswa")
)
DATA_ROOT = _os_paths.environ.get("GEOCHEM_DATA_ROOT", _DEFAULT_DATA_ROOT)
GEO_DIR  = _os_paths.path.join(DATA_ROOT, "geochemical")
SITE_DIR = _os_paths.path.join(DATA_ROOT, "site")

# ── Canada (area15-16): 92 geochemical columns – no _ppm suffix ──────────────
_CANADA_ELEMS = [
    "SIO2","TIO2","AL2O3","FE2O3_V","FE2O3_T","MNO","MGO","CAO","NA2O","K2O",
    "P2O5","C_TOT","CO2_IN","S","PAF","CR2O3","V2O5","AC","AG","AL","AS_O",
    "AU","B","BA","BE","BI","BR","C_ORG","CA","CD","CE","CL","CO","CR","CS",
    "CU","DY","ER","EU","F","FE","GA","GD","GE","HF","HG","HO","IN_O","IR",
    "K","LA","LI","LU","MG","MN","MO","NA","NB","ND","NI","P","PB","PD","PO",
    "PR","PT","RB","RE","SB","SC","SE","SI","SM","SN","SR","TA","TB","TE","TH",
    "TI","TL","TM","U","V","W","Y.1","YB","ZN","ZR","MOS2","TR2O3","TOT",
]

AREAS = [
    # ── Area 1-8 (original GeoChemADv2 subsets) ──────────────────────────────
    {
        "area_id":        "area1_sed1_au",
        "csv_path":       f"{GEO_DIR}/area1_sediment_au.csv",
        "deposits":       f"{SITE_DIR}/area1_sediment_au_site.csv",
        "target_element": "Au_ppm",
        "input_elements": None,
        "data_type":      "sediment",
    },
    {
        "area_id":        "area2_sed2_cu",
        "csv_path":       f"{GEO_DIR}/area2_sediment_cu.csv",
        "deposits":       f"{SITE_DIR}/area2_sediment_cu_site.csv",
        "target_element": "Cu_ppm",
        "input_elements": None,
        "data_type":      "sediment",
    },
    {
        "area_id":        "area3_rock1_w",
        "csv_path":       f"{GEO_DIR}/area3_rockchip_w.csv",
        "deposits":       f"{SITE_DIR}/area3_rockchip_w_site.csv",
        "target_element": "W_ppm",
        "input_elements": None,
        "data_type":      "rockchip",
    },
    {
        "area_id":        "area4_rock2_au",
        "csv_path":       f"{GEO_DIR}/area4_rockchip_au.csv",
        "deposits":       f"{SITE_DIR}/area4_rockchip_au_site.csv",
        "target_element": "Au_ppm",
        "input_elements": None,
        "data_type":      "rockchip",
    },
    {
        "area_id":        "area5_rock3_cu",
        "csv_path":       f"{GEO_DIR}/area5_rockchip_cu.csv",
        "deposits":       f"{SITE_DIR}/area5_rockchip_cu_site.csv",
        "target_element": "Cu_ppm",
        "input_elements": None,
        "data_type":      "rockchip",
    },
    {
        "area_id":        "area6_soil1_au",
        "csv_path":       f"{GEO_DIR}/area6_soil_au.csv",
        "deposits":       f"{SITE_DIR}/area6_soil_au_site.csv",
        "target_element": "Au_ppm",
        "input_elements": None,
        "data_type":      "soil",
    },
    {
        "area_id":        "area7_soil2_au",
        "csv_path":       f"{GEO_DIR}/area7_soil_au.csv",
        "deposits":       f"{SITE_DIR}/area7_soil_au_site.csv",
        "target_element": "Au_ppm",
        "input_elements": None,
        "data_type":      "soil",
    },
    {
        "area_id":        "area8_soil3_ni",
        "csv_path":       f"{GEO_DIR}/area8_soil_ni.csv",
        "deposits":       f"{SITE_DIR}/area8_soil_ni_site.csv",
        "target_element": "Ni_ppm",
        "input_elements": None,
        "data_type":      "soil",
    },
    # ── Area 9-16 (new GeoChemADv2 subsets) ──────────────────────────────────
    {
        "area_id":        "area9_dh1_au",
        "csv_path":       f"{GEO_DIR}/area9_drillhole_au.csv",
        "deposits":       f"{SITE_DIR}/area9_drillhole_au_site.csv",
        "target_element": "Au_ppm",
        "input_elements": None,
        "data_type":      "drillhole",
    },
    {
        "area_id":        "area10_dh2_ag",
        "csv_path":       f"{GEO_DIR}/area10_drillhole_au.csv",   # file named _au but contains Ag
        "deposits":       f"{SITE_DIR}/area10_drillhole_ag_site.csv",
        "target_element": "Ag_ppm",
        "input_elements": None,
        "data_type":      "drillhole",
    },
    {
        "area_id":        "area11_geo1_au",
        "csv_path":       f"{GEO_DIR}/area11-14_geochem.csv",
        "deposits":       f"{SITE_DIR}/area11_geochem_au_site.csv",
        "target_element": "Au_ppm",
        "input_elements": None,
        "data_type":      "geochem",
    },
    {
        "area_id":        "area12_geo2_cu",
        "csv_path":       f"{GEO_DIR}/area11-14_geochem.csv",
        "deposits":       f"{SITE_DIR}/area12_geochem_cu_site.csv",
        "target_element": "Cu_ppm",
        "input_elements": None,
        "data_type":      "geochem",
    },
    {
        "area_id":        "area13_geo3_ni",
        "csv_path":       f"{GEO_DIR}/area11-14_geochem.csv",
        "deposits":       f"{SITE_DIR}/area13_geochem_ni_site.csv",
        "target_element": "Ni_ppm",
        "input_elements": None,
        "data_type":      "geochem",
    },
    {
        "area_id":        "area14_geo4_w",
        "csv_path":       f"{GEO_DIR}/area11-14_geochem.csv",
        "deposits":       f"{SITE_DIR}/area14_geochem_w_site.csv",
        "target_element": "W_ppm",
        "input_elements": None,
        "data_type":      "geochem",
    },
    {
        "area_id":        "area15_sed3_au",
        "csv_path":       f"{GEO_DIR}/area15-16_geochem.csv",
        "deposits":       f"{SITE_DIR}/area15_geochem_au_site.csv",
        "target_element": "AU",
        "input_elements": _CANADA_ELEMS,
        "data_type":      "sediment",
    },
    {
        "area_id":        "area16_sed4_cu",
        "csv_path":       f"{GEO_DIR}/area15-16_geochem.csv",
        "deposits":       f"{SITE_DIR}/area16_geochem_cu_site.csv",
        "target_element": "CU",
        "input_elements": _CANADA_ELEMS,
        "data_type":      "sediment",
    },
]

# ── LLM feature selection ─────────────────────────────────────────────────────
# Load LLM-curated feature lists and inject into each area config.
# Rationale: remove oxide duplicates, remove lab artifacts, select target-
# relevant pathfinder suites per deposit type (Au/Cu/W/Li/Ni/Ag).

import json as _json
import os as _os

_LLM_SEL_PATH = _os.path.join(_os.path.dirname(__file__), "llm_feature_selection.json")
_LLM_FEATURES = {}
if _os.path.exists(_LLM_SEL_PATH):
    with open(_LLM_SEL_PATH) as _f:
        _LLM_FEATURES = _json.load(_f)

for _area in AREAS:
    # base_elements : original explicit element spec (None → auto-detect all cols)
    # llm_elements  : LLM-curated subset for this area (None if not in the JSON)
    # input_elements: kept = LLM subset when available, so the baseline /
    #                 interpolation pipelines that read it directly are unchanged.
    _area["base_elements"] = _area.get("input_elements")
    _area["llm_elements"]  = _LLM_FEATURES.get(_area["area_id"])
    if _area["llm_elements"]:
        _area["input_elements"] = _area["llm_elements"]

# ── Per-area hyper-parameters ─────────────────────────────────────────────────
# The 16 areas span a 300× range in spatial resolution (10 m – 3000 m), differ
# in channel count (35 – 123) and patch count (18 – 3626).  A single global
# config cannot fit all of them, so VAE-family training + spatial-evaluation
# hyper-parameters are *derived per area* from three measured statistics via
# fixed, documented formulas (see derive_hparams).
#
# _AREA_STATS: (resolution_m, n_channels, n_train_patches), measured from each
# area's vae_patches/metadata.json + dataset_split.pkl.
# Regenerate after re-building patches with:  python area_config.py --stats
_AREA_STATS = {
    "area1_sed1_au":  ( 500.0, 123,  400),
    "area2_sed2_cu":  ( 300.0, 123,  480),
    "area3_rock1_w":  (  80.0, 123, 3626),
    "area4_rock2_au": (  10.0, 123,  252),  # stat values copied from old Li run; rerun with --stats after patches are built
    "area5_rock3_cu": ( 160.0, 123, 2525),
    "area6_soil1_au": (  25.0, 123,   55),
    "area7_soil2_au": (  50.0, 123,  299),
    "area8_soil3_ni": ( 200.0, 123,  348),
    "area9_dh1_au":   ( 100.0, 123, 3264),
    "area10_dh2_ag":  (  80.0, 123, 3192),
    "area11_geo1_au": (3000.0,  45,   56),
    "area12_geo2_cu": (3000.0,  42,   56),
    "area13_geo3_ni": (3000.0,  35,   56),
    "area14_geo4_w":  (3000.0,  41,   56),
    "area15_sed3_au": (  70.0,  92, 2900),
    "area16_sed4_cu": (  70.0,  92, 2900),
}

import math as _math


def derive_hparams(resolution_m: float, n_channels: int, n_patches: int) -> dict:
    """
    Derive per-area training + evaluation hyper-parameters from three area
    statistics.  All formulas are deterministic and documented inline so the
    table can be regenerated whenever the patches change.

      batch_size : sized so each epoch has a healthy number of mini-batches
      epochs     : chosen so total gradient steps ≈ TARGET_STEPS, then clamped
      latent_dim : capacity scales with channel count; halved when data-starved
      lr / lr_d  : sqrt-scaled with batch_size from the 2e-4 @ bs128 reference
      min_dist   : negative-sampling buffer  =  8 px → km scales with resolution
      radius_km  : density neighbourhood     = 32 px → km scales with resolution
    """
    # — batch_size: keep ~8-30 mini-batches per epoch —
    if   n_patches <   32: batch_size = 8
    elif n_patches <  128: batch_size = 16
    elif n_patches <  512: batch_size = 32
    elif n_patches < 1500: batch_size = 64
    else:                  batch_size = 128

    # — epochs: total gradient steps ≈ TARGET_STEPS (clamped 100-300) —
    TARGET_STEPS = 6000
    spe    = _math.ceil(n_patches / batch_size)
    epochs = int(round((TARGET_STEPS / spe) / 10.0) * 10)
    epochs = max(100, min(300, epochs))

    # — latent_dim: capacity ~ channel count, halved for tiny datasets —
    if   n_channels >= 100: base = 80
    elif n_channels >=  64: base = 64
    else:                   base = 32
    latent_dim = max(32, base // 2) if n_patches < 100 else base

    # — lr: sqrt scaling rule, reference 2e-4 @ batch_size 128 —
    _LR  = {8: 5e-5, 16: 7e-5, 32: 1e-4, 64: 1.4e-4, 128: 2e-4}
    lr   = _LR[batch_size]
    lr_d = lr / 2.0

    # — eval radii: fixed pixel buffers → physical km scale with resolution —
    min_dist  = round(min(5.0,  max(0.1, resolution_m *  8 / 1000.0)), 2)
    radius_km = round(min(20.0, max(0.5, resolution_m * 32 / 1000.0)), 2)

    return {
        "batch_size": batch_size,
        "epochs":     epochs,
        "latent_dim": latent_dim,
        "lr":         lr,
        "lr_d":       lr_d,
        "min_dist":   min_dist,
        "radius_km":  radius_km,
    }


# Attach derived hyper-parameters to every area config (additive — pipelines
# that ignore the "hparams" key are unaffected).
for _area in AREAS:
    _stats = _AREA_STATS.get(_area["area_id"])
    _area["hparams"] = derive_hparams(*_stats) if _stats else {}

# ── Convenience helpers ───────────────────────────────────────────────────────

_ID_MAP = {a["area_id"]: a for a in AREAS}


def get_area(area_id: str) -> dict:
    """Return config dict for one area_id; raises KeyError if not found."""
    if area_id not in _ID_MAP:
        raise KeyError(f"Unknown area_id '{area_id}'. Available: {list_area_ids()}")
    return _ID_MAP[area_id]


def list_area_ids() -> list:
    return [a["area_id"] for a in AREAS]


def resolve_areas(area_arg: str) -> list:
    """
    Parse --areas CLI argument → list of config dicts.
    Accepts: 'all', comma-separated IDs, or single ID.
    """
    if area_arg == "all":
        return list(AREAS)
    ids = [s.strip() for s in area_arg.split(",")]
    return [get_area(i) for i in ids]


# ── CLI: print / regenerate the per-area hyper-parameter table ───────────────
if __name__ == "__main__":
    import sys as _sys

    if "--stats" in _sys.argv:
        # Re-measure _AREA_STATS from the patches on disk; print as a literal.
        import pickle as _pickle
        _root = _os_paths.environ.get("GEOCHEM_OUT_ROOT", "outputs")
        print("_AREA_STATS = {")
        for _a in AREAS:
            _md = f"{_root}/{_a['area_id']}/vae_patches/metadata.json"
            _sp = f"{_root}/{_a['area_id']}/vae_patches/dataset_split.pkl"
            if not _os.path.exists(_md):
                continue
            _m  = _json.load(open(_md))
            _np = len(_pickle.load(open(_sp, "rb")).get("train", [])) \
                  if _os.path.exists(_sp) else None
            print(f'    "{_a["area_id"]:<16}: ({_m["resolution_m"]:>7}, '
                  f'{_m["n_channels"]:>3}, {_np}),')
        print("}")
    else:
        hdr = (f"{'area':<18}{'res_m':>8}{'chan':>6}{'patch':>7}"
               f"{'bs':>5}{'ep':>6}{'lat':>6}{'lr':>9}{'lr_d':>9}"
               f"{'min_d':>8}{'radius':>8}")
        print(hdr)
        print("-" * len(hdr))
        for _a in AREAS:
            _h = _a.get("hparams") or {}
            _s = _AREA_STATS.get(_a["area_id"])
            if not _h or not _s:
                continue
            print(f"{_a['area_id']:<18}{_s[0]:>8.0f}{_s[1]:>6}{_s[2]:>7}"
                  f"{_h['batch_size']:>5}{_h['epochs']:>6}{_h['latent_dim']:>6}"
                  f"{_h['lr']:>9.1e}{_h['lr_d']:>9.1e}"
                  f"{_h['min_dist']:>8.2f}{_h['radius_km']:>8.2f}")
