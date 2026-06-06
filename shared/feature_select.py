"""
feature_select.py  –  Feature (element) selection for the VAE patch pipeline.

Three modes, exposed as --feature_selection {none, llm, pca} in
  vae/generate_vae_patches.py, vae/run_interpolation.py, vae/train_vae.py

Mode  Element list source                                       Output suffix
----  --------------------------------------------------------- -------------
none  area_cfg['base_elements'] (or auto-detect when None)      ""        → vae_patches/      + vae/ vaegan/ …
llm   area_cfg['llm_elements']  (shared/llm_feature_selection)  "_fs"     → vae_patches_fs/   + vae_fs/  vaegan_fs/  …
pca   pca_select_columns(csv, target_element, top_k)            "_pca"    → vae_patches_pca/  + vae_pca/ vaegan_pca/ …

The 'pca' mode is a target-aware, column-name-preserving variant of
preprocessing/feature_selection.pca_selection (which detects the target from
the filename and strips unit suffixes — neither works for our shared-CSV areas
or for downstream patch channel naming).
"""

from __future__ import annotations
from typing import Callable, Iterable, List

import numpy as np
import pandas as pd


# Output directory suffix per mode.
FS_SUFFIX = {"none": "", "llm": "_fs", "pca": "_pca"}
FS_CHOICES = list(FS_SUFFIX)


def pca_select_columns(csv_path: str,
                       target_element: str,
                       top_k: int = 15) -> List[str]:
    """Select the target column plus the `top_k` columns most correlated with it.

    Mirrors the spirit of preprocessing.feature_selection.pca_selection
    (correlation ranking after standardisation) but:
      * takes the target column **explicitly** (works for shared CSVs like
        area11-14_geochem.csv where filename-based detection fails),
      * **keeps the exact CSV column names** (no _ppm/_pct stripping) so they
        line up with the patch-folder naming used downstream,
      * **includes the target itself** in the returned list — the VAE
        reconstructs every channel and the target is the most informative one.

    Returns: [target_element, top-k correlated features], length ≤ top_k+1.
    """
    df = pd.read_csv(csv_path, low_memory=False).replace(-9999, np.nan)
    if target_element not in df.columns:
        raise ValueError(
            f"pca_select_columns: target '{target_element}' not in {csv_path}")

    drop_cols = {"X", "Y", "SAMPLEID", "SAMPLETYPE", "WAMEX_A_NO", "COMPSAMPID",
                 "ID", "LONGITUDE", "LATITUDE", "EASTING", "NORTHING"}
    num = df.drop(columns=[c for c in drop_cols if c in df.columns],
                  errors="ignore").select_dtypes(include=[np.number])

    y = num[target_element].to_numpy(float)
    feat_cols = [c for c in num.columns if c != target_element]

    corr = []
    for c in feat_cols:
        x = num[c].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 10 or np.nanstd(x[m]) == 0:
            continue
        cc = np.corrcoef(x[m], y[m])[0, 1]
        if np.isfinite(cc):
            corr.append((c, abs(cc)))

    corr.sort(key=lambda t: t[1], reverse=True)
    top = [c for c, _ in corr[:top_k]]
    return [target_element] + top


def select_raw_cols(area_cfg: dict,
                    df_cols: Iterable[str],
                    fs: str,
                    pca_top_k: int,
                    auto_detect: Callable[[], Iterable[str]]) -> List[str]:
    """Resolve the element column list for one area under feature-selection mode `fs`.

    Parameters
    ----------
    area_cfg     : entry from shared.area_config.AREAS
    df_cols      : iterable of column names present in the loaded dataframe
    fs           : 'none' | 'llm' | 'pca'
    pca_top_k    : top-k for 'pca' mode (target is added on top → k+1 channels)
    auto_detect  : callable returning the full element list when fs='none' and
                   the area has no explicit base_elements spec.  Each script
                   passes its own auto-detect to keep parity with its existing
                   behaviour (`get_df_element` vs `EXCLUDE_COLS`).

    Returns: ordered, de-duplicated list of column names that exist in `df_cols`.
    """
    if fs not in FS_SUFFIX:
        raise ValueError(f"feature_selection={fs!r}, must be one of {FS_CHOICES}")

    if fs == "llm":
        sel = area_cfg.get("llm_elements")
        if not sel:
            raise ValueError(
                f"--feature_selection llm: no LLM list for {area_cfg['area_id']} "
                f"(shared/llm_feature_selection.json does not cover this area)")
        cols = list(sel)
    elif fs == "pca":
        cols = pca_select_columns(area_cfg["csv_path"],
                                  area_cfg["target_element"],
                                  top_k=pca_top_k)
    else:  # none
        base = area_cfg.get("base_elements")
        cols = list(base) if base else list(auto_detect())

    # Filter to columns actually present, de-dup while preserving order.
    present = set(df_cols)
    seen: set = set()
    out: List[str] = []
    for c in cols:
        if c in present and c not in seen:
            seen.add(c)
            out.append(c)
    return out
