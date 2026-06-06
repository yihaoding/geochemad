#!/usr/bin/env python3
"""Fill ZS/MD PCA cells for the 4-area panel.

For each area, computes the per-area PCA-k element subset (k = LLM
cardinality, matching the T1 PCA ablation), then runs ZS/MD/KNN/OCSVM
on that subset. Output: outputs_phase3_fs/<full>/{zscore,mahal,knn,ocsvm}_pca/.
This subset is more comparable to LLM than the legacy *_pca15 dirs.
"""
from __future__ import annotations
import os
import sys, re, copy
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("GEOCHEM_OUT_ROOT", "outputs"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "t3_icdm"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # repo root
from data import _element_columns, _detect_bdl, _half_dl, _pca_topk  # noqa: E402
from baselines.run_baselines import run_area               # noqa: E402
from shared.area_config import AREAS as AREA_CFGS          # noqa: E402

PANEL = [("sed1","area1_sed1_au"),("sed3","area15_sed3_au"),
         ("rock2","area4_rock2_au"),("soil2","area7_soil2_au")]


def llm_k(short: str) -> int:
    sh = (ROOT / f"scripts/slurm/run_icdm_aligned_input_{short}.sh").read_text()
    m = re.search(r'^LLM_ELEMS_FULL="([^"]+)"', sh, re.M)
    return len([c for c in m.group(1).split(",") if c.strip()]) if m else 0


def main():
    cfg_by_id = {a["area_id"]: a for a in AREA_CFGS}
    for short, full in PANEL:
        cfg = cfg_by_id[full]; hp = cfg["hparams"]
        k = llm_k(short)
        # Load CSV, get all element-like columns, compute PCA top-k
        df = pd.read_csv(cfg["csv_path"], low_memory=False)
        elem_cols_all = _element_columns(df, paper_compatible=True)
        raw_all = df[elem_cols_all].to_numpy(dtype=np.float64)
        bdl = _detect_bdl(raw_all)
        x = _half_dl(raw_all, bdl)
        pca_elems = _pca_topk(x, elem_cols_all, k=k,
                              target_element=cfg["target_element"])
        if cfg["target_element"] not in pca_elems:
            pca_elems = pca_elems + [cfg["target_element"]]
        print(f"\n=== {short}  k={k}  PCA selected {len(pca_elems)} elements ===")
        print(f"    {pca_elems[:12]}{'...' if len(pca_elems) > 12 else ''}")

        cfg2 = copy.deepcopy(cfg)
        cfg2["input_elements"] = pca_elems
        run_area(cfg2, ["zscore", "mahal", "knn", "ocsvm"],
                 out_root=str(ROOT / "outputs_phase3_fs"),
                 transform="clr",
                 radius_km=hp["radius_km"], min_dist=hp["min_dist"],
                 neg_runs=20, out_suffix="_pca")


if __name__ == "__main__":
    main()
