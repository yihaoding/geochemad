#!/usr/bin/env python3
"""Fill ZS/MD/KNN/OCSVM NoFS + OCSVM LLM gaps for the 4-area table.

Reuses the existing classical-baseline runner by patching area_cfg in-place
before calling run_baselines.run_area().

  NoFS    -> input_elements = full element list from data.py's
             _element_columns (paper-regex + Canadian fallback so sed3
             also gets ~71 elements instead of degenerating to 1).
             Output: outputs_phase3_fs/<full>/{zscore,mahal,knn,ocsvm}_nofs/

  OCSVM LLM (missing in outputs/<full>/ocsvm/)
          -> per-area LLM_ELEMS_FULL parsed from the aligned_input script.
             Output: outputs/<full>/ocsvm/
"""
from __future__ import annotations
import os
import sys, re, time, copy
from pathlib import Path
import pandas as pd

ROOT = Path(os.environ.get("GEOCHEM_OUT_ROOT", "outputs"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "t3_icdm"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # repo root
from data import _element_columns                          # noqa: E402
from baselines.run_baselines import run_area               # noqa: E402
from shared.area_config import AREAS as AREA_CFGS          # noqa: E402

PANEL = [("sed1","area1_sed1_au"),("sed3","area15_sed3_au"),
         ("rock2","area4_rock2_au"),("soil2","area7_soil2_au")]

def llm_elems(short: str) -> list[str]:
    sh = (ROOT / f"scripts/slurm/run_icdm_aligned_input_{short}.sh").read_text()
    m = re.search(r'^LLM_ELEMS_FULL="([^"]+)"', sh, re.M)
    return [c.strip() for c in m.group(1).split(",") if c.strip()] if m else []


def main():
    cfg_by_id = {a["area_id"]: a for a in AREA_CFGS}
    t0 = time.time()

    for short, full in PANEL:
        cfg = cfg_by_id[full]; hp = cfg["hparams"]

        df_head = pd.read_csv(cfg["csv_path"], nrows=2, low_memory=False)
        nofs_elems = _element_columns(df_head, paper_compatible=True)
        if cfg["target_element"] not in nofs_elems:
            nofs_elems = nofs_elems + [cfg["target_element"]]
        print(f"\n=== {short}  NoFS element count = {len(nofs_elems)} ===")

        cfg_nofs = copy.deepcopy(cfg)
        cfg_nofs["input_elements"] = nofs_elems
        run_area(
            cfg_nofs, methods=["zscore", "mahal", "knn", "ocsvm"],
            out_root=str(ROOT / "outputs_phase3_fs"),
            transform="clr",
            radius_km=hp["radius_km"], min_dist=hp["min_dist"],
            neg_runs=20, out_suffix="_nofs",
        )

        # OCSVM with LLM list (fill missing column)
        llm = llm_elems(short)
        if llm:
            cfg_llm = copy.deepcopy(cfg)
            cfg_llm["input_elements"] = llm
            run_area(
                cfg_llm, methods=["ocsvm"],
                out_root=str(ROOT / "outputs"),
                transform="clr",
                radius_km=hp["radius_km"], min_dist=hp["min_dist"],
                neg_runs=20, out_suffix="",
            )

    print(f"\n[total wall time = {time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
