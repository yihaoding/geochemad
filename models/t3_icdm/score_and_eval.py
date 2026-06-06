"""
Stage 4+5: score every sample with the trained AnomalyT1 + run 20-run
negative-sampling evaluation (matches paper main-table reporting protocol).

Anomaly score = per-sample reconstruction MSE  (∑ over elements)/C

The 20-run eval comes from the existing shared evaluate.py used for VAE/T1
results in the paper main table, so metrics are directly comparable.
"""

import argparse, os, json, sys
import numpy as np, torch
import pandas as pd
import torch.nn.functional as F

from data  import load_area, build_neighbour_index, collate, iter_batches
from model import AnomalyT1
from train_anomaly import build_point_features, collate_aug

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vendor"))
from evaluate import evaluate, _standardize_xy        # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--anomaly_ckpt",  required=True)
    p.add_argument("--features_npy",  required=True)
    p.add_argument("--meta_json",     required=True)
    p.add_argument("--csv_path",      required=True)
    p.add_argument("--target_element",required=True)
    p.add_argument("--deposits_csv",  required=True)
    p.add_argument("--out_dir",       required=True)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--radius_km",     type=float, required=True)
    p.add_argument("--min_dist",      type=float, required=True)
    p.add_argument("--neg_runs",      type=int,   default=20)
    p.add_argument("--label",         default="icdm")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.anomaly_ckpt, map_location=device, weights_only=False)
    am_args = ckpt["args"]
    meta = json.load(open(args.meta_json))

    # Reload area with the EXACT preprocessing the anomaly model was trained on.
    ds = load_area(
        args.csv_path, args.target_element,
        transform      = ckpt["transform"],
        input_elements = ",".join(ckpt["element_cols"]),
        bdl_strategy   = am_args.get("bdl_strategy", "flag"),
        bdl_threshold  = am_args.get("bdl_threshold", 0.95),
    )
    N = len(ds["coords"]); C = ds["elem"].shape[1]
    if C != ckpt["n_elem"]:
        raise ValueError(f"C mismatch: csv={C} ckpt={ckpt['n_elem']}")

    h_raw  = np.load(args.features_npy)
    h_feat = build_point_features(ds, h_raw)
    d_backbone = ckpt["d_backbone"]

    nbr_idx = build_neighbour_index(ds["coords"], am_args["k_neighbors"])

    model = AnomalyT1(
        n_elem=C, d_backbone=d_backbone,
        d_model=am_args["d_model"], nhead=am_args["n_heads"],
        num_enc_layers=am_args["n_enc_layers"],
        num_dec_layers=am_args["n_dec_layers"],
        dim_feedforward=am_args["dim_feedforward"],
        dropout=am_args["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── Stage 4: score every sample (reconstruction MSE) ───────────────────
    scores = np.zeros(N, dtype=np.float32)
    all_ids = np.arange(N)
    with torch.no_grad():
        for q_ids in iter_batches(all_ids, args.batch_size, shuffle=False):
            q_xy, q_feat, rel_xy, n_feat = collate_aug(ds, h_feat, nbr_idx, q_ids, device)
            recon  = model(q_xy, q_feat, rel_xy, n_feat)
            target = torch.tensor(ds["elem"][q_ids], device=device)
            mse    = ((recon - target) ** 2).mean(dim=-1).cpu().numpy()
            scores[q_ids] = mse

    # Write anomaly_scores.csv (x, y, anomaly_score) for evaluate.py
    df = pd.DataFrame({"x": ds["coords"][:, 0],
                       "y": ds["coords"][:, 1],
                       "anomaly_score": scores})
    scores_csv = os.path.join(args.out_dir, "anomaly_scores.csv")
    df.to_csv(scores_csv, index=False)
    np.save(os.path.join(args.out_dir, "scores.npy"), scores)
    print(f"[stage4] scored N={N} → {scores_csv}")

    # ── Stage 5: 20-run negative-sampling eval (multi-metric) ──────────────
    dep_df = _standardize_xy(pd.read_csv(args.deposits_csv))
    samp_df= _standardize_xy(df)
    samp_xy     = samp_df[["x", "y"]].to_numpy()
    samp_scores = samp_df["anomaly_score"].to_numpy()
    dep_xy      = dep_df[["x", "y"]].to_numpy()
    metrics = evaluate(
        samp_xy, samp_scores, dep_xy,
        radius_km=args.radius_km, min_dist=args.min_dist,
        neg_runs=args.neg_runs, label=args.label,
    )
    json.dump(metrics, open(os.path.join(args.out_dir, "metrics.json"), "w"), indent=2)
    print(f"[stage5] metrics saved → {args.out_dir}/metrics.json")


if __name__ == "__main__":
    main()
