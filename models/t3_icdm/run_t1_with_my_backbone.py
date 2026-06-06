"""
Run paper T1 (new_geo_transformer_main.py) with MY backbone features as `tfeats`.

The only deviation from paper T1 is that we bypass paper's
`extract_hidden_features_from_model` (which requires a checkpoint in
our_model.GeoTransformer format) and instead load a pre-computed
features array of shape [N, hidden_dim] from --tfeats_npy.

Everything else (model, training loop, eval, contrastive distillation loss)
is exactly the paper T1 code path.
"""

import argparse, os, sys, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vendor")
sys.path.insert(0, SCR)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vendor"))

from geochem_preprocessor import preprocess_geochem_data_new       # noqa: E402
from new_geo_transformer    import (                                # noqa: E402
    GeoTransformer, GeoDataset, infer_scores, collate,
    evaluate_and_visualize, train_one_epoch, evaluate_and_visualize_multineg,
)


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",                required=True)
    ap.add_argument("--deposits",            required=True)
    ap.add_argument("--epochs",              type=int,   default=60)
    ap.add_argument("--batch",               type=int,   default=16)
    ap.add_argument("--lr",                  type=float, default=1e-3)
    ap.add_argument("--contrastive_weight",  type=float, default=0.01)
    ap.add_argument("--checkpoint",          required=True)
    ap.add_argument("--output",              required=True)
    ap.add_argument("--mode",                default="both",
                    choices=["train", "infer", "both"])
    ap.add_argument("--device",              default=None)
    ap.add_argument("--seed",                type=int,   default=42)
    ap.add_argument("--mapping_methods",     default="nn")
    ap.add_argument("--radius_km",           type=float, default=2.0)
    ap.add_argument("--min_dist",            type=float, default=5.0)
    ap.add_argument("--neg_runs",            type=int,   default=20)
    ap.add_argument("--target_elements",     default=None)
    ap.add_argument("--n_elements",          type=int,   default=None)
    ap.add_argument("--abnormal_method",     default="half_detection")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--distance_constraint",    dest="distance_constraint",
                   action="store_true")
    g.add_argument("--no_distance_constraint", dest="distance_constraint",
                   action="store_false")
    ap.set_defaults(distance_constraint=True)
    ap.add_argument("--transform",           default="clr")
    ap.add_argument("--feature_selection",   default=None)
    ap.add_argument("--target_element",      default=None)
    # ICDM-specific: pre-computed backbone features
    ap.add_argument("--tfeats_npy",          default=None,
                    help="[N, hidden_dim] features.npy from my Stage 2 extraction")
    ap.add_argument("--coords_npy",          default=None,
                    help="[N, 2] coords.npy from my Stage 2 (for sanity check)")
    ap.add_argument("--tfeats_mode",         default="distill",
                    choices=["distill", "input", "off"],
                    help="distill: use tfeats as recon_target (paper T1 default). "
                         "input: concat tfeats to X_ilr as extra channels, no distill loss. "
                         "off: ignore tfeats (= paper T1 baseline)")
    ap.add_argument("--ckpt_metric",         default="loss",
                    choices=["loss", "top_gap_mad", "kurtosis", "skewness",
                             "bimodality", "moran_score", "p99_over_p50",
                             "best_auc"],
                    help="loss / unsupervised (top_gap_mad, kurtosis, "
                         "skewness, bimodality, moran_score, p99_over_p50) "
                         "/ best_auc (supervised).")
    ap.add_argument("--ckpt_warmup_epochs",  type=int, default=5,
                    help="Epochs before ckpt_metric starts counting.")
    ap.add_argument("--top_frac",            type=float, default=0.05,
                    help="top fraction for top_gap_mad criterion.")
    ap.add_argument("--moran_k",             type=int, default=8,
                    help="k for KNN graph used in Moran's I.")
    ap.add_argument("--early_stop_patience", type=int, default=0,
                    help="Stop if ckpt_metric hasn't improved for N epochs "
                         "(0 = disabled). Counted after warmup.")
    ap.add_argument("--distill_weight", type=float, default=1.0,
                    help="Weight on L1(recon_target, tfeats). 1.0=original, "
                         "0.3=gadformer-aligned softer distillation.")
    ap.add_argument("--ctx_knn", type=int, default=None,
                    help="If set, T1 attention context = KNN-k nearest "
                         "neighbours (matches gadformer T1's --ctx_knn). "
                         "Implemented via distance_constraint=True + huge "
                         "radius + max_ctx=k.")
    return ap.parse_args()


def _morans_i(values, knn_idx):
    """Moran's I with row-standardized KNN weights. From gadformer/
    train_with_unsup_metrics.py:moran_i_rowstd."""
    z = values - values.mean()
    lag = z[knn_idx].mean(axis=1)
    den = (z * z).sum()
    if den <= 0:
        return 0.0
    return float((z * lag).sum() / den)


def _all_separation_metrics(scores, top_frac=0.05, knn_idx=None):
    """Compute every unsupervised ckpt-selection metric in one pass.
    Adapted from run_t1_cl.py and gadformer/train_with_unsup_metrics.py.
    Returns a dict; np.nan if undefined."""
    from scipy.stats import kurtosis as _kurt, skew as _skew
    s = np.asarray(scores, dtype=np.float64)
    s_finite = s[np.isfinite(s)]
    out = {k: float("nan") for k in ("top_gap_mad", "kurtosis", "skewness",
                                      "bimodality", "moran_score",
                                      "p99_over_p50")}
    n = s_finite.size
    if n < 8:
        return out
    k_top = max(1, int(np.ceil(n * top_frac)))
    top = np.partition(s_finite, -k_top)[-k_top:]
    med = np.median(s_finite)
    mad = np.median(np.abs(s_finite - med))
    out["top_gap_mad"] = float((top.mean() - med) / (mad + 1e-12))
    out["kurtosis"]    = float(_kurt(s_finite, fisher=True,  bias=False))
    out["skewness"]    = float(_skew(s_finite,              bias=False))
    g  = _skew(s_finite, bias=False)
    kt = _kurt(s_finite, fisher=False, bias=False)   # non-Fisher (normal=3)
    denom = kt + 3.0 * (n - 1) ** 2 / max(1.0, (n - 2) * (n - 3))
    out["bimodality"]  = float((g * g + 1.0) / denom) if denom > 0 else 0.0
    p99 = float(np.percentile(s_finite, 99))
    p50 = float(np.percentile(s_finite, 50))
    out["p99_over_p50"] = p99 / max(p50, 1e-12)
    if knn_idx is not None and s.shape[0] == knn_idx.shape[0]:
        out["moran_score"] = _morans_i(s, knn_idx)
    return out


def main():
    args = get_args()
    print("========== Parsed Arguments ==========")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("======================================")

    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output)     or ".", exist_ok=True)
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    if args.target_elements is not None:
        target_elements = args.target_elements.split(",")
    else:
        target_elements = None

    # Always pass model_ckpt=None so the paper code does not try to extract
    # features via its own extractor — we provide tfeats directly below.
    work_df, X_ilr, coords, elem_cols, _ = preprocess_geochem_data_new(
        args.data,
        target_elements,
        model_ckpt=None,
        device=device,
        verbose=True,
        target_element=args.target_element,
        transform=args.transform,
        n_elements=args.n_elements,
        abnormal_method=args.abnormal_method,
    )

    # ── Load my pre-computed backbone features ──────────────────────────────
    if args.tfeats_npy is not None and args.tfeats_mode != "off":
        tfeats_raw = np.load(args.tfeats_npy).astype(np.float32)
        print(f"[ICDM] loaded tfeats: {tfeats_raw.shape}  mode={args.tfeats_mode}")
        if args.coords_npy:
            coords_extracted = np.load(args.coords_npy)
            if coords_extracted.shape != coords.shape:
                print(f"[ICDM][warn] coords shape mismatch  "
                      f"npy={coords_extracted.shape}  csv={coords.shape}")
        if tfeats_raw.shape[0] != X_ilr.shape[0]:
            raise ValueError(
                f"tfeats N={tfeats_raw.shape[0]} does not match dataset N={X_ilr.shape[0]}. "
                f"Did you re-run Stage 2 after changing transform/element subset?")

        if args.tfeats_mode == "distill":
            # tfeats fed to dataset → enters loss as L1(recon_target, tfeats)
            tfeats = tfeats_raw
        elif args.tfeats_mode == "input":
            # concat tfeats to X_ilr as input channels;
            # dataset gets tfeats=None so l_rec_target collapses to 0
            X_ilr = np.concatenate([X_ilr, tfeats_raw], axis=1).astype(np.float32)
            tfeats = None
            print(f"[ICDM] tfeats_mode=input → augmented X_ilr to {X_ilr.shape}; "
                  f"distillation loss disabled")
    else:
        tfeats = None
        if args.tfeats_mode == "off":
            print("[ICDM] tfeats_mode=off → ignoring backbone features (paper T1 baseline)")
        else:
            print("[ICDM] no tfeats — running pure paper T1 baseline")

    print(f"Feature shape: {X_ilr.shape}, Coords shape: {coords.shape}")

    feat_dim = X_ilr.shape[1]
    model = GeoTransformer(feat_dim=feat_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.mode in ["train", "both"]:
        # KNN-k context (gadformer-style): use huge radius + cap with max_ctx.
        if args.ctx_knn is not None and args.ctx_knn > 0:
            ds_kwargs = dict(distance_constraint=True,
                             radius_km=99999.0,
                             max_ctx=args.ctx_knn,
                             max_mask_ratio=0.5)
            print(f"[T1] context = KNN-{args.ctx_knn} nearest neighbours "
                  f"(gadformer-aligned)")
        else:
            ds_kwargs = dict(distance_constraint=args.distance_constraint,
                             max_mask_ratio=0.5)
            print(f"[T1] context = "
                  f"{'distance-constrained' if args.distance_constraint else 'all-but-self'}")
        print(f"[T1] distill_weight = {args.distill_weight}")
        dataset = GeoDataset(X_ilr, coords, tfeats, **ds_kwargs)
        loader  = DataLoader(dataset, batch_size=args.batch,
                             shuffle=True, collate_fn=collate)

        # higher-is-better for every metric except "loss"
        HIGHER_BETTER = {"top_gap_mad", "kurtosis", "skewness",
                         "bimodality", "moran_score", "p99_over_p50",
                         "best_auc"}
        higher_better = (args.ckpt_metric in HIGHER_BETTER)
        best_crit = float("-inf") if higher_better else float("inf")
        best_epoch = -1
        no_improve = 0

        # Track best-per-metric so one training run produces N selector outputs.
        ALL_METRICS = ["loss", "top_gap_mad", "kurtosis", "skewness",
                       "bimodality", "moran_score", "p99_over_p50"]
        per_metric_best = {m: (float("-inf") if m in HIGHER_BETTER else float("inf"))
                           for m in ALL_METRICS}
        per_metric_ep   = {m: -1 for m in ALL_METRICS}
        import os as _os
        _out_dir = _os.path.dirname(args.output) or "."
        _out_base = _os.path.splitext(_os.path.basename(args.output))[0]
        print(f"[ckpt] selection = {args.ckpt_metric}  warmup={args.ckpt_warmup_epochs}"
              f"  top_frac={args.top_frac}  moran_k={args.moran_k}"
              f"  patience={args.early_stop_patience}")
        print(f"[ckpt] per-epoch panel logs ALL metrics so post-hoc you can "
              f"pick any selection criterion.")

        # precompute KNN index for Moran's I (uses raw coords, not augmented X)
        from sklearn.neighbors import NearestNeighbors
        _nbrs   = NearestNeighbors(n_neighbors=args.moran_k + 1).fit(coords)
        _, _idx = _nbrs.kneighbors(coords)
        knn_idx = _idx[:, 1:]  # drop self

        for ep in range(1, args.epochs + 1):
            r_all, c, r_target = train_one_epoch(
                model, loader, opt, epoch=ep, total_epochs=args.epochs,
                distill_weight=args.distill_weight,
                contrastive_weight=args.contrastive_weight)
            total = r_all + args.distill_weight * r_target + args.contrastive_weight * c
            print(f"Epoch {ep:02d} | L_all_rec={r_all:.4f}| "
                  f"L_target_rec={r_target:.4f} | L_con={c:.4f} | Total={total:.4f}")

            # ---- compute scores once with KNN context, reuse for eval ----
            scores, _ = infer_scores(model, X_ilr, coords, device=device,
                                     ctx_knn=args.ctx_knn)
            sep_metrics = _all_separation_metrics(
                scores, top_frac=args.top_frac, knn_idx=knn_idx)

            # eval gives us supervised AUC (and prints metrics);
            # pass precomputed_scores so it doesn't redo the inference.
            eval_metrics = None
            try:
                eval_metrics = evaluate_and_visualize_multineg(
                    model, X_ilr, coords, work_df,
                    epoch=ep,
                    CHECKPOINT_PATH=args.checkpoint,
                    DEPOSITS_CSV=args.deposits,
                    DEPOSIT_MAPPING_METHOD=args.mapping_methods,
                    DEPOSIT_RADIUS_KM=args.radius_km,
                    NEG_MIN_DIST_KM=args.min_dist,
                    NEG_SAMPLE_RUNS=args.neg_runs,
                    precomputed_scores=scores,
                )
            except Exception as e:
                print(f"  [eval epoch {ep}] {e}")
            auc_mean = float(eval_metrics["AUC_mean"]) if eval_metrics else float("nan")

            # one-line machine-parseable panel
            print(f"[PANEL] ep={ep:02d}  loss={total:.4f}  "
                  f"top_gap_mad={sep_metrics['top_gap_mad']:+.4f}  "
                  f"kurtosis={sep_metrics['kurtosis']:+.4f}  "
                  f"skewness={sep_metrics['skewness']:+.4f}  "
                  f"bimodality={sep_metrics['bimodality']:+.4f}  "
                  f"moran_score={sep_metrics['moran_score']:+.4f}  "
                  f"p99_over_p50={sep_metrics['p99_over_p50']:+.4f}  "
                  f"AUC={auc_mean:.4f}")

            # ---- per-metric best-scores saving ---------------------------
            # For each tracked metric, save anomaly_scores_<metric>.csv at
            # the epoch where that metric is best. Post-hoc selection: just
            # pick whichever file corresponds to the desired criterion.
            _vals = {**sep_metrics, "loss": total}
            for _m in ALL_METRICS:
                _v = _vals.get(_m)
                if _v is None or not np.isfinite(_v): continue
                if ep <= args.ckpt_warmup_epochs: continue
                _hb = _m in HIGHER_BETTER
                _improved = (_v > per_metric_best[_m] + 1e-8) if _hb else \
                            (_v < per_metric_best[_m] - 1e-8)
                if _improved:
                    per_metric_best[_m] = _v
                    per_metric_ep[_m] = ep
                    import pandas as _pd
                    _pd.DataFrame({"X": coords[:, 0], "Y": coords[:, 1],
                                   "anomaly_score": scores}).to_csv(
                        f"{_out_dir}/{_out_base}_{_m}.csv", index=False)

            # ---- selection ------------------------------------------------
            if args.ckpt_metric == "loss":
                new_crit = total
            elif args.ckpt_metric == "best_auc":
                new_crit = auc_mean
            else:
                new_crit = sep_metrics[args.ckpt_metric]

            eligible = (ep > args.ckpt_warmup_epochs) and np.isfinite(new_crit)
            if higher_better:
                improved = eligible and (new_crit > best_crit + 1e-8)
            else:
                improved = eligible and (new_crit < best_crit - 1e-8)

            if improved:
                best_crit, best_epoch = new_crit, ep
                no_improve = 0
                torch.save({"model_state_dict": model.state_dict(),
                            "feat_dim": feat_dim, "args": vars(args),
                            "ckpt_metric": args.ckpt_metric,
                            "ckpt_metric_val": best_crit,
                            "epoch": ep},
                           args.checkpoint)
                print(f"  ✅ Improved [{args.ckpt_metric}={new_crit:+.4f}]"
                      f"  Model saved at epoch {ep}")
            elif eligible:
                no_improve += 1
                if (args.early_stop_patience > 0
                        and no_improve >= args.early_stop_patience):
                    print(f"\n[early stop] {no_improve} epochs without "
                          f"improvement on {args.ckpt_metric}; halting at ep {ep}.")
                    break

        print(f"\n[ckpt] best epoch = {best_epoch}  "
              f"{args.ckpt_metric} = {best_crit:.4f}")
        print(f"[ckpt] per-metric best epochs (anomaly_scores_<m>.csv written):")
        for _m in ALL_METRICS:
            print(f"  {_m:>16}: ep={per_metric_ep[_m]:>2d}  val={per_metric_best[_m]:+.4f}")

    if args.mode in ["infer", "both"]:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        scores, _ = infer_scores(model, X_ilr, coords, device=device,
                                 ctx_knn=args.ctx_knn)
        import pandas as pd
        pd.DataFrame({"X": coords[:, 0], "Y": coords[:, 1],
                      "anomaly_score": scores}).to_csv(args.output, index=False)
        print(f"[infer] scores → {args.output}")

        # final 20-run eval on best ckpt (reuse the scores we just computed)
        evaluate_and_visualize_multineg(
            model, X_ilr, coords, work_df,
            epoch=-1,
            CHECKPOINT_PATH=args.checkpoint,
            DEPOSITS_CSV=args.deposits,
            DEPOSIT_MAPPING_METHOD=args.mapping_methods,
            DEPOSIT_RADIUS_KM=args.radius_km,
            NEG_MIN_DIST_KM=args.min_dist,
            NEG_SAMPLE_RUNS=args.neg_runs,
            output_path=args.output,
            precomputed_scores=scores,
        )


if __name__ == "__main__":
    main()
