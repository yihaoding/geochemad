"""
Stage 2 (v3): freeze the v3 hierarchical backbone and emit per-sample latent.
"""
import argparse, os, json
import numpy as np, torch

from data     import load_area, build_neighbour_index, collate, iter_batches
from model_v3 import HierarchicalBackbone


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone_ckpt", required=True)
    p.add_argument("--csv_path",      required=True)
    p.add_argument("--target_element",required=True)
    p.add_argument("--out_dir",       required=True)
    p.add_argument("--batch_size",    type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.backbone_ckpt, map_location=device, weights_only=False)
    a    = ckpt["args"]

    ds = load_area(
        args.csv_path, args.target_element,
        transform        = a["transform"],
        input_elements   = a["input_elements"],
        bdl_strategy     = a["bdl_strategy"],
        bdl_threshold    = a["bdl_threshold"],
        paper_compatible = a.get("paper_compatible", True),
    )
    N = len(ds["coords"]); C = ds["elem"].shape[1]
    if C != ckpt["n_elem"]:
        raise ValueError(f"C mismatch: csv={C}  ckpt={ckpt['n_elem']}")

    nbr_idx = build_neighbour_index(ds["coords"], a["k_neighbors"])
    K_total, k_l, k_m = a["k_neighbors"], a["k_local"], a["k_mid"]
    sl_l = slice(0, k_l); sl_m = slice(k_l, k_l + k_m); sl_g = slice(k_l + k_m, K_total)

    # Path B ablation toggles — fall back to defaults for ckpts written before
    # the toggles existed so old runs keep extracting cleanly.
    use_scales = tuple(s.strip() for s in
                       a.get("use_scales", "local,mid,glob").split(",")
                       if s.strip())
    use_scale_tag = not a.get("no_scale_tag", False)
    model = HierarchicalBackbone(
        n_elem=C, d_model=a["d_model"], n_heads=a["n_heads"],
        n_layers_local=a["n_layers_local"],
        n_layers_mid  =a["n_layers_mid"],
        n_layers_glob =a["n_layers_glob"],
        dropout=a["dropout"], k_local=k_l, k_mid=k_m,
        use_scales=use_scales,
        use_scale_tag=use_scale_tag,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    feats = np.zeros((N, a["d_model"]), dtype=np.float32)
    target_idx = ds["target_idx"]
    with torch.no_grad():
        for q_ids in iter_batches(np.arange(N), args.batch_size, shuffle=False):
            batch = collate(ds, nbr_idx, q_ids, device)
            q_elem = batch["q_elem"].clone()
            q_bdl  = batch["q_bdl" ].clone()
            # mask target — same as pretrain — so extraction matches training
            q_elem[:, target_idx] = 0.0
            q_bdl [:, target_idx] = 0.0
            h = model.forward_feature(
                q_elem, q_bdl,
                batch["n_elem"   ][:, sl_l], batch["n_bdl"     ][:, sl_l],
                batch["rel_xy"   ][:, sl_l], batch["n_density" ][:, sl_l],
                batch["n_elem"   ][:, sl_m], batch["n_bdl"     ][:, sl_m],
                batch["rel_xy"   ][:, sl_m], batch["n_density" ][:, sl_m],
                batch["n_elem"   ][:, sl_g], batch["n_bdl"     ][:, sl_g],
                batch["rel_xy"   ][:, sl_g], batch["n_density" ][:, sl_g],
            )
            feats[q_ids] = h.cpu().numpy()

    np.save(os.path.join(args.out_dir, "features.npy"), feats)
    np.save(os.path.join(args.out_dir, "coords.npy"),   ds["coords"])
    json.dump({"n": N, "d_model": a["d_model"], "transform": a["transform"],
               "element_cols": ds["element_cols"], "target_idx": target_idx},
              open(os.path.join(args.out_dir, "meta.json"), "w"), indent=2)
    print(f"[v3-extract] done → {args.out_dir}/features.npy  shape={feats.shape}")


if __name__ == "__main__":
    main()
