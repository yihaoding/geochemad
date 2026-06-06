
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KDTree
from pykrige.ok import OrdinaryKriging
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import random
from scipy.spatial import cKDTree

class PositionalEncoding2D(nn.Module):
    """
    2D sinusoidal positional encoding (longitude, latitude)
    Similar to what ViT or DETR use for 2D image positions.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert dim % 4 == 0, "hidden_dim must be divisible by 4 for 2D encoding"

    def forward(self, coords):
        """
        coords: [B, N, 2] or [B, 2]
        Output: [B, N, dim] or [B, dim]
        """
        if coords.dim() == 2:
            coords = coords.unsqueeze(1)  # [B, 1, 2]
        x = coords[..., 0]
        y = coords[..., 1]
        B, N = x.shape

        div_term = torch.exp(
            torch.arange(0, self.dim // 2, 2, device=coords.device).float()
            * -(np.log(10000.0) / (self.dim // 2))
        )

        # Encode longitude (x) and latitude (y) separately
        pe_x = torch.zeros(B, N, self.dim // 2, device=coords.device)
        pe_y = torch.zeros(B, N, self.dim // 2, device=coords.device)

        pe_x[..., 0::2] = torch.sin(x.unsqueeze(-1) * div_term)
        pe_x[..., 1::2] = torch.cos(x.unsqueeze(-1) * div_term)
        pe_y[..., 0::2] = torch.sin(y.unsqueeze(-1) * div_term)
        pe_y[..., 1::2] = torch.cos(y.unsqueeze(-1) * div_term)

        # Combine into single positional embedding
        pe = torch.cat([pe_x, pe_y], dim=-1)  # [B, N, dim]
        return pe


class GeoTransformer(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, n_heads=4, n_layers=2, num_elements=20):
        super().__init__()
        self.element_embed = nn.Embedding(num_elements, hidden_dim)  # Token 1: element type
        self.input_proj = nn.Linear(in_dim, hidden_dim)              # Neighbor tokens
        self.pos_encoder = PositionalEncoding2D(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.coord_proj = nn.Linear(2, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.linear_hidden =  nn.Linear(hidden_dim, hidden_dim)
        self.linear_prediction =  nn.Linear(hidden_dim, 1)

    def forward(self, element_idx, target_coord, neighbor_tokens):
        """
        Args:
            element_idx: [B] element ID (e.g., Cu=0, Ni=1)
            target_coord: [B, 2] (longitude, latitude)
            neighbor_tokens: [B, K, in_dim] (dx, dy + features)
        """
        B = neighbor_tokens.size(0)

        # 1️⃣ Token 1: element embedding
        token1 = self.element_embed(element_idx).unsqueeze(1)

        # 2️⃣ Token 2: target coordinate with 2D positional encoding
        pe_target = self.pos_encoder(target_coord)  # [B, 1, H]
        token2 = self.coord_proj(target_coord).unsqueeze(1) + pe_target

        # 3️⃣ Neighbor tokens
        pe_neighbors = self.pos_encoder(neighbor_tokens[:, :, :2])
        neigh_proj = self.input_proj(neighbor_tokens) + pe_neighbors

        # Concatenate sequence: [element, target_coord, neighbors...]
        seq = torch.cat([token1, token2, neigh_proj], dim=1)  # [B, 2+K, H]

        # Transformer encoding
        hidden = self.encoder(seq)

        # Use second token (index 1) for regression
        target_hidden = hidden[:, 1, :]                      # Select target token representation
        target_hidden = self.linear_hidden(target_hidden)    # Linear projection
        pre_regression = torch.relu(target_hidden)            # ✅ Apply ReLU activation
        out = self.linear_prediction(pre_regression).squeeze(-1)
        return out, target_hidden
    
# ========== 2️⃣ Preprocessing functions ==========
def half_detection_limit(df, cols):
    """Replace non-positive values with half the min positive per column."""
    df = df.copy()
    for c in cols:
        col = df[c]
        if np.any(col <= 0):
            min_pos = col[col > 0].min() if np.any(col > 0) else 1e-6
            df[c] = np.where(col <= 0, min_pos / 2, col)
    return df


def ilr_transform(X):
    """Simplified ILR transform: log ratio to geometric mean."""
    X = np.clip(X, 1e-12, None)
    gm = np.exp(np.mean(np.log(X), axis=1, keepdims=True))
    return np.log(X / gm)

def extract_hidden_state(
    model_ckpt: str,
    csv_path: str,
    target_element: str,
    target_coord: tuple,
    feature_cols: list = None,
    k_neighbors: int = 128,
    device: str = None,
):
    """
    Extract the hidden (latent) vector of the target element token
    from a trained GeoTransformer model for a specific (X, Y) location.

    Args:
        model_ckpt (str): Path to model checkpoint (.pt).
        csv_path (str): Path to geochemical dataset.
        target_element (str): Target element name (e.g., "Au_ppm").
        target_coord (tuple): (X, Y) coordinate for the query point.
        feature_cols (list, optional): Columns to use as features. Auto-detects if None.
        k_neighbors (int): Number of nearest neighbors.
        device (str, optional): "cuda" or "cpu". Defaults to auto-detect.

    Returns:
        pred_val (float): Predicted element value.
        hidden_vec (np.ndarray): Hidden embedding vector of the target token.
    """

    DEVICE = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Load dataset ----------
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["X", "Y", target_element])

    exclude_cols = ["X", "Y", "SAMPLEID", "SAMPLETYPE", "WAMEX_A_NO", "COMPSAMPID"]
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype != 'O']

    # ---------- Preprocessing (same as training) ----------
    df = half_detection_limit(df, feature_cols)
    df[feature_cols] = ilr_transform(df[feature_cols].values)
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # ---------- Neighbor search ----------
    coords = df[["X", "Y"]].values
    tree = KDTree(coords)

    # ---------- Load model checkpoint ----------
    ckpt = torch.load(model_ckpt, map_location=DEVICE)
    # backward compatibility: check if args are stored
    if "args" in ckpt:
        args = ckpt["args"]
        hidden_dim = args.get("hidden_dim", 128)
        num_layers = args.get("num_layers", 2)
        num_elements = args.get("num_elements", 20)
    else:
        hidden_dim, num_layers, num_elements = 128, 2, 20  # defaults

    model = GeoTransformer(
        in_dim=len(feature_cols) + 2,
        hidden_dim=hidden_dim,
        n_layers=num_layers,
        num_elements=num_elements
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ---------- Build input tokens ----------
    x, y = target_coord
    dist, ind = tree.query([[x, y]], k=k_neighbors)
    neighbor_rows = df.iloc[ind[0]]
    dxy = neighbor_rows[["X", "Y"]].values - np.array([x, y])
    feats = neighbor_rows[feature_cols].values
    tokens = np.hstack([dxy, feats])

    tokens = torch.tensor(tokens, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    coord = torch.tensor([x, y], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    element_idx = torch.tensor([0], dtype=torch.long).to(DEVICE)  # Au=0, adapt if mapping known

    # ---------- Forward pass ----------
    with torch.no_grad():
        pred_val, hidden_state = model(element_idx, coord, tokens)
        pred_val = float(pred_val.cpu().item())
        hidden_vec = hidden_state.cpu().numpy().squeeze()

    print(f"✅ Extracted hidden state for {target_element} at {target_coord}")
    print(f"Predicted {target_element}: {pred_val:.6f}")
    print(f"Hidden vector shape: {hidden_vec.shape}")

    return pred_val, hidden_vec