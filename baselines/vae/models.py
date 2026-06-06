"""
geo_models.py
=============
Six model architectures for geochemical anomaly detection.

Model           | Decoder          | Adversarial | Stochastic
-----------     | ---------------  | ----------- | ----------
AEModel         | bilinear+conv    | no          | no
DAEModel        | bilinear+conv    | no          | no (noise at train)
VAEModel        | bilinear+conv    | no          | yes (KL)
VAEGANModel     | bilinear+conv    | yes (SN-D)  | yes (KL)
VAECascadeGAN   | multi-scale      | yes (SN-D)  | yes (KL)
VAEDiffModel    | DDPM UNet        | no          | yes (KL)

All encoders share: SE channel-attention + InstanceNorm convolutions.
Each model exposes:
  encode(x)       -> (z, mu, logvar)   [AE/DAE: (z, None, None)]
  reconstruct(x)  -> x_hat             (final full-res)
  score_pixel(x)  -> (B, H, W)         per-pixel anomaly score
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# Shared building blocks
# ─────────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention.

    For geochemical data the key anomaly signal is *which elements
    co-occur*, not just their individual magnitudes. SE learns a
    per-element importance weight from the global channel statistics,
    suppressing irrelevant channels and amplifying the combinations
    that define normal background geochemistry.
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(2, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, mid), nn.ReLU(True),
            nn.Linear(mid, channels), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x).view(x.size(0), -1, 1, 1)


class Encoder(nn.Module):
    """Shared encoder: SE channel-attention + InstanceNorm convolutions.

    InstanceNorm (not BatchNorm) is used because it normalises each
    sample independently, preserving the per-sample deviation from
    the dataset mean that constitutes the anomaly signal.
    """
    def __init__(self, in_channels: int = 12, latent_dim: int = 64,
                 img_size: int = 16):
        super().__init__()
        IN = lambda c: nn.InstanceNorm2d(c, affine=True)
        self.ch_attn = SEBlock(in_channels, reduction=max(1, in_channels // 4))
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64,  4, 2, 1), IN(64),  nn.LeakyReLU(0.2, True),
            nn.Conv2d(64,  128, 4, 2, 1), IN(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1), IN(256), nn.LeakyReLU(0.2, True),
        )
        with torch.no_grad():
            h = self.conv(torch.zeros(1, in_channels, img_size, img_size))
            flat = h.view(1, -1).size(1)
        self.fc_mu     = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.conv(self.ch_attn(x)).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()


class Decoder(nn.Module):
    """Standard decoder: bilinear upsample + Conv.

    ConvTranspose2d produces checkerboard artefacts (periodic errors
    unrelated to geochemistry). Bilinear upsampling gives smoother
    reconstructions so reconstruction error is a reliable anomaly score.
    """
    def __init__(self, out_channels: int = 12, latent_dim: int = 64,
                 img_size: int = 16):
        super().__init__()
        _m = nn.Sequential(nn.Conv2d(out_channels, 64, 4, 2, 1),
                           nn.Conv2d(64, 128, 4, 2, 1),
                           nn.Conv2d(128, 256, 4, 2, 1))
        with torch.no_grad():
            h = _m(torch.zeros(1, out_channels, img_size, img_size))
            self._hw = h.shape[-2:]
        self.fc = nn.Linear(latent_dim, 256 * self._hw[0] * self._hw[1])
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64,  3, 1, 1), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, out_channels, 3, 1, 1), nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.up(self.fc(z).view(z.size(0), 256, *self._hw))


class CascadeDecoder(nn.Module):
    """Multi-scale cascade decoder (Laplacian-pyramid style).

    Reconstructs at 3 scales (quarter, half, full). Each coarser
    reconstruction is upsampled and used as a residual base for the
    next finer scale, encouraging the model to capture anomalies at
    multiple spatial frequencies simultaneously.

    forward() returns (full_res, half_res, quarter_res).
    """
    def __init__(self, out_channels: int = 12, latent_dim: int = 64,
                 img_size: int = 16):
        super().__init__()
        _m = nn.Sequential(nn.Conv2d(out_channels, 64, 4, 2, 1),
                           nn.Conv2d(64, 128, 4, 2, 1),
                           nn.Conv2d(128, 256, 4, 2, 1))
        with torch.no_grad():
            self._hw = _m(torch.zeros(1, out_channels, img_size, img_size)).shape[-2:]
        self.fc = nn.Linear(latent_dim, 256 * self._hw[0] * self._hw[1])
        UP = lambda: nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Stage 1: latent → quarter-res features + reconstruction
        self.s1_feat = nn.Sequential(UP(), nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True))
        self.s1_out  = nn.Conv2d(128, out_channels, 1)          # (B,C,H/4,W/4)

        # Stage 2: quarter → half-res (residual on upsampled s1_out)
        self.s2_feat = nn.Sequential(UP(), nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True))
        self.s2_out  = nn.Conv2d(64, out_channels, 1)            # (B,C,H/2,W/2)

        # Stage 3: half → full-res (residual on upsampled s2_out, + Tanh)
        self.s3_feat = nn.Sequential(UP(), nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(True))
        self.s3_out  = nn.Conv2d(32, out_channels, 1)            # (B,C,H,W)

    @staticmethod
    def _up2(t: torch.Tensor) -> torch.Tensor:
        return F.interpolate(t, scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, z: torch.Tensor):
        h    = self.fc(z).view(z.size(0), 256, *self._hw)
        f1   = self.s1_feat(h)
        out1 = self.s1_out(f1)                              # quarter-res
        f2   = self.s2_feat(f1)
        out2 = self.s2_out(f2) + self._up2(out1)           # half-res residual
        f3   = self.s3_feat(f2)
        out3 = torch.tanh(self.s3_out(f3) + self._up2(out2))  # full-res residual
        return out3, out2, out1


class Discriminator(nn.Module):
    """Discriminator with Spectral Normalization + InstanceNorm.

    SpectralNorm constrains D's Lipschitz constant, preventing it from
    dominating G. Without this, D saturates → vanishing gradient for G
    → model degrades with more training.
    """
    def __init__(self, in_channels: int = 12, img_size: int = 16):
        super().__init__()
        SN = nn.utils.spectral_norm
        IN = lambda c: nn.InstanceNorm2d(c, affine=True)
        self.b1 = nn.Sequential(SN(nn.Conv2d(in_channels, 64,  4, 2, 1)), nn.LeakyReLU(0.2, True))
        self.b2 = nn.Sequential(SN(nn.Conv2d(64,  128, 4, 2, 1)), IN(128), nn.LeakyReLU(0.2, True))
        self.b3 = nn.Sequential(SN(nn.Conv2d(128, 256, 4, 2, 1)), IN(256), nn.LeakyReLU(0.2, True))
        with torch.no_grad():
            h_hw = self.b3(self.b2(self.b1(torch.zeros(1, in_channels, img_size, img_size)))).shape[-1]
        self.final = SN(nn.Conv2d(256, 1, h_hw, 1, 0))

    def forward(self, x: torch.Tensor):
        f1 = self.b1(x); f2 = self.b2(f1); f3 = self.b3(f2)
        return self.final(f3).view(x.size(0), 1), (f1, f2, f3)


# ─────────────────────────────────────────────────────────────
# Diffusion UNet (conditioned on latent z and timestep t)
# ─────────────────────────────────────────────────────────────

class DiffUNet(nn.Module):
    """Lightweight UNet for DDPM, conditioned on z and time embedding."""
    def __init__(self, in_channels: int = 12, latent_dim: int = 64,
                 cond_dim: int = 64):
        super().__init__()
        # Time + latent → combined conditioning vector
        self.t_emb  = nn.Sequential(nn.Linear(1, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.z_proj = nn.Linear(latent_dim, cond_dim)

        GN = lambda g, c: nn.GroupNorm(g, c)
        # Down
        self.d1 = nn.Sequential(nn.Conv2d(in_channels, 32, 3, 1, 1), GN(4, 32), nn.SiLU())
        self.d2 = nn.Sequential(nn.Conv2d(32, 64, 4, 2, 1),           GN(8, 64), nn.SiLU())  # →8
        self.d3 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1),          GN(8,128), nn.SiLU())  # →4
        # Conditioning projections (additive spatial bias)
        self.c3 = nn.Linear(cond_dim, 128)
        self.c2 = nn.Linear(cond_dim, 64)
        self.c1 = nn.Linear(cond_dim, 32)
        # Up with skip connections
        self.u3 = nn.Sequential(nn.Conv2d(128+128, 64, 3, 1, 1), GN(8, 64), nn.SiLU())
        self.u2 = nn.Sequential(nn.Conv2d(64+64,   32, 3, 1, 1), GN(4, 32), nn.SiLU())
        self.u1 = nn.Sequential(nn.Conv2d(32+32,   32, 3, 1, 1), GN(4, 32), nn.SiLU())
        self.out = nn.Conv2d(32, in_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, z: torch.Tensor):
        cond = self.t_emb(t.float().unsqueeze(-1) / 100.0) + self.z_proj(z)  # (B, cond_dim)
        bias = lambda proj: proj(cond).view(cond.size(0), -1, 1, 1)
        d1 = self.d1(x)
        d2 = self.d2(d1) + bias(self.c2)
        d3 = self.d3(d2) + bias(self.c3)
        u3 = F.interpolate(self.u3(torch.cat([d3, d3], 1)), scale_factor=2, mode='bilinear', align_corners=False)
        u2 = F.interpolate(self.u2(torch.cat([u3, d2], 1)), scale_factor=2, mode='bilinear', align_corners=False)
        u1 = self.u1(torch.cat([u2, d1], 1))
        return self.out(u1)


# ─────────────────────────────────────────────────────────────
# Model classes
# ─────────────────────────────────────────────────────────────

class VAEModel(nn.Module):
    """Standard VAE. Anomaly score = per-pixel L1 reconstruction error."""

    def __init__(self, in_channels: int = 12, latent_dim: int = 64,
                 img_size: int = 16):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim, img_size)
        self.decoder = Decoder(in_channels, latent_dim, img_size)

    def encode(self, x):
        mu, logvar = self.encoder(x)
        z = Encoder.reparameterise(mu, logvar)
        return z, mu, logvar

    def reconstruct(self, x):
        z, _, _ = self.encode(x)
        return self.decoder(z)

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        return self.decoder(z), mu, logvar

    @torch.no_grad()
    def score_pixel(self, x):
        """Return (B, H, W) per-pixel anomaly score."""
        self.eval()
        x_hat = self.reconstruct(x)
        return torch.mean(torch.abs(x - x_hat), dim=1)


# ─────────────────────────────────────────────────────────────
# Convolutional Autoencoder (AE)
# ─────────────────────────────────────────────────────────────

class AEModel(nn.Module):
    """Deterministic convolutional autoencoder.

    No reparameterisation or KL term: the encoder output is used
    directly as the latent code (mu only, logvar ignored).
    Anomaly score = per-pixel L1 reconstruction error.

    Geochemical rationale: normal background geochemistry lives on a
    low-dimensional manifold; the AE compresses it to that manifold
    and reconstructs it faithfully.  Anomalous samples (mineralised
    zones) lie off-manifold → higher reconstruction error.
    """

    def __init__(self, in_channels: int = 12, latent_dim: int = 64,
                 img_size: int = 16):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim, img_size)
        self.decoder = Decoder(in_channels, latent_dim, img_size)

    def encode(self, x):
        mu, _ = self.encoder(x)   # discard logvar; use mu as code
        return mu, None, None

    def reconstruct(self, x):
        z, _, _ = self.encode(x)
        return self.decoder(z)

    def forward(self, x):
        z, _, _ = self.encode(x)
        return self.decoder(z)

    @torch.no_grad()
    def score_pixel(self, x):
        """Return (B, H, W) per-pixel anomaly score."""
        self.eval()
        x_hat = self.reconstruct(x)
        return torch.mean(torch.abs(x - x_hat), dim=1)


# ─────────────────────────────────────────────────────────────
# Denoising Autoencoder (DAE)
# ─────────────────────────────────────────────────────────────

class DAEModel(nn.Module):
    """Denoising convolutional autoencoder.

    Training: Gaussian noise is added to the input; the model learns
    to reconstruct the *clean* version.  This forces the encoder to
    capture the underlying geochemical structure rather than copying
    pixel values, making the learned manifold more robust.

    Inference / scoring: clean input is passed through (no noise),
    and the L1 reconstruction error is used as the anomaly score.
    Anomalous patches that lie off the normal manifold cannot be
    reconstructed cleanly even without noise → high error.

    Args:
        noise_std: standard deviation of additive Gaussian noise
                   applied during training (default 0.2; data is
                   normalised to [-1, 1] so 0.2 ≈ 10 % of range).
    """

    def __init__(self, in_channels: int = 12, latent_dim: int = 64,
                 img_size: int = 16, noise_std: float = 0.2):
        super().__init__()
        self.encoder   = Encoder(in_channels, latent_dim, img_size)
        self.decoder   = Decoder(in_channels, latent_dim, img_size)
        self.noise_std = noise_std

    def encode(self, x):
        mu, _ = self.encoder(x)
        return mu, None, None

    def reconstruct(self, x):
        """Encode clean x and decode (used at inference)."""
        z, _, _ = self.encode(x)
        return self.decoder(z)

    def forward(self, x):
        """During training: corrupt x then reconstruct clean x.
        During eval:        pass x through unchanged (for scoring).
        """
        if self.training and self.noise_std > 0:
            x_in = x + torch.randn_like(x) * self.noise_std
        else:
            x_in = x
        z, _, _ = self.encode(x_in)
        return self.decoder(z)

    @torch.no_grad()
    def score_pixel(self, x):
        """Return (B, H, W) per-pixel anomaly score (no noise at inference)."""
        self.eval()          # ensures forward() skips noise addition
        x_hat = self.forward(x)
        return torch.mean(torch.abs(x - x_hat), dim=1)


class VAEGANModel(nn.Module):
    """VAE + GAN discriminator.

    Anomaly score = α·recon + β·(1 − p_real) from D.
    A real-looking reconstruction gets a low score; anomalous
    (unreconstructable) inputs look fake to D → high score.
    """

    def __init__(self, in_channels: int = 12, latent_dim: int = 64,
                 img_size: int = 16):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim, img_size)
        self.decoder = Decoder(in_channels, latent_dim, img_size)
        self.disc    = Discriminator(in_channels, img_size)

    def encode(self, x):
        mu, logvar = self.encoder(x)
        z = Encoder.reparameterise(mu, logvar)
        return z, mu, logvar

    def reconstruct(self, x):
        z, _, _ = self.encode(x)
        return self.decoder(z)

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        return self.decoder(z), mu, logvar

    @torch.no_grad()
    def score_pixel(self, x, alpha_recon: float = 1.0, alpha_disc: float = 0.5,
                    alpha_fm: float = 0.5):
        """Return (B, H, W) per-pixel anomaly score."""
        self.eval()
        z, mu, logvar = self.encode(x)
        x_hat = self.decoder(z)

        recon = torch.mean(torch.abs(x - x_hat), dim=1)           # (B,H,W)

        return alpha_recon * recon


class VAECascadeGANModel(nn.Module):
    """VAE + multi-scale cascade decoder + GAN.

    Anomaly score = weighted sum of per-scale reconstruction errors.
    The cascade ensures sensitivity to anomalies at different spatial
    frequencies (local element concentrations vs. broad patterns).
    """

    def __init__(self, in_channels: int = 12, latent_dim: int = 64,
                 img_size: int = 16):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim, img_size)
        self.decoder = CascadeDecoder(in_channels, latent_dim, img_size)
        self.disc    = Discriminator(in_channels, img_size)
        self._C = in_channels

    def encode(self, x):
        mu, logvar = self.encoder(x)
        z = Encoder.reparameterise(mu, logvar)
        return z, mu, logvar

    def reconstruct(self, x):
        z, _, _ = self.encode(x)
        out3, _, _ = self.decoder(z)
        return out3

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        outs = self.decoder(z)          # (full, half, quarter)
        return outs, mu, logvar

    @torch.no_grad()
    def score_pixel(self, x, w_full: float = 1.0, w_half: float = 0.5,
                    w_quar: float = 0.25):
        """Return (B, H, W) multi-scale anomaly score."""
        self.eval()
        z, _, _ = self.encode(x)
        out3, out2, out1 = self.decoder(z)

        x_half = F.avg_pool2d(x, 2)
        x_quar = F.avg_pool2d(x, 4)

        err3 = torch.mean(torch.abs(x      - out3), dim=1)
        err2 = torch.mean(torch.abs(x_half - out2), dim=1)
        err1 = torch.mean(torch.abs(x_quar - out1), dim=1)

        up = lambda t, s: F.interpolate(t.unsqueeze(1), scale_factor=s,
                                         mode='bilinear', align_corners=False).squeeze(1)
        return w_full * err3 + w_half * up(err2, 2) + w_quar * up(err1, 4)


class VAEDiffModel(nn.Module):
    """VAE encoder + DDPM decoder (T=100 steps).

    Training: standard DDPM noise-prediction loss + KL.
    Scoring: partial noising (t_eval steps) then full reverse → L1 error.
    Anomalous inputs cannot be faithfully denoised → high score.
    """
    T = 100

    def __init__(self, in_channels: int = 12, latent_dim: int = 64,
                 img_size: int = 16, t_eval: int = 20):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim, img_size)
        self.unet    = DiffUNet(in_channels, latent_dim)
        self.t_eval  = t_eval

        betas = torch.linspace(1e-4, 0.02, self.T)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, 0)
        self.register_buffer('betas',     betas)
        self.register_buffer('alphas',    alphas)
        self.register_buffer('alpha_bar', alpha_bar)

    def encode(self, x):
        mu, logvar = self.encoder(x)
        z = Encoder.reparameterise(mu, logvar)
        return z, mu, logvar

    def reconstruct(self, x):
        """Partial forward noise + full reverse denoising."""
        z, _, _ = self.encode(x)
        t_val = self.t_eval
        sq_ab  = self.alpha_bar[t_val].sqrt()
        sq_1ab = (1 - self.alpha_bar[t_val]).sqrt()
        noise  = torch.randn_like(x)
        x_t = sq_ab * x + sq_1ab * noise
        for t in reversed(range(t_val)):
            t_tensor = torch.full((x.size(0),), t, device=x.device, dtype=torch.long)
            pred_noise = self.unet(x_t, t_tensor, z)
            beta_t     = self.betas[t].view(1, 1, 1, 1)
            alpha_t    = self.alphas[t].view(1, 1, 1, 1)
            ab_t       = self.alpha_bar[t].view(1, 1, 1, 1)
            mean = (1.0 / alpha_t.sqrt()) * (x_t - beta_t / (1 - ab_t).sqrt() * pred_noise)
            if t > 0:
                x_t = mean + beta_t.sqrt() * torch.randn_like(x_t)
            else:
                x_t = mean
        return x_t

    def forward(self, x):
        """DDPM forward pass; returns (diffusion_loss, kl_loss)."""
        mu, logvar = self.encoder(x)
        z = Encoder.reparameterise(mu, logvar)
        B = x.size(0)
        t = torch.randint(0, self.T, (B,), device=x.device)
        noise = torch.randn_like(x)
        sq_ab  = self.alpha_bar[t].view(B, 1, 1, 1).sqrt()
        sq_1ab = (1 - self.alpha_bar[t].view(B, 1, 1, 1)).sqrt()
        x_noisy   = sq_ab * x + sq_1ab * noise
        pred_noise = self.unet(x_noisy, t, z)
        diff_loss = F.mse_loss(pred_noise, noise)
        kl = 0.5 * torch.mean(torch.sum(logvar.exp() + mu ** 2 - 1.0 - logvar, dim=1))
        return diff_loss, kl

    @torch.no_grad()
    def score_pixel(self, x):
        """Return (B, H, W) per-pixel anomaly score."""
        self.eval()
        x_hat = self.reconstruct(x)
        return torch.mean(torch.abs(x - x_hat), dim=1)
