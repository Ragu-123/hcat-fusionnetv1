# save as example-algorithm/preprocess/hcat_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# =================================================================================
# ADVANCED IMPUTATION MODULES (Copied from train.py)
# These are required to correctly load the trained model weights.
# =================================================================================

class MultiModalVAE(nn.Module):
    """
    JAMIE-inspired Joint Variational Autoencoder for cross-modal imputation.
    This version is adapted from your training script to handle single-patient inference.
    """
    def __init__(self, n_modalities=5, d_model=512, latent_dim=64, n_scales=3):
        super().__init__()
        self.n_mod = n_modalities
        self.d = d_model
        self.latent_dim = latent_dim
        self.n_scales = n_scales

        self.modality_encoders = nn.ModuleList([
            nn.ModuleDict({
                f'scale_{s}': nn.Sequential(
                    nn.Linear(d_model, d_model // (2**s)),
                    nn.BatchNorm1d(d_model // (2**s)),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(d_model // (2**s), latent_dim // (2**s))
                ) for s in range(n_scales)
            }) for _ in range(n_modalities)
        ])

        self.total_scale_dim = sum(latent_dim // (2**s) for s in range(n_scales))

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.total_scale_dim, num_heads=8, batch_first=True, dropout=0.1
        )

        self.joint_mu = nn.Sequential(
            nn.Linear(self.total_scale_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.joint_logvar = nn.Sequential(
            nn.Linear(self.total_scale_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )

        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, d_model * 2),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(d_model * 2, d_model)
            ) for _ in range(n_modalities)
        ])

        self.uncertainty_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ) for _ in range(n_modalities)
        ])

        self.quality_transform = nn.Sequential(
            nn.Linear(n_modalities, 64),
            nn.GELU(),
            nn.Linear(64, n_modalities),
            nn.Softmax(dim=-1)
        )

    def encode_modality(self, x, modality_idx, mask):
        num_present = mask.sum()
        if num_present == 0:
            return torch.zeros(x.size(0), self.total_scale_dim, device=x.device)

        present_x = x[mask]
        scales = []
        for s in range(self.n_scales):
            encoder_scale_s = self.modality_encoders[modality_idx][f'scale_{s}']
            
            # CRITICAL FOR INFERENCE: Handle BatchNorm for a single sample (batch size = 1)
            if num_present > 1:
                scale_out = encoder_scale_s(present_x)
            else:
                original_mode = encoder_scale_s[1].training
                encoder_scale_s[1].eval()
                with torch.no_grad():
                    scale_out = encoder_scale_s(present_x)
                encoder_scale_s[1].train(original_mode)

            padded = torch.zeros(x.size(0), scale_out.size(-1), device=x.device)
            padded[mask] = scale_out
            scales.append(padded)
        
        return torch.cat(scales, dim=-1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, emb, present_mask, quality):
        B, M, D = emb.shape
        device = emb.device
        quality_weights = self.quality_transform(quality)

        encoded_modalities = []
        for m in range(M):
            mask = present_mask[:, m].bool()
            encoded = self.encode_modality(emb[:, m], m, mask)
            encoded = encoded * quality_weights[:, m:m+1]
            encoded_modalities.append(encoded)

        encoded_stack = torch.stack(encoded_modalities, dim=1)
        attn_mask = (present_mask == 0)
        
        attended, _ = self.cross_attn(
            encoded_stack, encoded_stack, encoded_stack,
            key_padding_mask=attn_mask
        )
        
        combined = encoded_stack + 0.5 * attended
        available_weights = present_mask.float().unsqueeze(-1)
        weighted_combined = combined * available_weights
        n_available = present_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
        joint_repr = weighted_combined.sum(dim=1) / n_available
        
        joint_mu = self.joint_mu(joint_repr)
        joint_logvar = self.joint_logvar(joint_repr)
        z_joint = self.reparameterize(joint_mu, joint_logvar)
        
        imputed_emb = emb.clone()
        uncertainties = torch.zeros(B, M, device=device)
        
        for m in range(M):
            decoded = self.decoders[m](z_joint)
            uncertainty = self.uncertainty_heads[m](z_joint).squeeze(-1)
            missing_mask = (present_mask[:, m] == 0)
            if missing_mask.any():
                imputed_emb[missing_mask, m] = decoded[missing_mask]
                uncertainties[missing_mask, m] = uncertainty[missing_mask]

        # Return dummy losses for inference, as they aren't used
        kl_loss = torch.tensor(0.0, device=device)
        recon_loss = torch.tensor(0.0, device=device)
        
        return imputed_emb, uncertainties, kl_loss, recon_loss


class AdvancedIterativeImputer(nn.Module):
    def __init__(self, n_modalities=5, d_model=512, n_iterations=3, n_heads=8):
        super().__init__()
        self.n_iter = n_iterations
        self.vae_imputer = MultiModalVAE(n_modalities, d_model)
        self.refinement_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2,
                dropout=0.1, activation='gelu', batch_first=True
            ) for _ in range(n_iterations)
        ])
        self.confidence_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 128),
                nn.GELU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            ) for _ in range(n_iterations)
        ])
        self.uncertainty_reduction = nn.Parameter(torch.linspace(1.0, 0.1, n_iterations))

    def forward(self, emb, present_mask, quality):
        current_emb, base_uncertainty, kl_loss, recon_loss = self.vae_imputer(emb, present_mask, quality)
        final_uncertainties = base_uncertainty.clone()
        
        for iter_idx in range(self.n_iter):
            refined = self.refinement_layers[iter_idx](current_emb)
            missing_mask = (present_mask == 0).unsqueeze(-1)
            current_emb = torch.where(missing_mask, refined, current_emb)
            
            confidences = []
            for m in range(emb.shape[1]):
                conf = self.confidence_heads[iter_idx](current_emb[:, m])
                confidences.append(conf.squeeze(-1))
            
            iter_confidences = torch.stack(confidences, dim=1)
            reduction_factor = self.uncertainty_reduction[iter_idx]
            final_uncertainties = final_uncertainties * reduction_factor + (1 - iter_confidences) * (1 - reduction_factor)
        
        return current_emb, final_uncertainties, kl_loss, recon_loss

# =================================================================================
# MAIN HCAT MODEL (Helper classes + Main class)
# =================================================================================

class ModalityPositionalEncoding(nn.Module):
    def __init__(self, n_modalities=5, d_model=512):
        super().__init__()
        self.mod_pos = nn.Parameter(torch.randn(n_modalities, d_model) * 0.02)
    def forward(self, x): return x + self.mod_pos.unsqueeze(0).to(x.device)

class QualityGate(nn.Module):
    def __init__(self, n_mod=5, hidden=64): # Corrected hidden dimension
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_mod, hidden), nn.ReLU(), nn.Linear(hidden, n_mod), nn.Sigmoid())
    def forward(self, quality): return self.net(quality)

class AttentionPool(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.q = nn.Linear(d_model, 128)
        self.v = nn.Linear(d_model, d_model)
    def forward(self, x, mask=None):
        scores = self.q(x).sum(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        w = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (w * self.v(x)).sum(dim=1)

class EnhancedHCAT(nn.Module):
    def __init__(self, d_model=512, n_modalities=5, n_heads=8, n_global_layers=3, dropout=0.2,
                 use_advanced_imputation=True, n_impute_iterations=3):
        super().__init__()
        self.use_advanced_imputation = use_advanced_imputation

        if use_advanced_imputation:
            self.imputer = AdvancedIterativeImputer(
                n_modalities=n_modalities,
                d_model=d_model,
                n_iterations=n_impute_iterations
            )
        else:
            self.imputer = None

        self.pos = ModalityPositionalEncoding(n_modalities, d_model)
        self.quality_gate = QualityGate(n_mod=n_modalities, hidden=64)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead=n_heads, dim_feedforward=2048,
            batch_first=True, dropout=dropout, activation="gelu",
            norm_first=True
        )

        self.local_temporal = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.local_spatial = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.semantic_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.global_enc = nn.TransformerEncoder(enc_layer, num_layers=n_global_layers)
        self.pool = AttentionPool(d_model)
        self.cross_enc = nn.TransformerEncoder(enc_layer, num_layers=2)

        self.route_surv = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 3)
        )
        self.route_rec = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 3)
        )

        self.head_surv = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        self.head_rec = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        self.contrast_proj = nn.Linear(d_model, 128)
        self.uncertainty_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, emb, quality, present_mask):
        # INFERENCE FORWARD PASS: Impute if needed, then predict.
        if self.imputer is not None:
            # `torch.no_grad()` is used in inference.py, so we don't need it here.
            imputed_emb, imp_uncertainty, _, _ = self.imputer(emb, present_mask, quality)
            missing_mask = (present_mask == 0).unsqueeze(-1)
            emb = torch.where(missing_mask, imputed_emb, emb)
            quality = torch.where(
                present_mask == 0,
                quality * (1.0 - imp_uncertainty * self.uncertainty_weight),
                quality
            )

        x = self.pos(emb)
        qg = self.quality_gate(quality) * present_mask.float()
        x = x * qg.unsqueeze(-1)

        # The order is: [clinical, temporal, pathological, semantic, spatial]
        temporal_tokens = x[:, [0, 1], :]
        spatial_tokens = x[:, [2, 4], :]
        semantic_token = x[:, [3], :]

        temporal_local = self.local_temporal(temporal_tokens)
        spatial_local = self.local_spatial(spatial_tokens)
        semantic_local = self.semantic_proj(semantic_token.squeeze(1)).unsqueeze(1)

        concat = torch.cat([temporal_local, spatial_local, semantic_local], dim=1)
        concat = self.global_enc(concat)

        t_out, s_out, sem_out = concat[:, :2, :], concat[:, 2:4, :], concat[:, 4:, :]
        t_vec, s_vec, sem_vec = self.pool(t_out), self.pool(s_out), self.pool(sem_out)

        branches = torch.stack([t_vec, s_vec, sem_vec], dim=1)
        fused = self.cross_enc(branches)
        pooled = fused.mean(dim=1)

        w_surv = torch.softmax(self.route_surv(pooled), dim=-1)
        w_rec = torch.softmax(self.route_rec(pooled), dim=-1)
        rep_surv = (w_surv.unsqueeze(-1) * fused).sum(dim=1)
        rep_rec = (w_rec.unsqueeze(-1) * fused).sum(dim=1)

        logit_surv = self.head_surv(rep_surv).squeeze(-1)
        logit_rec = self.head_rec(rep_rec).squeeze(-1)

        return {"logit_surv": logit_surv, "logit_rec": logit_rec}
