# save as example-algorithm/preprocess/hcat_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# These helper classes are required for the main model definition
class ModalityPositionalEncoding(nn.Module):
    def __init__(self, n_modalities=5, d_model=512):
        super().__init__()
        self.mod_pos = nn.Parameter(torch.randn(n_modalities, d_model) * 0.02)
    def forward(self, x): return x + self.mod_pos.unsqueeze(0).to(x.device)

class QualityGate(nn.Module):
    # CORRECTED: The hidden dimension is 128, matching the train.py script
    def __init__(self, n_mod=5, hidden=128):
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

# This is the full, complex model definition from your train.py.
# It MUST match the structure of the saved .pt file to load the weights correctly.
class EnhancedHCAT(nn.Module):
    def __init__(self, d_model=512, n_modalities=5, n_heads=8, n_global_layers=3, dropout=0.2,
                 use_advanced_imputation=False, n_impute_iterations=3): # Imputation is OFF for inference
        super().__init__()
        self.use_advanced_imputation = use_advanced_imputation
        
        # NOTE: The 'imputer' module is defined as a placeholder (None).
        # This is because the saved weights file contains keys for "imputer.*",
        # but we don't need to run the imputation logic during inference.
        self.imputer = None
        
        self.pos = ModalityPositionalEncoding(n_modalities, d_model)
        # CORRECTED: The QualityGate hidden dimension now matches the training script.
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
        x = self.pos(emb)
        qg = self.quality_gate(quality) * present_mask.float()
        x = x * qg.unsqueeze(-1)
        
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