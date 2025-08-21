# #!/usr/bin/env python3
# r"""
# Enhanced HCAT training with Advanced Multi-Modal Imputation Techniques:
# - JAMIE-style Joint Variational Autoencoders for cross-modal imputation
# - Multi-scale VAE with uncertainty quantification
# - Attention-based Cross-Modal Imputation
# - Iterative Self-Supervised Refinement

# Author: Enhanced for Journal Publication
# """
import os
import re
import json
import h5py
import math
import random
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# sklearn metrics
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ----------------------------
# Utilities for HDF5 handling (unchanged)
# ----------------------------
def normalize_pid(x):
    if x is None:
        return ""
    if isinstance(x, bytes):
        try:
            s = x.decode("utf-8")
        except Exception:
            s = str(x)
    else:
        s = str(x)
    s = s.strip()
    if s.isdigit():
        return str(int(s))
    m = re.search(r"\d+", s)
    if m:
        return str(int(m.group(0)))
    return s

def read_all_datasets_from_h5(path):
    out = {}
    with h5py.File(str(path), "r") as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                try:
                    out[name] = obj[()]
                except Exception:
                    out[name] = None
        f.visititems(visitor)
    return out

def find_patientid_and_embedding(h5dict):
    keys = list(h5dict.keys())
    lower_map = {k.lower(): k for k in keys}
    pid_key = None
    for candidate in ["patient_id", "patient_index_map", "patient_ids", "patientid", "patient"]:
        if candidate in lower_map:
            pid_key = lower_map[candidate]; break
    if pid_key is None:
        for k in keys:
            if k.lower().endswith("/patient_id") or k.lower().endswith("/patient_index_map"):
                pid_key = k; break
    patient_ids = None
    if pid_key is not None:
        arr = h5dict.get(pid_key)
        if arr is not None:
            if isinstance(arr, np.ndarray) and arr.dtype.type is np.bytes_:
                patient_ids = [normalize_pid(x) for x in arr.tolist()]
            else:
                patient_ids = [normalize_pid(x) for x in arr.tolist()]

    emb_key = None
    emb_candidates = ["embedding_512", "embedding_mean_512", "text_combined_embedding_512", "features_mean", "embedding", "embeddings", "features", "feature"]
    for cand in emb_candidates:
        if cand in lower_map:
            emb_key = lower_map[cand]; break
    if emb_key is None:
        if patient_ids is not None:
            n = len(patient_ids)
            for k in keys:
                arr = h5dict.get(k)
                if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] == n:
                    emb_key = k; break
        else:
            best = None; best_size = 0
            for k in keys:
                arr = h5dict.get(k)
                if isinstance(arr, np.ndarray) and arr.ndim == 2:
                    size = arr.size
                    if size > best_size:
                        best_size = size; best = k
            emb_key = best

    q_key = None
    for candidate in ["quality_score", "quality", "modality_quality", "quality_scores"]:
        if candidate in lower_map:
            q_key = lower_map[candidate]; break
    npatch_key = None
    for candidate in ["n_patches", "npatches", "n_patches_per_slide"]:
        if candidate in lower_map:
            npatch_key = lower_map[candidate]; break

    embedding = None
    quality = None
    n_patches = None

    if emb_key is not None:
        embedding = h5dict.get(emb_key)
    if q_key is not None:
        quality = h5dict.get(q_key)
    if npatch_key is not None:
        n_patches = h5dict.get(npatch_key)

    if patient_ids is None:
        candidate_parents = []
        for k in keys:
            parts = k.split("/")
            if len(parts) >= 2:
                parent = parts[0]
                candidate_parents.append(parent)
        candidate_parents = sorted(set(candidate_parents))
        if candidate_parents and embedding is None:
            collected = []
            pids = []
            for parent in candidate_parents:
                for suffix in ["features", "embedding", "embeddings", "features_mean", "embedding_mean"]:
                    kk = f"{parent}/{suffix}"
                    if kk in h5dict and isinstance(h5dict[kk], np.ndarray):
                        arr = h5dict[kk]
                        if arr.ndim == 1:
                            pass
                        else:
                            collected.append(arr)
                            pids.append(parent)
                            break
            if collected:
                embedding = np.stack(collected, axis=0)
                patient_ids = [normalize_pid(x) for x in pids]

    if embedding is None:
        return None, None, None, None, None
    if patient_ids is None:
        patient_ids = [str(i) for i in range(embedding.shape[0])]

    embedding = np.asarray(embedding, dtype=np.float32)
    if quality is not None:
        quality = np.asarray(quality, dtype=np.float32)
        if quality.shape == (): quality = np.full((embedding.shape[0],), float(quality), dtype=np.float32)
    if n_patches is not None:
        n_patches = np.asarray(n_patches, dtype=np.float32)
    return patient_ids, embedding, quality, n_patches, emb_key

def aggregate_spatial(spatial_path):
    h5dict = read_all_datasets_from_h5(spatial_path)
    pids, emb, quality, n_patches, emb_key = find_patientid_and_embedding(h5dict)
    if emb is None or pids is None:
        with h5py.File(str(spatial_path), "r") as f:
            if "embedding_mean_512" in f and "patient_id" in f:
                arr_emb = np.array(f["embedding_mean_512"])
                arr_pid = [normalize_pid(x.decode() if isinstance(x,(bytes,np.bytes_)) else x) for x in f["patient_id"][:]]
                return arr_pid, arr_emb, np.zeros_like(arr_emb), np.ones((len(arr_pid),), dtype=np.float32)
        raise RuntimeError("Could not parse spatial H5 file for embeddings.")

    normalized = [normalize_pid(x) for x in pids]
    groups = defaultdict(list)
    for i, pid in enumerate(normalized):
        groups[pid].append(i)

    pt_ids = []
    means = []
    vars_ = []
    quals = []
    for pid, idxs in groups.items():
        arrays = emb[np.array(idxs)]
        if n_patches is not None:
            w = n_patches[np.array(idxs)]
            w = np.maximum(w, 1.0)
            mean = np.average(arrays, axis=0, weights=w)
        else:
            mean = arrays.mean(axis=0)
        var = arrays.var(axis=0) if arrays.shape[0] > 1 else np.zeros_like(mean)
        q = float(np.mean(quality[np.array(idxs)]) if (quality is not None) else 1.0)
        pt_ids.append(pid)
        means.append(mean.astype(np.float32))
        vars_.append(var.astype(np.float32))
        quals.append(q)
    return pt_ids, np.stack(means, axis=0), np.stack(vars_, axis=0), np.array(quals, dtype=np.float32)

def build_patient_embedding_map(h5path, prefer_key_candidates=None):
    h5dict = read_all_datasets_from_h5(h5path)
    pids, emb, quality, n_patches, emb_key = find_patientid_and_embedding(h5dict)
    if emb is None or pids is None:
        with h5py.File(str(h5path), "r") as f:
            pid_arr = None
            for cand in ["patient_id","patient_index_map","patient_ids"]:
                if cand in f:
                    pid_arr = f[cand][:]
                    break
            if pid_arr is not None:
                pids = [normalize_pid(x.decode() if isinstance(x,(bytes,np.bytes_)) else x) for x in pid_arr]
            emb_arr = None
            for cand in ["embedding_512","embedding_mean_512","features_mean","embedding","features"]:
                if cand in f:
                    emb_arr = np.array(f[cand])
                    break
            if emb_arr is not None and pids is not None:
                emb = emb_arr
                if "quality_score" in f:
                    qtmp = np.array(f["quality_score"])
                    quality = qtmp

    if emb is None or pids is None:
        return {}

    emb = np.asarray(emb, dtype=np.float32)
    n_pat = emb.shape[0]

    if quality is None:
        quality_1d = np.ones((n_pat,), dtype=np.float32)
    else:
        quality_arr = np.asarray(quality)
        if quality_arr.ndim == 0:
            quality_1d = np.full((n_pat,), float(quality_arr), dtype=np.float32)
        elif quality_arr.ndim == 1 and quality_arr.shape[0] == n_pat:
            quality_1d = quality_arr.astype(np.float32)
        elif quality_arr.ndim == 1 and quality_arr.shape[0] != n_pat:
            if quality_arr.shape[0] < n_pat:
                pad = np.full((n_pat - quality_arr.shape[0],), float(np.mean(quality_arr)), dtype=np.float32)
                quality_1d = np.concatenate([quality_arr.astype(np.float32), pad], axis=0)
            else:
                quality_1d = quality_arr[:n_pat].astype(np.float32)
            print(f"[build_patient_embedding_map] warning: 1D quality length {quality_arr.shape[0]} != n_pat {n_pat}, coerced.")
        else:
            try:
                axes = tuple(range(1, quality_arr.ndim))
                reduced = np.nanmean(quality_arr, axis=axes)
                if reduced.shape[0] == n_pat:
                    quality_1d = reduced.astype(np.float32)
                else:
                    reduced_flat = reduced.ravel()
                    if reduced_flat.size >= n_pat:
                        quality_1d = reduced_flat[:n_pat].astype(np.float32)
                    else:
                        pad = np.full((n_pat - reduced_flat.size,), float(np.nanmean(reduced_flat)), dtype=np.float32)
                        quality_1d = np.concatenate([reduced_flat.astype(np.float32), pad], axis=0)
                    print(f"[build_patient_embedding_map] warning: reduced quality shape {reduced.shape} -> coerced to length {n_pat}")
            except Exception as e:
                print("[build_patient_embedding_map] warning: could not coerce quality array, using ones. Error:", e)
                quality_1d = np.ones((n_pat,), dtype=np.float32)

    if quality_1d.shape[0] != n_pat:
        if quality_1d.shape[0] > n_pat:
            quality_1d = quality_1d[:n_pat]
        else:
            pad = np.full((n_pat - quality_1d.shape[0],), float(np.nanmean(quality_1d)), dtype=np.float32)
            quality_1d = np.concatenate([quality_1d, pad], axis=0)
        print(f"[build_patient_embedding_map] final resize quality -> {quality_1d.shape[0]} items (n_pat={n_pat})")

    mapping = {}
    for i, pid in enumerate(pids[:n_pat]):
        q = float(quality_1d[i]) if (quality_1d is not None and i < quality_1d.shape[0]) else 1.0
        mapping[pid] = {"emb": emb[i].astype(np.float32), "quality": q}
    return mapping

# ----------------------------
# JAMIE-style Joint Variational Autoencoder for Multi-Modal Imputation
# ----------------------------
class MultiModalVAE(nn.Module):
    """
    JAMIE-inspired Joint Variational Autoencoder for cross-modal imputation
    with uncertainty quantification and multi-scale latent representations
    """
    def __init__(self, n_modalities=5, d_model=512, latent_dim=64, n_scales=3):
        super().__init__()
        self.n_mod = n_modalities
        self.d = d_model
        self.latent_dim = latent_dim
        self.n_scales = n_scales
        
        # Individual modality encoders (multi-scale)
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
        
        # Calculate total multi-scale dimension
        self.total_scale_dim = sum(latent_dim // (2**s) for s in range(n_scales))
        
        # Cross-modal attention mechanism - fixed dimension
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.total_scale_dim, num_heads=8, batch_first=True, dropout=0.1
        )
        
        # Joint latent space projectors
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
        
        # Modality-specific decoders with uncertainty
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, d_model * 2),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(d_model * 2, d_model)
            ) for _ in range(n_modalities)
        ])
        
        # Uncertainty estimators
        self.uncertainty_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ) for _ in range(n_modalities)
        ])
        
        # Quality-aware weighting
        self.quality_transform = nn.Sequential(
            nn.Linear(n_modalities, 64),
            nn.GELU(),
            nn.Linear(64, n_modalities),
            nn.Softmax(dim=-1)
        )

    def encode_modality(self, x, modality_idx, mask):
        """Encode single modality with multi-scale features"""
        num_present = mask.sum()
        if num_present == 0:
            return torch.zeros(x.size(0), self.total_scale_dim, device=x.device)

        present_x = x[mask]
        scales = []
        for s in range(self.n_scales):
            encoder_scale_s = self.modality_encoders[modality_idx][f'scale_{s}']
            
            # Handle BatchNorm edge case for single-sample sub-batches
            if num_present > 1:
                scale_out = encoder_scale_s(present_x)
            else:
                # Temporarily switch BatchNorm to eval mode to use running stats
                original_mode = encoder_scale_s[1].training
                encoder_scale_s[1].eval()
                with torch.no_grad():
                    scale_out = encoder_scale_s(present_x)
                # Revert BatchNorm to its original mode
                encoder_scale_s[1].train(original_mode)

            # Pad to match batch size
            padded = torch.zeros(x.size(0), scale_out.size(-1), device=x.device)
            padded[mask] = scale_out
            scales.append(padded)
        
        return torch.cat(scales, dim=-1)

    def reparameterize(self, mu, logvar):
        """VAE reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, emb, present_mask, quality):
        """
        Forward pass with joint latent space learning
        
        Args:
            emb: (B, M, D) embeddings
            present_mask: (B, M) presence indicators
            quality: (B, M) quality scores
            
        Returns:
            imputed_emb: (B, M, D) imputed embeddings
            uncertainties: (B, M) imputation uncertainties
            kl_loss: KL divergence loss
            recon_loss: Reconstruction loss
        """
        B, M, D = emb.shape
        device = emb.device
        
        # Quality-aware weighting
        quality_weights = self.quality_transform(quality)
        
        # Encode each modality with multi-scale features
        encoded_modalities = []
        for m in range(M):
            mask = present_mask[:, m].bool()
            # Note: The check for mask.any() is implicitly handled by encode_modality
            encoded = self.encode_modality(emb[:, m], m, mask)
            # Apply quality weighting
            encoded = encoded * quality_weights[:, m:m+1]
            encoded_modalities.append(encoded)

        encoded_stack = torch.stack(encoded_modalities, dim=1)  # (B, M, total_scale_dim)
        
        # Cross-modal attention with proper masking
        # Create attention mask for missing modalities
        attn_mask = (present_mask == 0)  # True for missing modalities
        
        try:
            attended, _ = self.cross_attn(
                encoded_stack, encoded_stack, encoded_stack,
                key_padding_mask=attn_mask
            )
        except Exception as e:
            # Fallback: use simple weighted average if attention fails
            print(f"Attention failed, using fallback: {e}")
            weights = present_mask.float().unsqueeze(-1)
            attended = encoded_stack * weights
        
        # Combine with original encoded features
        combined = encoded_stack + 0.5 * attended
        
        # Aggregate across modalities for joint representation
        # Use only available modalities for aggregation
        available_weights = present_mask.float().unsqueeze(-1)  # (B, M, 1)
        weighted_combined = combined * available_weights
        
        # Sum and normalize by number of available modalities
        n_available = present_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)  # (B, 1)
        joint_repr = weighted_combined.sum(dim=1) / n_available  # (B, total_scale_dim)
        
        # Joint latent space
        joint_mu = self.joint_mu(joint_repr)  # (B, latent_dim)
        joint_logvar = self.joint_logvar(joint_repr)  # (B, latent_dim)
        
        # Sample from joint latent space
        z_joint = self.reparameterize(joint_mu, joint_logvar)
        
        # Decode for each modality
        imputed_emb = emb.clone()
        uncertainties = torch.zeros(B, M, device=device)
        recon_losses = []
        
        for m in range(M):
            decoded = self.decoders[m](z_joint)
            uncertainty = self.uncertainty_heads[m](z_joint).squeeze(-1)
            
            # For missing modalities, use decoded output
            missing_mask = (present_mask[:, m] == 0)
            if missing_mask.any():
                imputed_emb[missing_mask, m] = decoded[missing_mask]
                uncertainties[missing_mask, m] = uncertainty[missing_mask]
            
            # Reconstruction loss for present modalities
            present_mask_m = (present_mask[:, m] == 1)
            if present_mask_m.any():
                recon_loss = F.mse_loss(decoded[present_mask_m], emb[present_mask_m, m])
                recon_losses.append(recon_loss)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + joint_logvar - joint_mu.pow(2) - joint_logvar.exp())
        kl_loss = kl_loss / B  # Normalize by batch size
        
        total_recon_loss = sum(recon_losses) / max(len(recon_losses), 1)
        
        return imputed_emb, uncertainties, kl_loss, total_recon_loss


class AdvancedIterativeImputer(nn.Module):
    """
    Advanced iterative imputation with self-attention and progressive refinement
    """
    def __init__(self, n_modalities=5, d_model=512, n_iterations=3, n_heads=8):
        super().__init__()
        self.n_mod = n_modalities
        self.d = d_model
        self.n_iter = n_iterations
        
        # Multi-modal VAE for initial imputation
        self.vae_imputer = MultiModalVAE(n_modalities, d_model)
        
        # Iterative refinement with self-attention
        self.refinement_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2,
                dropout=0.1, activation='gelu', batch_first=True
            ) for _ in range(n_iterations)
        ])
        
        # Confidence predictors for each iteration
        self.confidence_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 128),
                nn.GELU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            ) for _ in range(n_iterations)
        ])
        
        # Progressive uncertainty reduction
        self.uncertainty_reduction = nn.Parameter(torch.linspace(1.0, 0.1, n_iterations))

    def forward(self, emb, present_mask, quality):
        """
        Iterative imputation with progressive refinement
        """
        B, M, D = emb.shape
        device = emb.device
        
        # Initial VAE-based imputation
        current_emb, base_uncertainty, kl_loss, recon_loss = self.vae_imputer(emb, present_mask, quality)
        
        imputation_losses = [recon_loss]
        final_uncertainties = base_uncertainty.clone()
        
        # Iterative refinement
        for iter_idx in range(self.n_iter):
            # Self-attention refinement
            refined = self.refinement_layers[iter_idx](current_emb)
            
            # Update only missing modalities
            missing_mask = (present_mask == 0).unsqueeze(-1)
            current_emb = torch.where(missing_mask, refined, current_emb)
            
            # Update confidence/uncertainty
            confidences = []
            for m in range(M):
                conf = self.confidence_heads[iter_idx](current_emb[:, m])
                confidences.append(conf.squeeze(-1))
            
            iter_confidences = torch.stack(confidences, dim=1)  # (B, M)
            
            # Progressive uncertainty reduction
            reduction_factor = self.uncertainty_reduction[iter_idx]
            final_uncertainties = final_uncertainties * reduction_factor + (1 - iter_confidences) * (1 - reduction_factor)
            
            # Self-supervised loss: predict present modalities from refined features
            iter_loss = torch.tensor(0.0, device=device)
            for m in range(M):
                present_m = (present_mask[:, m] == 1)
                if present_m.any():
                    pred_loss = F.mse_loss(current_emb[present_m, m], emb[present_m, m])
                    iter_loss = iter_loss + pred_loss
            
            imputation_losses.append(iter_loss / M)
        
        total_imputation_loss = sum(imputation_losses) / len(imputation_losses)
        
        return current_emb, final_uncertainties, kl_loss, total_imputation_loss


# ----------------------------
# Dataset building & alignment (unchanged)
# ----------------------------
class MultiModalDataset(Dataset):
    def __init__(self, clinical_h5, semantic_h5, temporal_h5, pathological_h5, spatial_h5):
        print("Loading clinical (authoritative) ...", clinical_h5)
        clin_map = build_patient_embedding_map(clinical_h5)
        print("Loading semantic ...", semantic_h5)
        sem_map = build_patient_embedding_map(semantic_h5)
        print("Loading temporal ...", temporal_h5)
        temp_map = build_patient_embedding_map(temporal_h5)
        print("Loading pathological ...", pathological_h5)
        path_map = build_patient_embedding_map(pathological_h5)
        print("Aggregating spatial ...", spatial_h5)
        spat_pids, spat_mean, spat_var, spat_q = aggregate_spatial(spatial_h5)
        spat_map = {p: {"emb": spat_mean[i], "quality": float(spat_q[i])} for i,p in enumerate(spat_pids)}

        with h5py.File(str(clinical_h5), "r") as f:
            raw_pids = None
            for cand in ["patient_id","patient_index_map","patient_ids"]:
                if cand in f:
                    raw_pids = f[cand][:]
                    break
            if raw_pids is None:
                raise RuntimeError("Clinical file must contain patient_id dataset.")
            clin_pids = [normalize_pid(x.decode() if isinstance(x,(bytes,np.bytes_)) else x) for x in raw_pids]
            if "surv_5yr_label" in f:
                surv = np.array(f["surv_5yr_label"])
            else:
                raise RuntimeError("Clinical H5 missing surv_5yr_label")
            if "rec_2yr_label" in f:
                rec = np.array(f["rec_2yr_label"])
            else:
                raise RuntimeError("Clinical H5 missing rec_2yr_label")

        self.patients = []
        for i, pid in enumerate(clin_pids):
            entry = {"patient_id": pid}
            def fetch(m):
                if pid in m:
                    return m[pid]["emb"], m[pid].get("quality", 1.0), 1
                else:
                    return np.zeros((512,), dtype=np.float32), 0.0, 0
            entry["clinical_emb"], entry["clinical_q"], entry["clinical_present"] = fetch(clin_map)
            entry["semantic_emb"], entry["semantic_q"], entry["semantic_present"] = fetch(sem_map)
            entry["temporal_emb"], entry["temporal_q"], entry["temporal_present"] = fetch(temp_map)
            entry["path_emb"], entry["path_q"], entry["path_present"] = fetch(path_map)
            entry["spatial_emb"], entry["spatial_q"], entry["spatial_present"] = fetch(spat_map)
            entry["surv_label"] = int(surv[i])
            entry["rec_label"] = int(rec[i])
            self.patients.append(entry)
        print(f"Built dataset for {len(self.patients)} clinical patients.")

    def __len__(self): return len(self.patients)

    def __getitem__(self, idx):
        e = self.patients[idx]
        emb_stack = np.stack([
            e["clinical_emb"],
            e["temporal_emb"],
            e["path_emb"],
            e["semantic_emb"],
            e["spatial_emb"]
        ], axis=0).astype(np.float32)  # (5,512)
        quality = np.array([e["clinical_q"], e["temporal_q"], e["path_q"], e["semantic_q"], e["spatial_q"]], dtype=np.float32)
        present = np.array([e["clinical_present"], e["temporal_present"], e["path_present"], e["semantic_present"], e["spatial_present"]], dtype=np.int8)
        targets = np.array([e["surv_label"], e["rec_label"]], dtype=np.int8)
        return {"patient_id": e["patient_id"], "emb": emb_stack, "quality": quality, "present": present, "targets": targets}

# ----------------------------
# Enhanced HCAT with Advanced Imputation
# ----------------------------
class ModalityPositionalEncoding(nn.Module):
    def __init__(self, n_modalities=5, d_model=512):
        super().__init__()
        self.mod_pos = nn.Parameter(torch.randn(n_modalities, d_model) * 0.02)
    def forward(self, x): return x + self.mod_pos.unsqueeze(0).to(x.device)

class QualityGate(nn.Module):
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
        q = self.q(x)
        scores = q.sum(-1)
        if mask is not None:
            scores = scores.masked_fill(mask==0, float("-inf"))
        w = torch.softmax(scores, dim=1).unsqueeze(-1)
        v = self.v(x)
        return (w * v).sum(dim=1)

class EnhancedHCAT(nn.Module):
    def __init__(self, d_model=512, n_modalities=5, n_heads=8, n_global_layers=2, dropout=0.2, 
                 use_advanced_imputation=True, n_impute_iterations=3):
        super().__init__()
        self.use_advanced_imputation = use_advanced_imputation
        
        # Advanced imputation module
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
        
        # Enhanced transformer layers with better normalization
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead=n_heads, dim_feedforward=2048, 
            batch_first=True, dropout=dropout, activation="gelu",
            norm_first=True  # Pre-norm for better training stability
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
        
        # Enhanced prediction heads with residual connections
        self.route_surv = nn.Sequential(
            nn.Linear(d_model, 256), 
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(256, 3)
        )
        self.route_rec = nn.Sequential(
            nn.Linear(d_model, 256), 
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(256, 3)
        )
        self.head_surv = nn.Sequential(
            nn.Linear(d_model, 256), 
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        self.head_rec = nn.Sequential(
            nn.Linear(d_model, 256), 
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        self.contrast_proj = nn.Linear(d_model, 128)
        
        # Uncertainty-aware weighting
        self.uncertainty_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, emb, quality, present_mask, training=True):
        """
        Enhanced forward pass with advanced imputation
        """
        device = emb.device
        kl_loss = torch.tensor(0.0, device=device)
        imputation_loss = torch.tensor(0.0, device=device)
        
        if self.imputer is not None and training:
            # Advanced imputation with uncertainty quantification
            imputed_emb, imp_uncertainty, kl_loss, imputation_loss = self.imputer(emb, present_mask, quality)
            
            # Use imputed embeddings for missing modalities
            missing_mask = (present_mask == 0).unsqueeze(-1)
            emb = torch.where(missing_mask, imputed_emb, emb)
            
            # Adjust quality based on imputation uncertainty
            quality = torch.where(
                present_mask == 0, 
                quality * (1.0 - imp_uncertainty * self.uncertainty_weight), 
                quality
            )
        elif self.imputer is not None and not training:
            # Use imputation during inference as well
            with torch.no_grad():
                imputed_emb, imp_uncertainty, _, _ = self.imputer(emb, present_mask, quality)
                missing_mask = (present_mask == 0).unsqueeze(-1)
                emb = torch.where(missing_mask, imputed_emb, emb)
                quality = torch.where(
                    present_mask == 0, 
                    quality * (1.0 - imp_uncertainty * self.uncertainty_weight), 
                    quality
                )
        
        # Standard HCAT processing
        x = self.pos(emb)
        qg = self.quality_gate(quality) * present_mask.float()
        x = x * qg.unsqueeze(-1)
        
        # Branch processing with enhanced connections
        temporal_tokens = x[:, [0,1], :]  # clinical, temporal
        spatial_tokens = x[:, [2,4], :]   # pathological, spatial  
        semantic_token = x[:, [3], :]     # semantic
        
        temporal_local = self.local_temporal(temporal_tokens)
        spatial_local = self.local_spatial(spatial_tokens)
        semantic_local = self.semantic_proj(semantic_token.squeeze(1)).unsqueeze(1)
        
        # Global attention across branches
        concat = torch.cat([temporal_local, spatial_local, semantic_local], dim=1)
        concat = self.global_enc(concat)
        
        # Extract branch outputs
        t_len = temporal_local.shape[1]
        s_len = spatial_local.shape[1]
        t_out = concat[:, :t_len, :]
        s_out = concat[:, t_len:t_len+s_len, :]
        sem_out = concat[:, t_len+s_len:, :]
        
        # Branch-specific pooling
        t_vec = self.pool(t_out)
        s_vec = self.pool(s_out)
        sem_vec = self.pool(sem_out)
        
        # Cross-modal fusion
        branches = torch.stack([t_vec, s_vec, sem_vec], dim=1)
        fused = self.cross_enc(branches)
        pooled = fused.mean(dim=1)
        
        # Task-specific routing
        w_surv = torch.softmax(self.route_surv(pooled), dim=-1)
        w_rec = torch.softmax(self.route_rec(pooled), dim=-1)
        rep_surv = (w_surv.unsqueeze(-1) * fused).sum(dim=1)
        rep_rec = (w_rec.unsqueeze(-1) * fused).sum(dim=1)
        
        # Final predictions
        logit_surv = self.head_surv(rep_surv).squeeze(-1)
        logit_rec = self.head_rec(rep_rec).squeeze(-1)
        cproj = F.normalize(self.contrast_proj(fused.mean(dim=1)), dim=-1)
        
        return {
            "logit_surv": logit_surv, 
            "logit_rec": logit_rec, 
            "rep": pooled, 
            "cproj": cproj,
            "kl_loss": kl_loss,
            "imputation_loss": imputation_loss
        }

# ----------------------------
# Enhanced Loss Functions
# ----------------------------
def bce_masked_with_posweight(logits, targets, mask, pos_weight=None):
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    logits_sel = logits[mask==1]
    targets_sel = targets[mask==1].float()
    if pos_weight is None:
        return F.binary_cross_entropy_with_logits(logits_sel, targets_sel)
    else:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=logits.device))
        return loss_fn(logits_sel, targets_sel)

def info_nce_loss(z, temperature=0.07):
    z = F.normalize(z, dim=-1)
    sim = torch.matmul(z, z.t()) / temperature
    labels = torch.arange(z.size(0), device=z.device)
    return F.cross_entropy(sim, labels)

def focal_loss(logits, targets, alpha=1.0, gamma=2.0):
    """Focal loss for handling class imbalance"""
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
    p_t = torch.exp(-ce_loss)
    focal_loss = alpha * (1-p_t)**gamma * ce_loss
    return focal_loss.mean()

def compute_metrics(preds, targets):
    if len(targets) == 0:
        return {"acc":0.0, "f1":0.0, "auc":0.0}
    acc = float(accuracy_score(targets, preds))
    f1 = float(f1_score(targets, preds, zero_division=0))
    try:
        auc = float(roc_auc_score(targets, preds))
    except:
        auc = 0.0
    return {"acc": acc, "f1": f1, "auc": auc}

# ----------------------------
# Enhanced Training Loop
# ----------------------------
def train(args):
    ds = MultiModalDataset(args.clinical, args.semantic, args.temporal, args.pathological, args.spatial)
    n = len(ds)
    print("Total clinical patients:", n)
    idxs = list(range(n))
    random.seed(args.seed); random.shuffle(idxs)
    split = int(n * (1 - args.val_fraction))
    train_idx, val_idx = idxs[:split], idxs[split:]
    train_loader = DataLoader(torch.utils.data.Subset(ds, train_idx), batch_size=args.batch, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(torch.utils.data.Subset(ds, val_idx), batch_size=args.batch, shuffle=False, pin_memory=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model = EnhancedHCAT(
        d_model=args.d_model, 
        n_modalities=5, 
        n_heads=args.n_heads, 
        n_global_layers=args.n_global_layers, 
        dropout=args.dropout,
        use_advanced_imputation=args.use_advanced_imputation,
        n_impute_iterations=args.n_impute_iterations
    ).to(device)

    # Enhanced optimizer with weight decay scheduling
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.epochs//4, T_mult=2, eta_min=args.lr*0.01
    )

    # Compute pos_weights from training data
    surv_pos = 0; surv_neg = 0; rec_pos = 0; rec_neg = 0
    for i in train_idx:
        t = ds.patients[i]['surv_label']
        if t == 1: surv_pos += 1
        elif t == 0: surv_neg += 1
        t2 = ds.patients[i]['rec_label']
        if t2 == 1: rec_pos += 1
        elif t2 == 0: rec_neg += 1
    pos_w_surv = (surv_neg / max(1, surv_pos)) if surv_pos>0 else 1.0
    pos_w_rec = (rec_neg / max(1, rec_pos)) if rec_pos>0 else 1.0
    print(f"[pos_weights] surv pos_weight={pos_w_surv:.3f}, rec pos_weight={pos_w_rec:.3f}")

    best = {"surv_f1": -1.0, "rec_f1": -1.0, "avg_f1": -1.0}
    history = {"train_loss": [], "val_surv_f1": [], "val_rec_f1": [], "val_avg_f1": []}
    no_improv = 0

    for epoch in range(1, args.epochs+1):
        model.train()
        running_loss = 0.0; steps = 0; 
        total_kl = 0.0; total_imp = 0.0
        
        for batch in train_loader:
            emb = batch["emb"].to(device)
            quality = batch["quality"].to(device)
            present = batch["present"].to(device)
            targets = batch["targets"].to(device)
            surv_t = targets[:,0]; rec_t = targets[:,1]
            surv_mask = (surv_t != -1); rec_mask = (rec_t != -1)

            # Modality dropout augmentation
            if args.moddrop > 0:
                if args.moddrop_mode == "per_sample":
                    drop = (torch.rand_like(present.float()) < args.moddrop).to(device)
                    drop[:,0] = 0  # always keep clinical
                    present = present * (~drop).to(torch.int8)
                    emb = emb * present.unsqueeze(-1)
                    quality = quality * present.float()
                else:
                    if random.random() < args.moddrop:
                        keep = torch.ones_like(present).to(device)
                        keep[:,0] = 1
                        other_idx = random.choice([1,2,3,4])
                        keep[:,other_idx] = 0
                        present = present * keep
                        emb = emb * present.unsqueeze(-1)
                        quality = quality * present.float()

            # Embedding noise
            if args.emb_noise > 0:
                emb = emb + torch.randn_like(emb) * args.emb_noise

            out = model(emb, quality, present, training=True)

            # Main task losses with focal loss option
            if args.use_focal_loss:
                loss_surv = focal_loss(out["logit_surv"][surv_mask], (surv_t[surv_mask]==1).float()) if surv_mask.any() else torch.tensor(0.0, device=device)
                loss_rec = focal_loss(out["logit_rec"][rec_mask], (rec_t[rec_mask]==1).float()) if rec_mask.any() else torch.tensor(0.0, device=device)
            else:
                loss_surv = bce_masked_with_posweight(out["logit_surv"], (surv_t==1).float(), surv_mask, pos_weight=pos_w_surv)
                loss_rec = bce_masked_with_posweight(out["logit_rec"], (rec_t==1).float(), rec_mask, pos_weight=pos_w_rec)
            
            loss_con = info_nce_loss(out["cproj"], temperature=args.contrastive_temp) if args.use_contrastive else torch.tensor(0.0, device=device)
            
            # Enhanced loss with KL and imputation terms
            loss = (args.alpha_surv * loss_surv + 
                   args.alpha_rec * loss_rec + 
                   args.alpha_con * loss_con + 
                   args.alpha_kl * out["kl_loss"] + 
                   args.alpha_impute * out["imputation_loss"])

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.detach())
            total_kl += float(out["kl_loss"].detach())
            total_imp += float(out["imputation_loss"].detach())
            steps += 1

        train_loss = running_loss / max(1, steps)
        avg_kl = total_kl / max(1, steps)
        avg_imp = total_imp / max(1, steps)
        history["train_loss"].append(train_loss)

        # Enhanced validation with MC-dropout
        model.eval()
        T = max(1, int(args.mc_samples))
        all_surv_preds = []; all_surv_targets = []; all_surv_probs = []
        all_rec_preds = []; all_rec_targets = []; all_rec_probs = []
        
        with torch.no_grad():
            for batch in val_loader:
                emb = batch["emb"].to(device)
                quality = batch["quality"].to(device)
                present = batch["present"].to(device)
                targets = batch["targets"].to(device)
                surv_t = targets[:,0]; rec_t = targets[:,1]
                surv_mask = (surv_t != -1); rec_mask = (rec_t != -1)
                
                preds_surv_T = []; preds_rec_T = []
                
                # Enable dropout for MC sampling
                for module in model.modules():
                    if isinstance(module, nn.Dropout):
                        module.train()

                for _ in range(T):
                    out = model(emb, quality, present, training=False)
                    preds_surv_T.append(torch.sigmoid(out["logit_surv"]).cpu().numpy())
                    preds_rec_T.append(torch.sigmoid(out["logit_rec"]).cpu().numpy())
                
                model.eval() # Revert all modules to eval mode
                
                # Average over MC samples
                preds_surv = np.stack(preds_surv_T, axis=0).mean(axis=0)
                preds_rec = np.stack(preds_rec_T, axis=0).mean(axis=0)
                
                pred_sv_b = (preds_surv >= 0.5).astype(int)
                pred_rec_b = (preds_rec >= 0.5).astype(int)
                
                for i in range(len(surv_t)):
                    if surv_mask[i]:
                        all_surv_preds.append(int(pred_sv_b[i]))
                        all_surv_targets.append(int((surv_t[i]==1).cpu().item()))
                        all_surv_probs.append(float(preds_surv[i]))
                    if rec_mask[i]:
                        all_rec_preds.append(int(pred_rec_b[i]))
                        all_rec_targets.append(int((rec_t[i]==1).cpu().item()))
                        all_rec_probs.append(float(preds_rec[i]))

        surv_metrics = compute_metrics(all_surv_preds, all_surv_targets)
        rec_metrics = compute_metrics(all_rec_preds, all_rec_targets)
        avg_f1 = (surv_metrics["f1"] + rec_metrics["f1"]) / 2
        
        history["val_surv_f1"].append(surv_metrics["f1"])
        history["val_rec_f1"].append(rec_metrics["f1"])
        history["val_avg_f1"].append(avg_f1)
        
        print(f"Epoch {epoch}/{args.epochs} | Loss: {train_loss:.4f} (KL: {avg_kl:.4f}, Imp: {avg_imp:.4f}) | "
              f"Val - Surv F1: {surv_metrics['f1']:.4f} (AUC: {surv_metrics['auc']:.4f}) | "
              f"Rec F1: {rec_metrics['f1']:.4f} (AUC: {rec_metrics['auc']:.4f}) | Avg F1: {avg_f1:.4f}")

        # Enhanced model saving
        saved = False
        if surv_metrics["f1"] > best["surv_f1"]:
            best["surv_f1"] = surv_metrics["f1"]
            torch.save(model.state_dict(), os.path.join(args.outdir, "enhanced_hcat_best_surv.pt"))
            saved = True
        if rec_metrics["f1"] > best["rec_f1"]:
            best["rec_f1"] = rec_metrics["f1"]
            torch.save(model.state_dict(), os.path.join(args.outdir, "enhanced_hcat_best_rec.pt"))
            saved = True
        if avg_f1 > best["avg_f1"]:
            best["avg_f1"] = avg_f1
            torch.save(model.state_dict(), os.path.join(args.outdir, "enhanced_hcat_best_avg.pt"))
            saved = True
        if saved:
            print("✓ Saved improved checkpoint.")

        # Early stopping based on average F1
        if avg_f1 <= best["avg_f1"]:
            no_improv += 1
        else:
            no_improv = 0
        if args.early_stop_patience > 0 and no_improv >= args.early_stop_patience:
            print(f"Early stopping after {no_improv} epochs without improvement.")
            break

    torch.save(model.state_dict(), os.path.join(args.outdir, "enhanced_hcat_final.pt"))
    
    # Enhanced summary
    summary = {
        "n_patients": len(ds),
        "best_metrics": best,
        "history": history,
        "model_config": {
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_global_layers": args.n_global_layers,
            "dropout": args.dropout,
            "use_advanced_imputation": args.use_advanced_imputation,
            "n_impute_iterations": args.n_impute_iterations
        },
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "moddrop": args.moddrop,
            "emb_noise": args.emb_noise
        }
    }
    
    with open(os.path.join(args.outdir, "enhanced_hcat_training_summary.json"), "w") as wf:
        json.dump(summary, wf, indent=2)
    print("Enhanced training finished. Summary saved to:", os.path.join(args.outdir, "enhanced_hcat_training_summary.json"))

# ----------------------------
# Enhanced CLI & entrypoint
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Enhanced HCAT with Advanced Multi-Modal Imputation")
    
    # Data paths
    p.add_argument("--clinical", required=True, help="Path to clinical H5 file")
    p.add_argument("--semantic", required=True, help="Path to semantic H5 file")
    p.add_argument("--temporal", required=True, help="Path to temporal H5 file")
    p.add_argument("--pathological", required=True, help="Path to pathological H5 file")
    p.add_argument("--spatial", required=True, help="Path to spatial H5 file")
    p.add_argument("--outdir", required=True, help="Output directory for checkpoints and logs")
    
    # Training parameters
    p.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    p.add_argument("--batch", type=int, default=32, help="Batch size")
    p.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    p.add_argument("--device", type=str, default="cuda", help="Training device")
    p.add_argument("--val_fraction", type=float, default=0.2, help="Validation fraction")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    p.add_argument("--early_stop_patience", type=int, default=12, help="Early stopping patience")
    
    # Model architecture
    p.add_argument("--d_model", type=int, default=512, help="Model dimension")
    p.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    p.add_argument("--n_global_layers", type=int, default=3, help="Number of global transformer layers")
    p.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    
    # Advanced imputation
    p.add_argument("--use_advanced_imputation", action="store_true", default=True, 
                   help="Use advanced multi-modal imputation")
    p.add_argument("--n_impute_iterations", type=int, default=3, 
                   help="Number of iterative imputation refinements")
    
    # Loss weights
    p.add_argument("--alpha_surv", type=float, default=1.0, help="Survival loss weight")
    p.add_argument("--alpha_rec", type=float, default=1.0, help="Recurrence loss weight")
    p.add_argument("--alpha_con", type=float, default=0.1, help="Contrastive loss weight")
    p.add_argument("--alpha_kl", type=float, default=0.05, help="KL divergence loss weight")
    p.add_argument("--alpha_impute", type=float, default=0.1, help="Imputation loss weight")
    
    # Enhanced features
    p.add_argument("--use_contrastive", action="store_true", help="Use contrastive learning")
    p.add_argument("--contrastive_temp", type=float, default=0.07, help="Contrastive temperature")
    p.add_argument("--use_focal_loss", action="store_true", help="Use focal loss for class imbalance")
    p.add_argument("--mc_samples", type=int, default=16, help="MC dropout samples for uncertainty")
    
    # Augmentation
    p.add_argument("--moddrop", type=float, default=0.1, help="Modality dropout probability")
    p.add_argument("--moddrop_mode", type=str, default="per_sample", choices=["per_sample", "per_batch"])
    p.add_argument("--emb_noise", type=float, default=0.05, help="Embedding noise std")
    
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    
    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    print("🚀 Starting Enhanced HCAT Training with Advanced Multi-Modal Imputation")
    print("=" * 80)
    print(f"Advanced Imputation: {args.use_advanced_imputation}")
    print(f"Imputation Iterations: {args.n_impute_iterations}")
    print(f"Model Dimension: {args.d_model}")
    print(f"Attention Heads: {args.n_heads}")
    print(f"Global Layers: {args.n_global_layers}")
    print(f"Dropout: {args.dropout}")
    print(f"Learning Rate: {args.lr}")
    print(f"Batch Size: {args.batch}")
    print(f"Epochs: {args.epochs}")
    print("=" * 80)
    
    train(args)
