# #!/usr/bin/env python3
# """
# spatial_preproc_to_512.py

# Aggregate UNI patch-level WSI embeddings (features + coords) into patient-level 512-d
# embeddings using a spatially-aware Transformer aggregator with representative patch
# clustering and Monte-Carlo dropout uncertainty estimation.

# Saves outputs to the provided --outdir (default: ...\training_dataset\spatial).

# """
import os
import json
import h5py
import math
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import joblib
import warnings
warnings.filterwarnings("ignore")

# sklearn for clustering & scaling
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

# PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH = True
except Exception:
    TORCH = False

# ---------------- utils ----------------
def find_h5_files(d):
    p = Path(d)
    if not p.exists():
        return []
    return sorted([str(x) for x in p.glob("*.h5")])

def load_patch_h5(path):
    """
    Try to read features and coords from an h5 file.
    Assumes datasets are named 'features' and 'coords' or similar.
    Returns (features_np, coords_np) with shapes (n_patches, feat_dim), (n_patches, 2)
    """
    try:
        with h5py.File(path, "r") as f:
            # common keys
            for key in ("features","feature","patch_features","embeddings"):
                if key in f:
                    feats = np.array(f[key])
                    break
            else:
                # fallback: find first 2D dataset
                feats = None
                for k in f.keys():
                    data = f[k]
                    if isinstance(data, h5py.Dataset) and data.ndim == 2:
                        feats = np.array(data)
                        break
            # coords
            for key in ("coords","coordinates","patch_coords","locations"):
                if key in f:
                    coords = np.array(f[key])
                    break
            else:
                coords = None
    except Exception as e:
        # try json
        feats = None
        coords = None
    # postprocess
    if feats is None:
        raise RuntimeError(f"No patch features found in {path}")
    if coords is None:
        # create dummy coords if missing (grid indices)
        n = feats.shape[0]
        coords = np.vstack([np.arange(n), np.zeros(n)]).T
    # ensure float32
    feats = np.asarray(feats, dtype=np.float32)
    coords = np.asarray(coords, dtype=np.float32)
    return feats, coords

# ---------------- model components ----------------
if TORCH:
    class PositionalMLP(nn.Module):
        """Project 2D coordinates to model_dim and produce a learned positional bias."""
        def __init__(self, model_dim):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(2, model_dim),
                nn.GELU(),
                nn.Linear(model_dim, model_dim)
            )
        def forward(self, coords_norm):
            # coords_norm: (batch, n, 2)
            return self.mlp(coords_norm)

    class PatchProjector(nn.Module):
        """Project patch features to model_dim."""
        def __init__(self, feat_dim, model_dim):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(feat_dim, model_dim),
                nn.GELU(),
                nn.LayerNorm(model_dim)
            )
        def forward(self, x):
            return self.proj(x)

    class SpatialTransformerAggregator(nn.Module):
        """
        Transformer aggregator:
         - input: (batch=1, n_patches, model_dim)
         - prepend a learnable CLS token and apply TransformerEncoder
         - output: CLS embedding (model_dim)
        """
        def __init__(self, model_dim=512, n_heads=8, n_layers=4, dropout=0.1):
            super().__init__()
            self.model_dim = model_dim
            encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=n_heads,
                                                       dim_feedforward=model_dim*4, dropout=dropout,
                                                       activation='gelu', batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)
            self.dropout = nn.Dropout(dropout)
            # small MLP projection to 512 (identity if same)
            self.out_proj = nn.Identity()

        def forward(self, patch_embs, attn_mask=None):
            # patch_embs: (1, n, model_dim)
            b = patch_embs.shape[0]
            cls = self.cls_token.expand(b, -1, -1)  # (b,1,model_dim)
            x = torch.cat([cls, patch_embs], dim=1)  # (b, n+1, d)
            x = self.transformer(x, src_key_padding_mask=attn_mask)  # attn_mask: (b, n) True indicates padding
            cls_out = x[:, 0, :]  # (b, d)
            out = self.out_proj(cls_out)
            return out

# ---------------- pipeline ----------------
def pipeline(args):
    # gather all input files from multiple directories
    input_dirs = [
        args.wsi_lymph, args.wsi_cup, args.wsi_larynx, args.wsi_oral,
        args.wsi_oroph2, args.wsi_oroph1, args.wsi_hypo
    ]
    input_files = []
    for d in input_dirs:
        if d:
            input_files.extend(find_h5_files(d))
    input_files = sorted(list(set(input_files)))
    if len(input_files) == 0:
        print("No h5 files found in given WSI directories. Exiting.")
        return

    OUT_DIR = Path(args.outdir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_h5 = OUT_DIR / "spatial_embeddings_512.h5"
    summary_json = OUT_DIR / "spatial_summary.json"
    objfile = OUT_DIR / "spatial_preproc_objects.joblib"

    device = "cuda" if (args.use_gpu and TORCH and torch.cuda.is_available()) else "cpu"
    print("Device used:", device)

    # model setup
    model_dim = args.model_dim
    if TORCH:
        # instantiate modules
        # placeholder feat_dim; will create projector on-the-fly per slide if dims vary
        aggregator = SpatialTransformerAggregator(model_dim=model_dim, n_heads=args.n_heads,
                                                 n_layers=args.n_layers, dropout=args.dropout).to(device)
        pos_mlp = PositionalMLP(model_dim).to(device)
        # projector is created per-slide if feature dims differ (we'll create a new PatchProjector)
    else:
        aggregator = None
        pos_mlp = None

    patient_ids = []
    embeddings_mean = []
    embeddings_var = []
    n_patches_list = []
    selected_coords_list = []
    quality_list = []
    failed = []

    # For reproducible clustering result across slides, seed numpy
    np.random.seed(args.seed)

    for path in tqdm(input_files, desc="slides"):
        fname = Path(path).name
        # derive patient id from filename (numeric pattern)
        m = None
        import re
        m = re.search(r'(\d{1,6})', fname)
        if m:
            pid = m.group(1).zfill(3)
        else:
            pid = fname  # fallback
        try:
            feats, coords = load_patch_h5(path)  # (n_patches, feat_dim), (n_patches,2)
        except Exception as e:
            failed.append((path, str(e)))
            continue

        n_patches = feats.shape[0]
        feat_dim = feats.shape[1]
        if n_patches == 0:
            failed.append((path, "no patches"))
            continue

        # L2 normalize patch features (makes cosine/attention behave nicely)
        fnorm = np.linalg.norm(feats, axis=1, keepdims=True)
        fnorm[fnorm==0] = 1.0
        feats = feats / fnorm

        # normalize coords to [0,1] based on min/max present (per-slide)
        coords_min = coords.min(axis=0)
        coords_max = coords.max(axis=0)
        span = coords_max - coords_min
        span[span == 0] = 1.0
        coords_norm = (coords - coords_min) / span

        # Optionally reduce patches if > max_patches using MiniBatchKMeans representative clustering
        max_patches = args.max_patches
        if n_patches > max_patches:
            # cluster in feature space + coords concatenated (so clustering respects spatial and feature similarity)
            cluster_X = np.concatenate([feats, coords_norm], axis=1)
            k = max_patches
            mbk = MiniBatchKMeans(n_clusters=k, random_state=args.seed, batch_size=256)
            mbk.fit(cluster_X)
            centers = mbk.cluster_centers_
            # Use centers' feature part as representative; also compute corresponding coords
            rep_feats = centers[:, :feat_dim]
            rep_coords = centers[:, feat_dim:feat_dim+2]
            sel_feats = rep_feats.astype(np.float32)
            sel_coords = rep_coords.astype(np.float32)
            selected_count = k
        else:
            sel_feats = feats.astype(np.float32)
            sel_coords = coords_norm.astype(np.float32)
            selected_count = n_patches

        # quality score: fraction of patches retained and patch coverage
        quality = float(selected_count / max(1, n_patches))

        # prepare tensors
        if TORCH:
            # create projector for this feat_dim
            projector = PatchProjector(feat_dim, model_dim).to(device)
            # project features, add positional bias
            with torch.no_grad():
                Xp = torch.from_numpy(sel_feats).float().to(device)  # (n, feat_dim)
                patch_emb = projector(Xp)  # (n, model_dim)
                coords_t = torch.from_numpy(sel_coords).float().unsqueeze(0).to(device)  # (1, n, 2)
                pos_bias = pos_mlp(coords_t).squeeze(0)  # (n, model_dim)
                patch_emb = patch_emb + pos_bias  # spatial bias
                patch_emb = patch_emb.unsqueeze(0)  # (1, n, d)

            # create attention mask: no padding here (all valid)
            attn_mask = None

            # Monte-Carlo dropout: do T forward passes with dropout enabled to get mean+var
            T = max(1, int(args.mc_samples))
            aggregator.train()  # enable dropout
            out_embs = []
            for t in range(T):
                with torch.no_grad():
                    emb = aggregator(patch_emb, attn_mask=attn_mask)  # (1, d)
                    out_embs.append(emb.cpu().numpy().reshape(-1))
            out_arr = np.stack(out_embs, axis=0)  # (T, d)
            mean_emb = out_arr.mean(axis=0).astype(np.float32)
            var_emb = out_arr.var(axis=0).astype(np.float32)
        else:
            # CPU fallback: simple attention-free aggregation: mean-pool of sel_feats projected to 512 via linear approx
            # We'll do PCA-like linear map (random projection) deterministically
            rng = np.random.RandomState(args.seed)
            W = rng.normal(size=(feat_dim, model_dim)).astype(np.float32)
            patch_emb_np = sel_feats.dot(W)  # (n, model_dim)
            mean_emb = patch_emb_np.mean(axis=0).astype(np.float32)
            var_emb = patch_emb_np.var(axis=0).astype(np.float32)

        # append results
        patient_ids.append(pid)
        embeddings_mean.append(mean_emb)
        embeddings_var.append(var_emb)
        n_patches_list.append(n_patches)
        selected_coords_list.append(sel_coords.astype(np.float32))
        quality_list.append(quality)

    # Save results into HDF5. We store selected_coords as variable-length arrays using h5py special dtype.
    print("Saving outputs to:", out_h5)
    dt = h5py.string_dtype(encoding='utf-8')
    with h5py.File(str(out_h5), "w") as f:
        f.create_dataset("patient_id", data=np.array(patient_ids, dtype='S'))
        f.create_dataset("embedding_mean_512", data=np.stack(embeddings_mean).astype(np.float32))
        f.create_dataset("embedding_var_512", data=np.stack(embeddings_var).astype(np.float32))
        f.create_dataset("n_patches", data=np.array(n_patches_list, dtype=np.int32))
        f.create_dataset("quality_score", data=np.array(quality_list, dtype=np.float32))
        # store coords as ragged: create a group and variable-length datasets
        coords_grp = f.create_group("selected_patch_coords")
        for i, coords in enumerate(selected_coords_list):
            coords_grp.create_dataset(str(i), data=coords.astype(np.float32))
        # store mapping index -> patient_id
        f.create_dataset("patient_index_map", data=np.array(patient_ids, dtype='S'))

    # Save summary.json and preproc objects
    summary = {
        "n_slides_processed": len(patient_ids),
        "n_files_found": len(input_files),
        "n_failed": len(failed),
        "failed_samples": failed,
        "max_patches_kept": args.max_patches,
        "model_dim": model_dim,
        "transformer_layers": args.n_layers if TORCH else 0,
        "mc_samples": args.mc_samples if TORCH else 1,
        "notes": [
            "Representative patch clustering (MiniBatchKMeans) used when slide patches > max_patches.",
            "Positional MLP projects normalized patch (x,y) to the same model dim and is added to projected features.",
            "Global slide embedding is the CLS token output of a Transformer encoder aggregated over patches.",
            "Monte-Carlo dropout (aggregator.train() with no grad) used to estimate embedding uncertainty."
        ]
    }
    with open(str(summary_json), "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2)

    # save preproc objects (none heavy) for reproducibility
    joblib.dump({
        "args": vars(args),
    }, str(objfile))

    print("Done. Outputs saved to:", OUT_DIR)
    return

# ----------------- CLI -----------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--wsi_lymph", default=None, help="folder with LymphNode h5 files")
    p.add_argument("--wsi_cup", default=None, help="folder with PrimaryTumor CUP h5 files")
    p.add_argument("--wsi_larynx", default=None)
    p.add_argument("--wsi_oral", default=None)
    p.add_argument("--wsi_oroph2", default=None)
    p.add_argument("--wsi_oroph1", default=None)
    p.add_argument("--wsi_hypo", default=None)
    p.add_argument("--outdir", required=True)
    p.add_argument("--max_patches", type=int, default=1024)
    p.add_argument("--model_dim", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--mc_samples", type=int, default=16)
    p.add_argument("--use_gpu", action="store_true", help="attempt to use GPU (requires torch)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    pipeline(args)
