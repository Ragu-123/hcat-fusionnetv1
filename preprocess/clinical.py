# #!/usr/bin/env python3
# """
# clinical_preproc_novel.py

# Preprocess clinical_data.json -> advanced imputation ensemble -> save HDF5.

# Outputs:
#  C:\Users\haris\Downloads\project phase-1\training_dataset\clinical_preprocessed_advanced.h5
#  and preprocessing objects in same folder.

# Author: ChatGPT (pipeline for HANCOCK clinical preprocessing)
# """

import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import h5py
import joblib
import argparse
import math
import warnings
warnings.filterwarnings("ignore")

# ML libs
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split

# torch for VAE
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH = True
except Exception:
    TORCH = False

# ---------- Config ----------
INPUT_JSON = r"C:\Users\haris\Downloads\project phase-1\dataset\clinicalStructuredData\StructuredData\clinical_data.json"
OUT_DIR = Path(r"C:\Users\haris\Downloads\project phase-1\training_dataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_H5 = OUT_DIR / "clinical_preprocessed_advanced.h5"
OBJ_STORE = OUT_DIR / "clinical_preproc_objects.joblib"
VAE_WEIGHTS = OUT_DIR / "clinical_vae.pt"

EMBED_DIM = 512  # not used here but kept for compatibility if you extend
SEED = 42
M_IMPUTATIONS = 5  # number of multiple imputations to generate
VAE_EPOCHS = 60
VAE_BS = 64
VAE_LR = 1e-3
GRAPH_ALPHA = 0.6   # smoothing weight (0..1) -- higher means more neighbor influence
DEVICE = ("cuda" if TORCH and torch.cuda.is_available() else "cpu")

# time windows for labels
DAYS_5Y = 5 * 365
DAYS_2Y = 2 * 365

np.random.seed(SEED)
if TORCH:
    torch.manual_seed(SEED)

# ---------- Helpers ----------
def load_json(path):
    df = pd.read_json(path)
    return df

def derive_three_state_labels(df):
    df = df.copy()
    surv_stat = df.get("survival_status", pd.Series(["unknown"] * len(df))).astype(str).str.lower()
    surv_cause = df.get("survival_status_with_cause", pd.Series([""] * len(df))).astype(str).str.lower()
    followup = pd.to_numeric(df.get("days_to_last_information", pd.Series([np.nan]*len(df))), errors="coerce")
    rec_flag = df.get("recurrence", pd.Series(["unknown"] * len(df))).astype(str).str.lower()
    days_rec = pd.to_numeric(df.get("days_to_recurrence", pd.Series([np.nan]*len(df))), errors="coerce")

    n = len(df)
    surv = np.full(n, -1, dtype=np.int8)
    rec = np.full(n, -1, dtype=np.int8)

    is_deceased_tumor = surv_cause.str.contains("tumor", na=False)
    died_within = is_deceased_tumor & (followup <= DAYS_5Y)
    died_after = is_deceased_tumor & (followup > DAYS_5Y)
    surv[died_within.values] = 0
    surv[died_after.values] = 1
    surv[(surv_stat == "living") & (followup >= DAYS_5Y)] = 1

    rec_pos = (rec_flag == "yes") & (days_rec <= DAYS_2Y)
    rec_neg = (rec_flag == "no") & (followup >= DAYS_2Y)
    rec[rec_pos.values] = 1
    rec[rec_neg.values] = 0

    return surv, rec

def auto_select_columns(df):
    skip = {"patient_id", "surv_5yr_label", "rec_2yr_label", "survival_status", "survival_status_with_cause",
            "days_to_last_information", "days_to_recurrence"}
    numeric = []
    categorical = []
    for c in df.columns:
        if c in skip:
            continue
        ser = df[c]
        if pd.api.types.is_numeric_dtype(ser):
            numeric.append(c)
        else:
            coercible = pd.to_numeric(ser, errors="coerce")
            if coercible.notna().sum() / max(1, len(ser)) > 0.5:
                numeric.append(c)
            else:
                categorical.append(c)
    return numeric, categorical

def frequency_encode(series):
    return series.fillna("<<MISSING>>").astype(str).map(series.fillna("<<MISSING>>").astype(str).value_counts(normalize=True))

# ---------- VAE imputer ----------
if TORCH:
    class VAEImputer(nn.Module):
        def __init__(self, inp_dim, latent_dim=64):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(inp_dim, max(128, inp_dim//2)),
                nn.ReLU(),
                nn.Linear(max(128, inp_dim//2), 128),
                nn.ReLU()
            )
            self.fc_mu = nn.Linear(128, latent_dim)
            self.fc_logvar = nn.Linear(128, latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, max(128, inp_dim//2)),
                nn.ReLU(),
                nn.Linear(max(128, inp_dim//2), inp_dim)
            )

        def reparameterize(self, mu, logvar):
            std = (0.5*logvar).exp()
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, x):
            h = self.encoder(x)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            z = self.reparameterize(mu, logvar)
            recon = self.decoder(z)
            return recon, mu, logvar

    def train_vae_imputer(X_obs, mask_obs, latent_dim=64, epochs=VAE_EPOCHS, batch_size=VAE_BS, lr=VAE_LR, device=DEVICE):
        """
        X_obs: numpy (n,d) with NaNs replaced by 0 for missing
        mask_obs: numpy (n,d) with 1 if observed, 0 if missing
        We train VAE to reconstruct only observed entries (mask-weighted MSE) + KL loss.
        """
        n,d = X_obs.shape
        model = VAEImputer(inp_dim=d, latent_dim=latent_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        X_t = torch.from_numpy(X_obs).float().to(device)
        M_t = torch.from_numpy(mask_obs).float().to(device)
        ds = TensorDataset(X_t)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        for epoch in range(1, epochs+1):
            model.train()
            epoch_loss = 0.0
            for (batch,) in dl:
                bmask = (batch != 0).float()  # not used; supply mask instead
                recon, mu, logvar = model(batch)
                # mask from global M_t by indexing batch rows
                # can't directly get batch indices from TensorDataset easily, so compute batch mask by matching
                # For simplicity and speed: compute per-batch mask by checking where batch != 0 and using provided mask.
                # Alternative: pass M as dataset; here we assume X_obs has zeros where missing and M gives mask
                # But to keep this simple: we will compute masked loss using M_t for same row indices - need indices
                # So we instead create a DataLoader of indices
                pass
        # NOTE: We will implement a separate training loop below that uses indices to get masks per batch.
        return None

# Because we need to access masks per-batch, let's write a small training helper that uses indices
def train_vae_imputer_full(X_filled, mask, latent_dim=64, epochs=VAE_EPOCHS, batch_size=VAE_BS, lr=VAE_LR, device=DEVICE):
    """
    Proper training loop where DataLoader yields indices to fetch rows from X_filled and mask.
    """
    if not TORCH:
        raise RuntimeError("PyTorch required for VAE imputer.")
    import torch
    from torch.utils.data import DataLoader, Dataset

    class IdxDataset(Dataset):
        def __init__(self, n):
            self.n = n
        def __len__(self): return self.n
        def __getitem__(self, idx): return idx

    n,d = X_filled.shape
    ds = IdxDataset(n)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = VAEImputer(inp_dim=d, latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X_tensor = torch.from_numpy(X_filled).float().to(device)
    M_tensor = torch.from_numpy(mask).float().to(device)

    for epoch in range(1, epochs+1):
        model.train()
        total = 0.0
        for idx_batch in dl:
            idx = idx_batch.to(device)
            xb = X_tensor[idx]
            mb = M_tensor[idx]
            recon, mu, logvar = model(xb)
            # reconstruction loss only on observed entries
            recon_loss = ((recon - xb)**2 * mb).sum() / (mb.sum() + 1e-8)
            # KL
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / xb.size(0)
            loss = recon_loss + 1e-3 * kld
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        total /= n
        if epoch % 10 == 0 or epoch == 1:
            print(f"VAE epoch {epoch}/{epochs} loss={total:.6f}")
    return model

def sample_vae_imputations(model, X_filled, mask, M_samples=5, device=DEVICE):
    """Return list of M_samples numpy arrays with imputed values (only fill missing positions)."""
    model.eval()
    import torch
    X_t = torch.from_numpy(X_filled).float().to(device)
    mask_t = torch.from_numpy(mask).float().to(device)
    outputs = []
    with torch.no_grad():
        for m in range(M_samples):
            # sample by passing through encoder to get mu/logvar then sample latent, decode
            # Our VAE uses reparameterize inside forward; to force randomness we just call forward
            recon, mu, logvar = model(X_t)
            recon_np = recon.cpu().numpy()
            filled = X_filled.copy()
            miss_idx = (mask == 0)
            filled[miss_idx] = recon_np[miss_idx]
            outputs.append(filled)
    return outputs

# ---------- Graph smoothing ----------
def patient_graph_smoothing(X, missing_mask, n_neighbors=10, alpha=0.6, n_iter=10):
    """
    X: (n,d) complete data (after some imputation)
    missing_mask: (n,d) 1 if observed, 0 if missing (original)
    We compute similarity on a subset of robust features (columns with low missingness).
    Then iteratively smooth only missing positions using neighbors.
    """
    n,d = X.shape
    # pick robust columns (low missingness)
    col_obs_frac = missing_mask.mean(axis=0)
    robust_cols = np.where(col_obs_frac >= 0.5)[0]
    if len(robust_cols) == 0:
        robust_cols = np.arange(min(5, d))
    S = rbf_kernel(X[:, robust_cols], gamma=1.0 / max(1.0, X[:, robust_cols].var()))
    # zero diagonal and keep top-k neighbors per row
    np.fill_diagonal(S, 0.0)
    # keep top-k
    if n_neighbors < n:
        idx_part = np.argpartition(-S, n_neighbors, axis=1)[:, :n_neighbors]
        mask_topk = np.zeros_like(S, dtype=bool)
        rows = np.arange(n)[:, None]
        mask_topk[rows, idx_part] = True
        S = S * mask_topk
    # row-normalize
    row_sums = S.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = S / row_sums
    Xs = X.copy()
    for it in range(n_iter):
        X_neighbors = P.dot(Xs)
        # update only originally missing positions: convex combination
        Xs[missing_mask == 0] = alpha * X_neighbors[missing_mask == 0] + (1 - alpha) * Xs[missing_mask == 0]
    return Xs

# ---------- Main pipeline ----------
def pipeline(args):
    print("Loading JSON:", INPUT_JSON)
    df = load_json(INPUT_JSON)
    n0 = len(df)
    print("Records loaded:", n0)

    if "patient_id" not in df.columns:
        raise ValueError("patient_id required in JSON")
    df["patient_id"] = df["patient_id"].astype(str)
    df = df.drop_duplicates(subset=["patient_id"]).set_index("patient_id", drop=False)
    patient_ids = df["patient_id"].astype(str).values
    n = len(df)
    print("Unique patient count:", n)

    # derive labels
    surv_labels, rec_labels = derive_three_state_labels(df)
    df["surv_5yr_label"] = surv_labels
    df["rec_2yr_label"] = rec_labels
    print("Label distributions (surv, rec):", np.bincount((surv_labels + 1).astype(int)), np.bincount((rec_labels + 1).astype(int)))

    # auto-select features
    numeric_cols, categorical_cols = auto_select_columns(df)
    print("Numeric cols:", numeric_cols)
    print("Categorical cols:", categorical_cols)

    # build feature frame
    feat_df = pd.DataFrame(index=df.index)
    if numeric_cols:
        feat_df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    for c in categorical_cols:
        feat_df[f"freq__{c}"] = frequency_encode(df[c])

    # missingness indicators
    for c in list(feat_df.columns):
        feat_df[f"miss__{c}"] = feat_df[c].isna().astype(int)

    feature_names = list(feat_df.columns)
    X_raw = feat_df.values.astype(float)  # may contain nan
    print("Feature matrix shape:", X_raw.shape)

    # mask of observed entries
    mask = (~np.isnan(X_raw)).astype(int)

    # preliminary simple fill for scaling (median)
    median_imp = SimpleImputer(strategy="median")
    X_median = median_imp.fit_transform(X_raw)

    # standardize features (fit on median-imputed)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_median)

    # ---------- KNN imputer baseline ----------
    print("Running KNN imputer baseline...")
    knn = KNNImputer(n_neighbors=5)
    X_knn = knn.fit_transform(X_std)
    # unscale back
    X_knn_unscaled = scaler.inverse_transform(X_knn)

    # ---------- VAE imputer (probabilistic) ----------
    vae_imputations = []
    if args.no_vae:
        print("Skipping VAE (use --no_vae to skip). Using KNN outputs as repeated imputations.")
        for _ in range(M_IMPUTATIONS):
            vae_imputations.append(X_knn_unscaled.copy())
        vae_model = None
    else:
        if not TORCH:
            raise RuntimeError("PyTorch is required for VAE imputer. Install torch or run with --no_vae.")
        print("Preparing VAE imputer (train)...")
        # For training, work in standardized space (X_std) and fill missing with 0 (since scaler fitted on median).
        X_filled = X_std.copy()
        X_filled[np.isnan(X_raw)] = 0.0
        mask_std = (~np.isnan(X_raw)).astype(float)
        # Train VAE
        latent_dim = min(64, max(8, X_std.shape[1] // 4))
        model = train_vae_imputer_full(X_filled, mask_std, latent_dim=latent_dim,
                                       epochs=args.epochs, batch_size=args.bs, lr=args.lr, device=DEVICE)
        vae_model = model
        # sample multiple imputations (standardized), then inverse transform
        vae_imps_std = sample_vae_imputations(model, X_filled, mask_std, M_samples=M_IMPUTATIONS, device=DEVICE)
        for imp_std in vae_imps_std:
            # inverse transform (use scaler)
            imp_unscaled = scaler.inverse_transform(imp_std)
            vae_imputations.append(imp_unscaled)

    # ---------- Graph smoothing refinement ----------
    print("Applying graph smoothing refinement to each imputation (leveraging cohort structure)...")
    smoothed_imputations = []
    for i, Ximp in enumerate(vae_imputations):
        # Use original missing mask (1 observed, 0 missing)
        smoothed = patient_graph_smoothing(Ximp, mask, n_neighbors=10, alpha=GRAPH_ALPHA, n_iter=10)
        smoothed_imputations.append(smoothed)
        print(f" - imputation {i+1} smoothed.")

    # Also include the KNN baseline (smoothed) as part of ensemble if not already same
    # (optional) add KNN as one of the imputations
    smoothed_imputations.append(X_knn_unscaled)

    # ---------- Aggregate multiple imputations -> mean & variance ----------
    print("Aggregating imputations to produce mean & variance (uncertainty)...")
    Imps = np.stack(smoothed_imputations, axis=0)  # shape (M', n, d)
    # If shape mismatch, ensure shapes match
    Mprime = Imps.shape[0]
    mean_imp = Imps.mean(axis=0)
    var_imp = Imps.var(axis=0)

    # For observed entries, replace with original observed values to preserve truth
    mean_imp[mask == 1] = X_raw[mask == 1]
    var_imp[mask == 1] = 0.0

    # ---------- Save HDF5 ----------
    print("Saving HDF5 to:", OUT_H5)
    dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(str(OUT_H5), "w") as f:
        f.create_dataset("patient_id", data=np.array(patient_ids, dtype="S"), dtype=dt)
        f.create_dataset("feature_names", data=np.array(feature_names, dtype="S"), dtype=dt)
        f.create_dataset("features_mean", data=mean_imp.astype(np.float32))
        f.create_dataset("features_var", data=var_imp.astype(np.float32))
        f.create_dataset("missing_mask", data=mask.astype(np.int8))
        f.create_dataset("surv_5yr_label", data=surv_labels.astype(np.int8))
        f.create_dataset("rec_2yr_label", data=rec_labels.astype(np.int8))
        f.create_dataset("n_imputations", data=np.array([Mprime], dtype=np.int32))
        # store first few imputation samples for inspection (if small)
        max_store = min(5, Mprime)
        f.create_dataset("imputation_samples", data=Imps[:max_store].astype(np.float32))

    # ---------- Save preprocessing objects ----------
    print("Saving preprocessing objects to:", OBJ_STORE)
    save_obj = {
        "feature_names": feature_names,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "median_imputer": median_imp,
        "scaler": scaler,
        "knn_imputer": knn,
        "graph_alpha": GRAPH_ALPHA,
        "vae_latent": latent_dim if not args.no_vae else None,
        "vae_epochs": args.epochs,
        "vae_bs": args.bs
    }
    joblib.dump(save_obj, str(OBJ_STORE))
    if not args.no_vae and vae_model is not None:
        torch.save(vae_model.state_dict(), str(VAE_WEIGHTS))
        print("Saved VAE weights to:", VAE_WEIGHTS)

    print("Done. Saved preprocessed HDF5 and preprocess objects.")
    # quick stats
    print(" - patients:", n)
    print(" - features:", len(feature_names))
    print(" - imputation draws used:", Mprime)
    # distribution of labels (print counts of known vs censored)
    def label_stats(arr):
        unique, counts = np.unique(arr, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
    print(" - surv_5yr_label distribution:", label_stats(surv_labels))
    print(" - rec_2yr_label distribution:", label_stats(rec_labels))
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=VAE_EPOCHS)
    parser.add_argument("--bs", type=int, default=VAE_BS)
    parser.add_argument("--lr", type=float, default=VAE_LR)
    parser.add_argument("--no_vae", action="store_true", help="skip VAE step (faster)")
    args = parser.parse_args()
    pipeline(args)
