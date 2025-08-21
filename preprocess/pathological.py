# #!/usr/bin/env python3
# """
# pathological_preproc_novel.py

# Preprocess /mnt/data/pathological_data.json -> probabilistic imputation ensemble ->
# graph smoothing -> produce 512-d pathological embeddings -> save HDF5 + summary.json.

# Saves to:
#  C:\Users\haris\Downloads\project phase-1\training_dataset\pathological

# Author: ChatGPT (adapted to your HANCOCK pipeline)
# """

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import joblib
import argparse
import warnings
warnings.filterwarnings("ignore")

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import PCA

# Try to import torch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset, TensorDataset
    TORCH = True
except Exception:
    TORCH = False

# ---------- Config ----------
INPUT_JSON = r"C:\Users\haris\Downloads\project phase-1\dataset\clinicalStructuredData\StructuredData\pathological_data.json"
OUT_DIR = Path(r"C:\Users\haris\Downloads\project phase-1\training_dataset\pathological")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_H5 = OUT_DIR / "pathological_preprocessed_advanced.h5"
EMB_H5 = OUT_DIR / "pathological_embedding_512.h5"
OBJ_STORE = OUT_DIR / "pathological_preproc_objects.joblib"
VAE_WEIGHTS = OUT_DIR / "pathological_vae.pt"
SUMMARY_JSON = OUT_DIR / "summary.json"

SEED = 42
np.random.seed(SEED)
EMBED_DIM = 512
M_IMPUTATIONS = 5
VAE_EPOCHS = 50
VAE_BS = 64
VAE_LR = 1e-3
GRAPH_ALPHA = 0.6
DEVICE = ("cuda" if TORCH and torch.cuda.is_available() else "cpu")

# ---------- Helpers ----------
def load_json(path):
    df = pd.read_json(path)
    return df

def clean_numeric_string(x):
    # convert strings like "<0.1" -> 0.05 (half); keep "0" or numeric strings; return nan if None
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return x
    s = str(x).strip()
    if s == "":
        return np.nan
    try:
        # direct numeric
        return float(s)
    except:
        # handle <val
        if s.startswith("<"):
            try:
                v = float(s.lstrip("<"))
                return v / 2.0
            except:
                return np.nan
        # handle other non-numeric tokens
        return np.nan

def auto_select_columns(df):
    skip = {"patient_id"}
    numeric = []
    categorical = []
    for c in df.columns:
        if c in skip:
            continue
        ser = df[c]
        # try coercion
        coerced = pd.to_numeric(ser.replace({None: np.nan}), errors="coerce")
        if pd.api.types.is_numeric_dtype(ser) or (coerced.notna().sum() / max(1, len(ser)) > 0.5):
            numeric.append(c)
        else:
            categorical.append(c)
    return numeric, categorical

def frequency_encode(series):
    s = series.fillna("<<MISSING>>").astype(str)
    vc = s.value_counts(normalize=True)
    return s.map(vc).astype(float)

# ---------- VAE imputer (same design as clinical) ----------
if TORCH:
    class VAEImputer(nn.Module):
        def __init__(self, inp_dim, latent_dim=64):
            super().__init__()
            hid = max(128, inp_dim//2)
            self.encoder = nn.Sequential(nn.Linear(inp_dim, hid), nn.ReLU(),
                                         nn.Linear(hid, 128), nn.ReLU())
            self.fc_mu = nn.Linear(128, latent_dim)
            self.fc_logvar = nn.Linear(128, latent_dim)
            self.decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(),
                                         nn.Linear(128, hid), nn.ReLU(),
                                         nn.Linear(hid, inp_dim))
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

    class IdxDataset(Dataset):
        def __init__(self, n): self.n=n
        def __len__(self): return self.n
        def __getitem__(self, idx): return idx

    def train_vae_imputer_full(X_filled, mask, latent_dim=64, epochs=VAE_EPOCHS, batch_size=VAE_BS, lr=VAE_LR, device=DEVICE):
        n,d = X_filled.shape
        ds = IdxDataset(n)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        model = VAEImputer(inp_dim=d, latent_dim=latent_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        X_t = torch.from_numpy(X_filled).float().to(device)
        M_t = torch.from_numpy(mask).float().to(device)
        for epoch in range(1, epochs+1):
            model.train()
            total=0.0
            for idx_batch in dl:
                idx = idx_batch.to(device)
                xb = X_t[idx]
                mb = M_t[idx]
                recon, mu, logvar = model(xb)
                recon_loss = ((recon - xb)**2 * mb).sum() / (mb.sum() + 1e-8)
                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / xb.size(0)
                loss = recon_loss + 1e-3 * kld
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item() * xb.size(0)
            total /= n
            if epoch % 10 == 0 or epoch == 1:
                print(f"VAE epoch {epoch}/{epochs} loss={total:.6f}")
        return model

    def sample_vae_imputations(model, X_filled, mask, M_samples=5, device=DEVICE):
        model.eval()
        X_t = torch.from_numpy(X_filled).float().to(device)
        outputs=[]
        with torch.no_grad():
            for _ in range(M_samples):
                recon, mu, logvar = model(X_t)
                recon_np = recon.cpu().numpy()
                filled = X_filled.copy()
                miss_idx = (mask == 0)
                filled[miss_idx] = recon_np[miss_idx]
                outputs.append(filled)
        return outputs

# ---------- Graph smoothing ----------
def patient_graph_smoothing(X, missing_mask, n_neighbors=10, alpha=GRAPH_ALPHA, n_iter=10):
    n,d = X.shape
    col_obs_frac = missing_mask.mean(axis=0)
    robust_cols = np.where(col_obs_frac >= 0.5)[0]
    if len(robust_cols) == 0:
        robust_cols = np.arange(min(5, d))
    # compute RBF on robust cols (handle zero variance)
    sub = X[:, robust_cols]
    var = sub.var()
    gamma = 1.0 / (var + 1e-6)
    S = rbf_kernel(sub, gamma=gamma)
    np.fill_diagonal(S, 0.0)
    if n_neighbors < n:
        idx_part = np.argpartition(-S, n_neighbors, axis=1)[:, :n_neighbors]
        mask_topk = np.zeros_like(S, dtype=bool)
        rows = np.arange(n)[:, None]
        mask_topk[rows, idx_part] = True
        S = S * mask_topk
    row_sums = S.sum(axis=1, keepdims=True)
    row_sums[row_sums==0] = 1.0
    P = S / row_sums
    Xs = X.copy()
    for _ in range(n_iter):
        X_neighbors = P.dot(Xs)
        Xs[missing_mask == 0] = alpha * X_neighbors[missing_mask == 0] + (1-alpha) * Xs[missing_mask == 0]
    return Xs

# ---------- Embedding AE ----------
if TORCH:
    class DenoisingAE(nn.Module):
        def __init__(self, inp_dim, bottleneck=EMBED_DIM):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(inp_dim, max(512, inp_dim//2)),
                nn.GELU(),
                nn.Linear(max(512, inp_dim//2), 1024),
                nn.GELU(),
                nn.Linear(1024, bottleneck)
            )
            self.dec = nn.Sequential(
                nn.Linear(bottleneck, 1024),
                nn.GELU(),
                nn.Linear(1024, max(512, inp_dim//2)),
                nn.GELU(),
                nn.Linear(max(512, inp_dim//2), inp_dim)
            )
        def forward(self, x):
            z = self.enc(x)
            rec = self.dec(z)
            return rec, z

    def train_autoencoder(X, epochs=40, batch_size=64, lr=1e-3, device=DEVICE):
        torch.manual_seed(SEED)
        inp_dim = X.shape[1]
        model = DenoisingAE(inp_dim=inp_dim, bottleneck=EMBED_DIM).to(device)
        X_t = torch.from_numpy(X).float().to(device)
        dl = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
        loss_fn = nn.MSELoss()
        for ep in range(1, epochs+1):
            model.train()
            tot = 0.0
            for (batch,) in dl:
                mask_drop = (torch.rand_like(batch) > 0.10).float()
                noisy = batch * mask_drop
                rec, _ = model(noisy)
                loss = loss_fn(rec, batch)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * batch.size(0)
            tot /= len(X)
            if ep % 10 == 0 or ep == 1:
                print(f"AE epoch {ep}/{epochs} loss={tot:.6f}")
        model.eval()
        with torch.no_grad():
            Z = model(torch.from_numpy(X).float().to(device))[1].cpu().numpy()
        return model, Z

# ---------- Main pipeline ----------
def pipeline(args):
    print("Loading JSON:", INPUT_JSON)
    df = load_json(INPUT_JSON)
    print("Records loaded:", len(df))
    if "patient_id" not in df.columns:
        raise ValueError("patient_id required")
    df["patient_id"] = df["patient_id"].astype(str)
    df = df.drop_duplicates(subset=["patient_id"]).set_index("patient_id", drop=False)
    patient_ids = df["patient_id"].values
    n = len(df)

    # Clean specific numeric-ish string fields
    if "closest_resection_margin_in_cm" in df.columns:
        df["closest_resection_margin_in_cm_clean"] = df["closest_resection_margin_in_cm"].apply(clean_numeric_string)
    # attempt to coerce infiltration depth and lymph counts to numeric
    for c in ["infiltration_depth_in_mm", "number_of_positive_lymph_nodes", "number_of_resected_lymph_nodes"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # auto-select features
    numeric_cols, categorical_cols = auto_select_columns(df)
    # include the cleaned margin if present
    if "closest_resection_margin_in_cm_clean" in df.columns and "closest_resection_margin_in_cm_clean" not in numeric_cols:
        numeric_cols.append("closest_resection_margin_in_cm_clean")

    print("Numeric cols:", numeric_cols)
    print("Categorical cols:", categorical_cols)

    feat_df = pd.DataFrame(index=df.index)
    if numeric_cols:
        feat_df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    for c in categorical_cols:
        feat_df[f"freq__{c}"] = frequency_encode(df[c])

    # missingness indicators
    for c in list(feat_df.columns):
        feat_df[f"miss__{c}"] = feat_df[c].isna().astype(int)

    feature_names = list(feat_df.columns)
    X_raw = feat_df.values.astype(float)
    mask = (~np.isnan(X_raw)).astype(int)
    n_features = X_raw.shape[1]
    print("Feature matrix shape:", X_raw.shape)

    # median fill for scaling
    median_imp = SimpleImputer(strategy="median")
    X_median = median_imp.fit_transform(X_raw)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_median)

    # KNN baseline
    print("Running KNN imputer baseline...")
    knn = KNNImputer(n_neighbors=5)
    X_knn = knn.fit_transform(X_std)
    X_knn_unscaled = scaler.inverse_transform(X_knn)

    # VAE probabilistic imputations
    vae_imputations = []
    if args.no_vae:
        for _ in range(M_IMPUTATIONS):
            vae_imputations.append(X_knn_unscaled.copy())
        vae_model = None
    else:
        if not TORCH:
            raise RuntimeError("PyTorch required for VAE. Use --no_vae to skip.")
        print("Training VAE imputer...")
        X_filled = X_std.copy()
        X_filled[np.isnan(X_raw)] = 0.0
        mask_std = (~np.isnan(X_raw)).astype(float)
        latent_dim = min(64, max(8, X_std.shape[1] // 4))
        vae_model = train_vae_imputer_full(X_filled, mask_std, latent_dim=latent_dim,
                                           epochs=args.epochs, batch_size=args.bs, lr=args.lr, device=DEVICE)
        vae_imps_std = sample_vae_imputations(vae_model, X_filled, mask_std, M_samples=M_IMPUTATIONS, device=DEVICE)
        for imp_std in vae_imps_std:
            imp_unscaled = scaler.inverse_transform(imp_std)
            vae_imputations.append(imp_unscaled)

    # Graph smoothing
    print("Applying graph smoothing refinement...")
    smoothed_imps = []
    for Ximp in vae_imputations:
        sm = patient_graph_smoothing(Ximp, mask, n_neighbors=10, alpha=GRAPH_ALPHA, n_iter=10)
        smoothed_imps.append(sm)
    # also include KNN baseline
    smoothed_imps.append(X_knn_unscaled)

    # aggregate mean & var
    Imps = np.stack(smoothed_imps, axis=0)
    mean_imp = Imps.mean(axis=0)
    var_imp = Imps.var(axis=0)
    # preserve observed entries
    mean_imp[mask == 1] = X_raw[mask == 1]
    var_imp[mask == 1] = 0.0

    # Save HDF5 preproc
    print("Saving HDF5:", OUT_H5)
    dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(str(OUT_H5), "w") as f:
        f.create_dataset("patient_id", data=np.array(patient_ids, dtype="S"), dtype=dt)
        f.create_dataset("feature_names", data=np.array(feature_names, dtype="S"), dtype=dt)
        f.create_dataset("features_mean", data=mean_imp.astype(np.float32))
        f.create_dataset("features_var", data=var_imp.astype(np.float32))
        f.create_dataset("missing_mask", data=mask.astype(np.int8))
        f.create_dataset("n_imputations", data=np.array([Imps.shape[0]], dtype=np.int32))
        f.create_dataset("imputation_samples", data=Imps[:min(5,Imps.shape[0])].astype(np.float32))

    # Save preproc objects
    save_obj = {
        "feature_names": feature_names,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "median_imputer": median_imp,
        "scaler": scaler,
        "knn_imputer": knn,
        "vae_latent": latent_dim if (not args.no_vae and TORCH) else None
    }
    joblib.dump(save_obj, str(OBJ_STORE))
    if not args.no_vae and TORCH and vae_model is not None:
        torch.save(vae_model.state_dict(), str(VAE_WEIGHTS))

    # ---------- Embeddings ----------
    # Build input matrix: concat mean, var, mask -> (n, 3*d)
    X_in = np.concatenate([mean_imp, var_imp, mask.astype(np.float32)], axis=1)
    scaler_emb = StandardScaler()
    Xs = scaler_emb.fit_transform(X_in)

    if args.mode == "pca" or not TORCH:
        pca_obj = PCA(n_components=min(EMBED_DIM, Xs.shape[1]), random_state=SEED)
        Z = pca_obj.fit_transform(Xs)
        if Z.shape[1] < EMBED_DIM:
            pad = np.zeros((Z.shape[0], EMBED_DIM - Z.shape[1]), dtype=Z.dtype)
            Z = np.concatenate([Z, pad], axis=1)
        ae_model = None
    else:
        print("Training denoising autoencoder for embeddings...")
        ae_model, Z = train_autoencoder(Xs, epochs=args.embed_epochs, batch_size=args.embed_bs, lr=args.embed_lr, device=DEVICE)
        if TORCH and ae_model is not None:
            torch.save(ae_model.state_dict(), str(OUT_DIR / "pathological_embedding_ae.pt"))

    # Save embeddings HDF5
    print("Saving embeddings:", EMB_H5)
    with h5py.File(str(EMB_H5), "w") as f:
        f.create_dataset("patient_id", data=np.array(patient_ids, dtype="S"), dtype=dt)
        f.create_dataset("embedding_512", data=Z.astype(np.float32))
        f.create_dataset("orig_feature_names", data=np.array(feature_names, dtype="S"), dtype=dt)
        f.create_dataset("n_samples", data=np.array([n], dtype=np.int32))

    # Summary JSON
    summary = {
        "n_patients": int(n),
        "n_features": int(n_features),
        "feature_names_count": { "total": len(feature_names) },
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "imputations_used": int(Imps.shape[0]),
        "vae_used": not args.no_vae and TORCH,
        "saved_files": {
            "preproc_h5": str(OUT_H5),
            "embeddings_h5": str(EMB_H5),
            "preproc_objects": str(OBJ_STORE),
            "vae_weights": str(VAE_WEIGHTS) if (not args.no_vae and TORCH) else None
        }
    }
    with open(str(SUMMARY_JSON), "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2)

    print("Preprocessing complete. Summary saved to:", SUMMARY_JSON)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_vae", action="store_true", help="skip VAE imputer")
    parser.add_argument("--epochs", type=int, default=VAE_EPOCHS)
    parser.add_argument("--bs", type=int, default=VAE_BS)
    parser.add_argument("--lr", type=float, default=VAE_LR)
    parser.add_argument("--mode", choices=["auto","pca"], default="auto")
    parser.add_argument("--embed_epochs", type=int, default=40)
    parser.add_argument("--embed_bs", type=int, default=64)
    parser.add_argument("--embed_lr", type=float, default=1e-3)
    args = parser.parse_args()
    pipeline(args)
