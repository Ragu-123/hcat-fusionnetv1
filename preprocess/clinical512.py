# #!/usr/bin/env python3
# """
# clinical_make_embedding_512.py

# Load clinical_preprocessed_advanced.h5 -> build input (mean, var, mask) -> standardize ->
# create 512-d embeddings (denoising AE if torch available, else PCA) -> save clinical_embedding_512.h5.

# Also evaluates each task separately (accuracy & F1 on known labels).
# """

import os
from pathlib import Path
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import argparse
import joblib
import warnings
warnings.filterwarnings("ignore")

# Optional PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH = True
except Exception:
    TORCH = False

BASE = Path(r"C:\Users\haris\Downloads\project phase-1\training_dataset\clinical")
IN_H5 = BASE / "clinical_preprocessed_advanced.h5"
OUT_H5 = BASE / "clinical_embedding_512.h5"
SCHEMA_OBJ = BASE / "clinical_embedding_preproc.joblib"
AE_WEIGHTS = BASE / "clinical_embedding_ae.pt"

EMBED_DIM = 512
SEED = 42
DEVICE = "cuda" if (TORCH and torch.cuda.is_available()) else "cpu"

def load_preproc(h5path):
    if not h5path.exists():
        raise FileNotFoundError(f"Preprocessed file not found: {h5path}")
    with h5py.File(str(h5path), "r") as f:
        pid = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in f["patient_id"][:]]
        feat_names = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in f["feature_names"][:]]
        X_mean = f["features_mean"][:].astype(np.float32)
        X_var = f["features_var"][:].astype(np.float32)
        mask = f["missing_mask"][:].astype(np.int8)
        surv = f["surv_5yr_label"][:].astype(np.int8)
        rec = f["rec_2yr_label"][:].astype(np.int8)
    return np.array(pid), feat_names, X_mean, X_var, mask, surv, rec

# Simple AE definition (same shape as earlier scripts)
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
        import torch
        torch.manual_seed(SEED)
        inp_dim = X.shape[1]
        model = DenoisingAE(inp_dim=inp_dim, bottleneck=EMBED_DIM).to(device)
        X_t = torch.from_numpy(X).float().to(device)
        dl = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=True, drop_last=False)
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
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * batch.size(0)
            tot /= len(X)
            if ep % 10 == 0 or ep == 1:
                print(f" AE epoch {ep}/{epochs} loss={tot:.6f}")
        model.eval()
        with torch.no_grad():
            Z = model(torch.from_numpy(X).float().to(device))[1].cpu().numpy()
        return model, Z

def make_pca_embeddings(Xs, out_dim=EMBED_DIM):
    d = Xs.shape[1]
    n_comp = min(d, out_dim)
    pca = PCA(n_components=n_comp, random_state=SEED)
    Z = pca.fit_transform(Xs)
    if n_comp < out_dim:
        pad = np.zeros((Z.shape[0], out_dim - n_comp), dtype=Z.dtype)
        Z = np.concatenate([Z, pad], axis=1)
    return pca, Z

def evaluate_task(Z, labels, name="task"):
    mask = labels != -1
    if mask.sum() == 0:
        print(f"[{name}] No known labels to evaluate.")
        return None
    Xk, yk = Z[mask], labels[mask]
    try:
        Xtr, Xte, ytr, yte = train_test_split(Xk, yk, test_size=0.2, random_state=SEED, stratify=yk)
    except Exception:
        Xtr, Xte, ytr, yte = train_test_split(Xk, yk, test_size=0.2, random_state=SEED)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)
    acc = accuracy_score(yte, ypred)
    f1 = f1_score(yte, ypred, average='binary', zero_division=0)
    print(f"[{name}] known={len(yk)} test={len(yte)} acc={acc:.4f} f1={f1:.4f}")
    return {"acc": acc, "f1": f1, "n_known": len(yk)}

def main(args):
    print("Loading preprocessed file:", IN_H5)
    pid, feat_names, X_mean, X_var, mask, surv, rec = load_preproc(IN_H5)
    print("Patients:", len(pid), "features:", len(feat_names))

    # Build input: mean, var, mask (as float)
    X_in = np.concatenate([X_mean, X_var, mask.astype(np.float32)], axis=1)
    print("Input matrix shape:", X_in.shape)

    # Standardize
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_in)

    # Generate embeddings
    if args.mode == "pca" or not TORCH:
        print("Using PCA (fast fallback) to create 512-d embeddings.")
        pca_obj, Z = make_pca_embeddings(Xs, out_dim=EMBED_DIM)
        embeddings = Z.astype(np.float32)
        joblib.dump({"scaler": scaler, "pca": pca_obj, "feat_names": feat_names}, str(SCHEMA_OBJ))
    else:
        print("Training denoising autoencoder to create 512-d embeddings (may take minutes)...")
        model, Z = train_autoencoder(Xs, epochs=args.epochs, batch_size=args.bs, lr=args.lr, device=DEVICE)
        embeddings = Z.astype(np.float32)
        torch.save(model.state_dict(), str(AE_WEIGHTS))
        joblib.dump({"scaler": scaler, "feat_names": feat_names}, str(SCHEMA_OBJ))

    # Evaluate each task separately
    print("Evaluating tasks separately on known labels:")
    res_surv = evaluate_task(embeddings, surv, name="5yr_survival")
    res_rec = evaluate_task(embeddings, rec, name="2yr_recurrence")

    # Save embedding HDF5
    print("Saving embeddings HDF5 to:", OUT_H5)
    dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(str(OUT_H5), "w") as f:
        f.create_dataset("patient_id", data=np.array(pid, dtype="S"), dtype=dt)
        f.create_dataset("embedding_512", data=embeddings)
        f.create_dataset("surv_5yr_label", data=surv)
        f.create_dataset("rec_2yr_label", data=rec)
        f.create_dataset("surv_known_mask", data=(surv != -1).astype(np.int8))
        f.create_dataset("rec_known_mask", data=(rec != -1).astype(np.int8))
        f.create_dataset("orig_feature_names", data=np.array(feat_names, dtype="S"), dtype=dt)
    print("Saved:", OUT_H5)
    print("Saved preproc/encoder schema to:", SCHEMA_OBJ)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["auto","pca"], default="auto")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
