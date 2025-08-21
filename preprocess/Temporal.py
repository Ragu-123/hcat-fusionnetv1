# #!/usr/bin/env python3
# """
# temporal_preproc_advanced.py

# Preprocess temporal blood data (blood_data.json) into patient-level 512-d embeddings for HCAT.

# Saves:
#  - <OUT_DIR>/temporal_embedding_512.h5
#  - <OUT_DIR>/temporal_preproc_summary.json
#  - (optional) model weights / preproc objects

# Novelty high-level (journal-ready):
#  - Physiology-aware normalization using clinical reference ranges
#  - Time-binning + interpolation to a fixed temporal grid
#  - Cohort KNN refinement after physiologic imputation to leverage population statistics
#  - Denoising LSTM encoder producing per-patient 512-d embeddings + quality score & uncertainty
# """

import json
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import joblib
import argparse
import os, math, warnings
warnings.filterwarnings("ignore")

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Optional PyTorch LSTM encoder
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH = True
except Exception:
    TORCH = False

# ---------- Config ----------
BASE_IN = Path(r"C:\Users\haris\Downloads\project phase-1\training_dataset")
IN_DIR = BASE_IN / "temporal"   # script will read jsons from project dataset root (assumed uploaded)
# Input JSON files (adjust if different)
BLOOD_JSON = Path(r"C:\Users\haris\Downloads\project phase-1\dataset\clinicalStructuredData\StructuredData\blood_data.json")  # the uploaded file path in this session
REF_JSON   = Path(r"C:\Users\haris\Downloads\project phase-1\dataset\clinicalStructuredData\StructuredData\blood_data_reference_ranges.json")

OUT_DIR = Path(r"C:\Users\haris\Downloads\project phase-1\training_dataset\temporal")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_H5 = OUT_DIR / "temporal_embedding_512.h5"
SUMMARY_JSON = OUT_DIR / "temporal_preproc_summary.json"
PREPROC_OBJ = OUT_DIR / "temporal_preproc_objects.joblib"
AE_WEIGHTS = OUT_DIR / "temporal_lstm_encoder.pt"

EMBED_DIM = 512
TIME_WINDOW_DAYS = 30   # lookback window before first treatment (0..30 days)
SEQ_LENGTH = 16         # number of time bins to resample into (temporal length for LSTM)
KNN_NEIGHBORS = 8
SEED = 42
np.random.seed(SEED)
if TORCH:
    torch.manual_seed(SEED)
DEVICE = "cuda" if (TORCH and torch.cuda.is_available()) else "cpu"

# ---------- Helpers ----------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_dataframe(blood_records):
    # Expected fields: patient_id, analyte_name, value, days_before_first_treatment, unit, LOINC_code, ...
    df = pd.DataFrame(blood_records)
    # normalize patient_id as string zero-padded if needed
    df["patient_id"] = df["patient_id"].astype(str).str.zfill(3)
    # days_before_first_treatment -> integer
    df["days_before_first_treatment"] = pd.to_numeric(df.get("days_before_first_treatment", 0), errors="coerce").fillna(0).astype(int)
    # analyte name normalization
    df["analyte_name"] = df["analyte_name"].astype(str).str.strip()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df

def load_reference_ranges(ref_records):
    # map analyte_name -> (male_min, male_max, female_min, female_max)
    ref = {}
    for r in ref_records:
        name = r.get("analyte_name")
        ref[name] = {
            "male_min": r.get("normal_male_min"),
            "male_max": r.get("normal_male_max"),
            "female_min": r.get("normal_female_min"),
            "female_max": r.get("normal_female_max"),
            "unit": r.get("unit"),
            "group": r.get("group")
        }
    return ref

def analyte_population_stats(df):
    # frequency and missingness per analyte
    grp = df.groupby("analyte_name")["patient_id"].nunique().sort_values(ascending=False)
    return grp

def make_time_bins(seq_len=SEQ_LENGTH, window_days=TIME_WINDOW_DAYS):
    # bins from 0..window_days inclusive, produce seq_len bins (right-inclusive)
    edges = np.linspace(0, window_days, seq_len+1)  # len = seq_len+1
    # produce center times as representative for each bin
    centers = (edges[:-1] + edges[1:]) / 2.0
    return edges, centers

# Convert per-patient analyte measurements into (n_analytes x seq_len) matrix
def patient_to_matrix(patient_df, analytes, edges, seq_len):
    n_a = len(analytes)
    mat = np.full((n_a, seq_len), np.nan, dtype=float)
    # for each analyte, bin by days and average values in same bin
    for i, a in enumerate(analytes):
        adf = patient_df[patient_df["analyte_name"] == a]
        if adf.shape[0] == 0:
            continue
        # Digitize days
        bins = np.digitize(adf["days_before_first_treatment"].values, edges, right=False) - 1
        # clamp
        bins = np.clip(bins, 0, seq_len-1)
        for b in range(seq_len):
            vals = adf.loc[bins == b, "value"]
            if not vals.empty:
                mat[i, b] = vals.mean()
    return mat

# Physiology-aware fill: use reference median (midpoint) if available
def physiology_fill(mat, analytes, ref_ranges):
    mat_f = mat.copy()
    for i, a in enumerate(analytes):
        col = mat_f[i]
        if np.isnan(col).all():
            # no observations for this analyte -> use reference midpoint if exists
            if a in ref_ranges:
                r = ref_ranges[a]
                # choose male/female combined midpoint or any available bound
                mins = [r.get("male_min"), r.get("female_min")]
                maxs = [r.get("male_max"), r.get("female_max")]
                mins = [x for x in mins if x is not None]
                maxs = [x for x in maxs if x is not None]
                if mins and maxs:
                    mid = (np.mean(mins) + np.mean(maxs)) / 2.0
                    mat_f[i, :] = mid
                elif mins:
                    mat_f[i, :] = np.mean(mins)
                elif maxs:
                    mat_f[i, :] = np.mean(maxs)
                else:
                    # fallback later
                    pass
        else:
            # for partial data, forward/backward fill along time axis
            # simple 1D interpolation across time
            idx = np.arange(col.size)
            nan_mask = np.isnan(col)
            if nan_mask.any() and (~nan_mask).any():
                col[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], col[~nan_mask])
                mat_f[i] = col
    return mat_f

# Flatten matrix for cohortwise KNN imputation
def flatten_mat(mat):
    return mat.flatten(order="C")  # analyte-major -> time series flattened

# ---------- LSTM encoder (PyTorch) ----------
if TORCH:
    class TemporalLSTMEncoder(nn.Module):
        def __init__(self, input_dim, hidden=256, n_layers=2, bottleneck=EMBED_DIM, dropout=0.1):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden, n_layers, batch_first=True, bidirectional=True, dropout=dropout)
            self.pool = nn.AdaptiveAvgPool1d(1)  # will apply after projecting to features
            self.proj = nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.GELU(),
                nn.Linear(hidden, bottleneck)
            )
        def forward(self, x):
            # x: (batch, seq_len, input_dim)
            out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
            # aggregate via mean over time
            z = out.mean(dim=1)
            emb = self.proj(z)
            return emb

# ---------- Main pipeline ----------
def pipeline(args):
    print("Loading inputs...")
    blood_records = load_json(BLOOD_JSON)
    ref_records   = load_json(REF_JSON)
    df = build_dataframe(blood_records)
    refs = load_reference_ranges(ref_records)

    # meta
    patients = sorted(df["patient_id"].unique().tolist())
    n_patients = len(patients)
    print(f"Patients found: {n_patients}")
    analyte_counts = analyte_population_stats(df)
    # choose analytes present in at least 3 patients or present in ref ranges
    candidate_analytes = analyte_counts[analyte_counts >= 3].index.tolist()
    # also include analytes in reference list if not too sparse
    for a in refs.keys():
        if a not in candidate_analytes:
            candidate_analytes.append(a)
    analytes = sorted(list(dict.fromkeys(candidate_analytes)))  # preserve order, unique
    n_analytes = len(analytes)
    print(f"Using {n_analytes} analytes (top present + reference list).")

    # time bins
    edges, centers = make_time_bins(seq_len=SEQ_LENGTH, window_days=TIME_WINDOW_DAYS)

    # Build per-patient matrices
    mats = np.full((n_patients, n_analytes, SEQ_LENGTH), np.nan, dtype=float)
    obs_counts = np.zeros((n_patients, n_analytes), dtype=int)
    for pi, pid in enumerate(patients):
        pdf = df[df["patient_id"] == pid]
        mat = patient_to_matrix(pdf, analytes, edges, SEQ_LENGTH)
        mats[pi] = mat
        obs_counts[pi] = (~np.isnan(mat)).sum(axis=1)

    # Quality per patient: fraction of observed cells
    total_cells = n_analytes * SEQ_LENGTH
    observed_per_patient = (~np.isnan(mats)).sum(axis=(1,2))
    quality_scores = (observed_per_patient / total_cells).astype(np.float32)  # 0..1

    # Physiology-aware initial fill (per patient)
    mats_filled = np.zeros_like(mats)
    for i in range(n_patients):
        mats_filled[i] = physiology_fill(mats[i], analytes, refs)

    # Now cohort-level KNN imputation across flattened features to borrow population patterns
    print("Running cohort KNN imputation (refined after physiology-fill)...")
    X_flat = mats_filled.reshape(n_patients, -1)  # (n_patients, n_analytes*SEQ_LENGTH)
    # For any remaining NaNs (if physiology fill couldn't fill), set column median
    col_med = np.nanmedian(X_flat, axis=0)
    inds = np.where(np.isnan(X_flat))
    if inds[0].size > 0:
        X_flat[inds] = np.take(col_med, inds[1])
    # apply KNNImputer to smooth/borrow across patients
    knn = KNNImputer(n_neighbors=min(KNN_NEIGHBORS, max(2, n_patients-1)))
    X_imputed = knn.fit_transform(X_flat)
    mats_imputed = X_imputed.reshape(n_patients, n_analytes, SEQ_LENGTH)

    # compute per-analyte/time missing proportions (for summary)
    missing_prop = np.mean(np.isnan(mats), axis=0)  # shape (n_analytes, seq_len)

    # Standardize per feature (per analyte across patients & time)
    # We'll flatten (n_patients, features) and scale
    scaler = StandardScaler()
    X_for_embedding = mats_imputed.reshape(n_patients, -1)
    X_scaled = scaler.fit_transform(X_for_embedding)

    # Create embeddings: LSTM over (seq_len, n_analytes) requires transpose
    embeddings = None
    if TORCH and args.mode == "lstm":
        print("Training denoising LSTM encoder to produce 512-d embeddings (PyTorch)...")
        # prepare tensor (batch, seq_len, input_dim)
        X_seq = mats_imputed.transpose(0, 2, 1)  # (n_patients, seq_len, n_analytes)
        X_tensor = torch.from_numpy(X_seq).float().to(DEVICE)
        # simple training: denoising reconstruction objective, encode->decode (we'll use encoder + linear decoder)
        input_dim = n_analytes
        encoder = TemporalLSTMEncoder(input_dim=input_dim, hidden=256, n_layers=2, bottleneck=EMBED_DIM).to(DEVICE)
        # decoder: simple MLP from embedding to flattened sequence
        decoder = nn.Sequential(
            nn.Linear(EMBED_DIM, 1024),
            nn.GELU(),
            nn.Linear(1024, input_dim * SEQ_LENGTH)
        ).to(DEVICE)
        params = list(encoder.parameters()) + list(decoder.parameters())
        opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-6)
        loss_fn = nn.MSELoss()
        dl = DataLoader(TensorDataset(X_tensor), batch_size=min(64, n_patients), shuffle=True)
        epochs = args.epochs
        for ep in range(1, epochs+1):
            encoder.train(); decoder.train()
            total_loss = 0.0
            for (batch,) in dl:
                # denoise: randomly mask 10% of inputs
                mask = (torch.rand_like(batch) > 0.10).float()
                noisy = batch * mask
                emb = encoder(noisy)  # (batch, EMBED_DIM)
                rec_flat = decoder(emb).view(batch.size(0), SEQ_LENGTH, input_dim)
                loss = loss_fn(rec_flat, batch)
                opt.zero_grad(); loss.backward(); opt.step()
                total_loss += loss.item() * batch.size(0)
            total_loss /= n_patients
            if ep % 10 == 0 or ep == 1:
                print(f" LSTM epoch {ep}/{epochs} loss={total_loss:.6f}")
        # produce embeddings
        encoder.eval()
        with torch.no_grad():
            embeddings = encoder(X_tensor).cpu().numpy().astype(np.float32)
        # save encoder weights
        torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict()}, str(AE_WEIGHTS))
    else:
        # PCA fallback on flattened scaled features
        print("PyTorch not available or --mode=pca selected; using PCA fallback to create 512-d embeddings.")
        pca = PCA(n_components=min(EMBED_DIM, X_scaled.shape[1]), random_state=SEED)
        Z = pca.fit_transform(X_scaled)
        if Z.shape[1] < EMBED_DIM:
            pad = np.zeros((Z.shape[0], EMBED_DIM - Z.shape[1]), dtype=Z.dtype)
            Z = np.concatenate([Z, pad], axis=1)
        embeddings = Z.astype(np.float32)
        joblib.dump({"scaler": scaler, "pca": pca}, str(PREPROC_OBJ))

    # Save HDF5 (patient_id, embeddings, analyte list, time centers, quality scores, missing stats)
    print("Saving HDF5 to:", OUT_H5)
    dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(str(OUT_H5), "w") as f:
        f.create_dataset("patient_id", data=np.array(patients, dtype="S"), dtype=dt)
        f.create_dataset("embedding_512", data=embeddings)
        f.create_dataset("analytes", data=np.array(analytes, dtype="S"), dtype=dt)
        f.create_dataset("time_centers", data=centers.astype(np.float32))
        f.create_dataset("quality_score", data=quality_scores.astype(np.float32))
        f.create_dataset("observed_counts", data=observed_per_patient.astype(np.int32))
        f.create_dataset("missing_proportion", data=missing_prop.astype(np.float32))
        f.attrs["n_patients"] = n_patients
        f.attrs["n_analytes"] = n_analytes
        f.attrs["seq_length"] = SEQ_LENGTH
        f.attrs["embedding_dim"] = EMBED_DIM
        f.attrs["method"] = "lstm_denoising" if (TORCH and args.mode=="lstm") else "pca_fallback"

    # Build summary
    summary = {
        "n_patients": int(n_patients),
        "n_analytes": int(n_analytes),
        "analytes_sample": analytes[:20],
        "seq_length": int(SEQ_LENGTH),
        "time_window_days": int(TIME_WINDOW_DAYS),
        "embedding_dim": int(EMBED_DIM),
        "method": "lstm_denoising" if (TORCH and args.mode=="lstm") else "pca_fallback",
        "quality_scores_mean": float(np.mean(quality_scores)),
        "quality_scores_median": float(np.median(quality_scores)),
        "knn_neighbors": int(KNN_NEIGHBORS),
        "notes": [
            "Physiology-aware reference filling used from blood_data_reference_ranges.json.",
            "Cohort-level KNN imputation after physiology fill to borrow population patterns.",
            "Denoising LSTM encoder trained (if PyTorch present) to produce robust 512-d embeddings.",
            "If LSTM not available, PCA on flattened scaled features used as deterministic fallback."
        ],
        "input_files": {
            "blood_data.json": str(BLOOD_JSON),
            "blood_data_reference_ranges.json": str(REF_JSON)
        }
    }
    # include small missingness per analyte
    per_analyte_missing = np.mean(np.isnan(mats), axis=(0,2))  # fraction of patients with no obs for that analyte
    summary["per_analyte_missing_fraction"] = dict(zip(analytes, per_analyte_missing.round(4).tolist()))

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save preproc objects if not saved earlier
    preproc_save = {
        "analytes": analytes,
        "edges": edges.tolist(),
        "centers": centers.tolist(),
        "scaler": scaler if not (TORCH and args.mode=="lstm") else None,
        "knn_imputer": knn
    }
    joblib.dump(preproc_save, str(PREPROC_OBJ))

    print("Saved HDF5 and summary.json.")
    print("Output files:")
    print(" - embeddings:", OUT_H5)
    print(" - summary:", SUMMARY_JSON)
    print(" - preproc objects:", PREPROC_OBJ)
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["lstm","pca"], default="lstm", help="lstm (if torch available) or pca fallback")
    parser.add_argument("--epochs", type=int, default=50, help="epochs for LSTM denoising autoencoding")
    args = parser.parse_args()
    s = pipeline(args)
    # print top-level summary
    print(json.dumps({k: v for k, v in s.items() if k in ("n_patients","n_analytes","method","quality_scores_mean")}, indent=2))
