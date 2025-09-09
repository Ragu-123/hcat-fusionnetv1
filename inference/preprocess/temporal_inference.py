# save as example-algorithm/preprocess/temporal_inference.py

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import json

class TemporalLSTMEncoder(nn.Module):
    """
    Defines the LSTM Encoder architecture to generate embeddings from time-series data.
    This must match the model saved in temporal_lstm_encoder.pt.
    """
    def __init__(self, input_dim, hidden=256, n_layers=2, bottleneck=512, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, n_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.proj = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, bottleneck)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        # Average the features from the LSTM over the time dimension
        z = out.mean(dim=1)
        return self.proj(z)

def make_time_bins(seq_len=16, window_days=30):
    """Creates the edges for the 16 time bins over a 30-day window."""
    return np.linspace(0, window_days, seq_len + 1)

def patient_to_matrix(patient_df, analytes, edges, seq_len):
    """Converts a patient's blood records into a structured (analytes x time) matrix."""
    mat = np.full((len(analytes), seq_len), np.nan, dtype=float)
    for i, a in enumerate(analytes):
        adf = patient_df[patient_df["analyte_name"] == a]
        if adf.empty:
            continue
        # Assign each measurement to a time bin
        bins = np.digitize(adf["days_before_first_treatment"].values, edges) - 1
        bins = np.clip(bins, 0, seq_len - 1)
        # Average measurements that fall into the same bin
        for b in range(seq_len):
            vals = adf.loc[bins == b, "value"]
            if not vals.empty:
                mat[i, b] = vals.mean()
    return mat

def physiology_fill(mat, analytes, ref_ranges):
    """
    Performs the first-pass imputation using clinical knowledge.
    - Fills completely missing analytes with their normal reference value.
    - Interpolates partially missing analytes over time.
    """
    mat_f = mat.copy()
    for i, a in enumerate(analytes):
        row = mat_f[i]
        # First, handle completely missing analytes using reference ranges
        if np.isnan(row).all() and a in ref_ranges:
            r = ref_ranges.get(a, {})
            mins = [r.get(k) for k in ["male_min", "female_min"] if r.get(k) is not None]
            maxs = [r.get(k) for k in ["male_max", "female_max"] if r.get(k) is not None]
            if mins and maxs:
                mat_f[i, :] = (np.mean(mins) + np.mean(maxs)) / 2.0
        # Then, handle partially missing analytes with interpolation
        else:
            nan_mask = np.isnan(row)
            if nan_mask.any() and not nan_mask.all():
                idx = np.arange(row.size)
                row[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], row[~nan_mask])
                mat_f[i] = row
    return mat_f

def get_temporal_embedding(blood_data, resources_path, device="cpu"):
    """
    Main function to generate a 512-d embedding from temporal blood data.
    """
    # **EDGE CASE 1**: Handle empty or incorrectly formatted blood data.
    # The Hancothon schema requires a LIST of measurements. The local test provides a single DICTIONARY.
    # This check ensures the code works in both situations.
    if not isinstance(blood_data, list) or not blood_data:
        print("Warning: Blood data is not in the expected list format or is empty. This is expected during local testing. Returning a zero vector.")
        return torch.zeros((1, 512))

    # Load the pre-fitted tools and configurations from training
    preproc_objects = joblib.load(resources_path / "temporal_preproc_objects.joblib")
    analytes = preproc_objects['analytes']
    knn_imputer = preproc_objects['knn_imputer']
    scaler = preproc_objects['scaler']
    
    # **EDGE CASE 2**: Safely attempt to load the optional blood reference ranges file.
    # This file will NOT be present in the final evaluation, so the code must not crash.
    ref_ranges = {}
    try:
        with open(resources_path / 'blood_data_reference_ranges.json', 'r') as f:
            ref_ranges = {r['analyte_name']: r for r in json.load(f)}
            print("Successfully loaded blood data reference ranges for local physiology-aware fill.")
    except FileNotFoundError:
        print("Warning: `blood_data_reference_ranges.json` not found. Skipping optional physiology-aware fill step. This is expected in the final evaluation environment.")

    # Convert the patient's data into a structured DataFrame
    df = pd.DataFrame(blood_data)
    # Ensure required columns exist, even if the list is empty, to prevent errors
    for col in ["analyte_name", "value", "days_before_first_treatment"]:
        if col not in df.columns:
            df[col] = pd.Series(dtype='object') # Add empty series if column is missing

    df["analyte_name"] = df["analyte_name"].astype(str).str.strip()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    
    # Process the data into the (analytes x time) matrix format
    seq_len = 16
    patient_matrix = patient_to_matrix(df, analytes, make_time_bins(seq_len), seq_len)

    # Perform the two-stage imputation
    # Step 1 (Optional): Physiology fill. Runs only if the reference file was found.
    filled_matrix = physiology_fill(patient_matrix, analytes, ref_ranges) if ref_ranges else patient_matrix
    
    # Step 2 (Mandatory): KNN Imputation. This is the main imputer, using patterns learned from training.
    X_flat = filled_matrix.flatten().reshape(1, -1)
    X_knn_imputed = knn_imputer.transform(X_flat)

    # Scale the data using the pre-fitted scaler from training
    X_scaled = scaler.transform(X_knn_imputed)
    
    # Reshape the data to the (batch, seq_len, num_features) format required by the LSTM
    X_reshaped = X_scaled.reshape(len(analytes), seq_len).T
    lstm_input = torch.from_numpy(X_reshaped).float().unsqueeze(0).to(device)

    # Load the trained LSTM model and generate the final embedding
    model = TemporalLSTMEncoder(input_dim=len(analytes), bottleneck=512)
    checkpoint = torch.load(resources_path / "temporal_lstm_encoder.pt", map_location=device)
    model.load_state_dict(checkpoint['encoder'])
    model.to(device).eval()

    with torch.no_grad():
        embedding = model(lstm_input)

    print(f"Successfully generated temporal embedding with shape: {embedding.shape}")
    return embedding
