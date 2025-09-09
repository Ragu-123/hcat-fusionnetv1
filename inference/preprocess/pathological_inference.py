# save as example-algorithm/preprocess/pathological_inference.py

"""
Handles the preprocessing and embedding generation for a single patient's pathological data.

**Inference Logic Explained:**

1.  **Why we use `.joblib` files:** The `pathological_preproc_objects.joblib` file is critical. It stores the `SimpleImputer` (median imputer) and `StandardScaler` that were fitted on the *entire training dataset*. To process a new patient correctly, we must use these exact "pre-trained" tools to ensure the new data is imputed and scaled in a way that is consistent with the original dataset. We only call the `.transform()` method, not `.fit_transform()`.

2.  **Why we use `pathological_embedding_ae.pt`:** This file holds the trained weights of the Denoising Autoencoder. This model's specific purpose is to take the final, preprocessed feature set for a patient and compress it into a meaningful 512-dimensional vector (the embedding). This is the final step of our feature extraction.

3.  **Why we DON'T use `pathological_vae.pt`:** The VAE (Variational Autoencoder) model was used during the *training data preparation phase* to perform a very complex, probabilistic imputation across all patients at once. It learns patterns from the entire dataset to make intelligent guesses for missing values. This process is not suitable for a single patient in an inference setting. For inference, we use the much simpler and faster `SimpleImputer` that was saved in the `.joblib` file.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib

# Denoising Autoencoder for generating the final embedding.
# This class definition is taken from your training script (`pathological.py`) to ensure perfect compatibility.
class DenoisingAE(nn.Module):
    def __init__(self, inp_dim, bottleneck=512):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(inp_dim, max(512, inp_dim//2)), nn.GELU(),
            nn.Linear(max(512, inp_dim//2), 1024), nn.GELU(),
            nn.Linear(1024, bottleneck)
        )
        self.dec = nn.Sequential(
            nn.Linear(bottleneck, 1024), nn.GELU(),
            nn.Linear(1024, max(512, inp_dim//2)), nn.GELU(),
            nn.Linear(max(512, inp_dim//2), inp_dim)
        )

    def forward(self, x):
        # In inference, we only need the encoder to produce the embedding
        return self.enc(x)

def clean_numeric_string(x):
    """
    Helper function to clean string-based numeric values, like '<0.1'.
    This logic is copied directly from your `pathological.py` script.
    """
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float)): return x
    s = str(x).strip()
    if s.startswith("<"):
        try: return float(s[1:]) / 2.0
        except: return np.nan
    try: return float(s)
    except: return np.nan

def get_pathological_embedding(patho_data, resources_path):
    """
    Generates a 512-dimensional embedding from pathological data for a single patient.

    Args:
        patho_data (dict): The JSON data for one patient.
        resources_path (Path): The path to the 'resources' directory.

    Returns:
        torch.Tensor: A (1, 512) tensor representing the patient's pathological embedding.
    """
    print("Starting pathological data preprocessing...")

    # 1. Load pre-fitted objects and feature names from the joblib file.
    preproc_objects = joblib.load(resources_path / "pathological_preproc_objects.joblib")
    scaler = preproc_objects['scaler']
    median_imputer = preproc_objects['median_imputer']
    feature_names = preproc_objects['feature_names']
    
    # 2. Convert patient data to a DataFrame and perform initial cleaning.
    df = pd.DataFrame([patho_data])
    if "closest_resection_margin_in_cm" in df.columns:
        df["closest_resection_margin_in_cm_clean"] = df["closest_resection_margin_in_cm"].apply(clean_numeric_string)

    # 3. Reconstruct the feature DataFrame to match the training format.
    feat_df = pd.DataFrame(columns=feature_names, index=df.index)

    for col in feature_names:
        # Direct match from input data
        if col in df.columns:
            feat_df[col] = pd.to_numeric(df[col], errors='coerce')
        # Handle special cleaned columns
        elif col == "closest_resection_margin_in_cm_clean" and "closest_resection_margin_in_cm_clean" in df.columns:
             feat_df[col] = df["closest_resection_margin_in_cm_clean"]
        # For frequency encoded columns, we can't compute frequency for one sample.
        # We will insert NaN and let the median imputer handle it.
        elif col.startswith("freq__"):
             feat_df[col] = np.nan
        # Create missingness indicators based on the original data
        elif col.startswith("miss__"):
            original_col = col.replace("miss__", "")
            if original_col in df.columns:
                feat_df[col] = df[original_col].isna().astype(int)
            else:
                 feat_df[col] = 1 # Mark as missing if column wasn't in the input
    
    print(f"Constructed feature dataframe with {feat_df.shape[1]} columns.")

    X_raw = feat_df.values.astype(float)
    mask = (~np.isnan(X_raw)).astype(int)

    # 4. Impute and Scale using the loaded, pre-fitted objects
    X_imputed = median_imputer.transform(X_raw)
    X_scaled = scaler.transform(X_imputed)
    
    # 5. Build the final input matrix for the Denoising AE
    # The AE was trained on the concatenation of [mean_features, variance_features, mask].
    # In inference, we have no variance from multiple imputations, so we use zeros.
    X_in = np.concatenate([X_scaled, np.zeros_like(X_scaled), mask.astype(np.float32)], axis=1)
    print(f"Final input shape for embedding model: {X_in.shape}")

    # 6. Load the trained Denoising AE model
    inp_dim = X_in.shape[1]
    embedding_model = DenoisingAE(inp_dim=inp_dim, bottleneck=512)
    embedding_model.load_state_dict(torch.load(
        resources_path / "pathological_embedding_ae.pt",
        map_location=torch.device('cpu')
    ))
    embedding_model.eval()

    # 7. Generate and return the final 512-dimensional embedding
    with torch.no_grad():
        embedding = embedding_model(torch.from_numpy(X_in).float())

    print(f"Successfully generated pathological embedding with shape: {embedding.shape}")
    return embedding