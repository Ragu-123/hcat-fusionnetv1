# save as example-algorithm/preprocess/clinical_inference.py

"""
Handles the preprocessing and embedding generation for a single patient's clinical data.

**Inference Logic Explained:**

1.  **Why we use `.joblib` files:** The `clinical_preproc_objects.joblib` file contains the `SimpleImputer` and `StandardScaler` that were *fitted on the entire training dataset*. For inference, we MUST use these exact same objects to transform the new patient's data. This ensures the new data is scaled and imputed consistently with the training data. We use `.transform()` not `.fit_transform()`.

2.  **Why we use `clinical_embedding_ae.pt`:** This file contains the weights for the final Denoising Autoencoder that was trained to convert the preprocessed, clean data into a 512-dimensional embedding. This is the model we need to generate the final feature vector for our main HCAT model.

3.  **Why we DON'T use `clinical_vae.pt`:** The VAE imputer was a tool used during the *training data preparation phase* to intelligently fill missing values across all 763 patients at once. It requires a large dataset to work effectively and is not suitable for imputing a single new patient during inference. We use the simpler, pre-fitted `SimpleImputer` from the `.joblib` file instead for this task.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib

# Denoising Autoencoder for generating the final embedding.
# This class definition must match the one used during training in `clinical512.py`.
class DenoisingAE(nn.Module):
    def __init__(self, inp_dim, bottleneck=512):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(inp_dim, max(512, inp_dim // 2)), nn.GELU(),
            nn.Linear(max(512, inp_dim // 2), 1024), nn.GELU(),
            nn.Linear(1024, bottleneck)
        )
        self.dec = nn.Sequential(
            nn.Linear(bottleneck, 1024), nn.GELU(),
            nn.Linear(1024, max(512, inp_dim // 2)), nn.GELU(),
            nn.Linear(max(512, inp_dim // 2), inp_dim)
        )

    def forward(self, x):
        # For inference, we only need the encoder part to get the embedding
        return self.enc(x)

def get_clinical_embedding(clinical_data, resources_path):
    """
    Generates a 512-dimensional embedding from clinical data for a single patient.

    Args:
        clinical_data (dict): The JSON data for one patient.
        resources_path (Path): The path to the 'resources' directory.

    Returns:
        torch.Tensor: A (1, 512) tensor representing the patient's clinical embedding.
    """
    print("Starting clinical data preprocessing...")

    # 1. Load the pre-fitted objects from the .joblib file
    # These objects were fitted on the full training dataset.
    preproc_objects = joblib.load(resources_path / "clinical_preproc_objects.joblib")
    median_imputer = preproc_objects['median_imputer']
    scaler = preproc_objects['scaler']
    feature_names = preproc_objects['feature_names']
    
    # 2. Convert single patient dictionary to a pandas DataFrame
    df = pd.DataFrame([clinical_data])

    # 3. Reconstruct the feature DataFrame exactly as in training
    feat_df = pd.DataFrame(columns=feature_names, index=df.index)

    # Populate with available data
    for col in df.columns:
        if col in feat_df.columns:
            feat_df[col] = pd.to_numeric(df[col], errors='coerce')
        # Handle frequency encoded columns: In inference, we can't calculate frequency.
        # We will rely on the median imputer to fill these.
        elif f"freq__{col}" in feat_df.columns:
             feat_df[f"freq__{col}"] = np.nan # Will be imputed later
    
    # Create missingness indicators
    for col in feat_df.columns:
        if col.startswith("miss__"):
            original_col = col.replace("miss__", "")
            if original_col in df.columns:
                 feat_df[col] = df[original_col].isna().astype(int)
            else: # If the original column is missing entirely from input
                 feat_df[col] = 1

    print(f"Constructed feature dataframe with {feat_df.shape[1]} columns.")
    
    X_raw = feat_df.values.astype(float)
    mask = (~np.isnan(X_raw)).astype(int)

    # 4. Impute and Scale using the loaded objects
    # IMPORTANT: We use .transform() here, NOT .fit() or .fit_transform()
    X_imputed = median_imputer.transform(X_raw)
    X_scaled = scaler.transform(X_imputed)
    
    # 5. Build the final input matrix for the embedding model
    # The model was trained on [mean_features, variance_features, mask]
    # For inference, we don't have imputation variance, so we use an array of zeros.
    X_in = np.concatenate([X_scaled, np.zeros_like(X_scaled), mask.astype(np.float32)], axis=1)
    print(f"Final input shape for embedding model: {X_in.shape}")

    # 6. Load the trained Denoising Autoencoder model
    inp_dim = X_in.shape[1]
    embedding_model = DenoisingAE(inp_dim=inp_dim, bottleneck=512)
    embedding_model.load_state_dict(torch.load(
        resources_path / "clinical_embedding_ae.pt",
        map_location=torch.device('cpu')
    ))
    embedding_model.eval() # Set model to evaluation mode

    # 7. Generate and return the embedding
    with torch.no_grad():
        embedding = embedding_model(torch.from_numpy(X_in).float())

    print(f"Successfully generated clinical embedding with shape: {embedding.shape}")
    return embedding