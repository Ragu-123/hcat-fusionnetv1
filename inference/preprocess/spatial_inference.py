# save as example-algorithm/preprocess/spatial_inference.py

"""
Handles the preprocessing and embedding generation for a single patient's spatial WSI data.

**Correct Inference Logic Explained:**

1.  **Separate Model Files:** This script now correctly loads the weights for the `PositionalMLP` and `SpatialTransformerAggregator` from their respective `.pt` files (`positional_mlp.pt` and `spatial_transformer_aggregator.pt`) located in the `resources` directory, just as they are saved by the training script.

2.  **On-the-Fly Projector:** The `PatchProjector` is a special case. Its input dimension depends on the feature dimension of the specific WSI file being processed. Therefore, it's not saved as a single pre-trained model but is created dynamically for each patient, just as it is in the training script.

3.  **Joblib for Arguments:** The `spatial_preproc_objects.joblib` file contains the arguments (`args`) used during the original preprocessing run. While not strictly needed for inference calculations, it's good practice to be aware of it for ensuring consistency.

4.  **Full Replication:** All other steps, including patch reduction via `MiniBatchKMeans`, feature/coordinate normalization, and Monte-Carlo dropout, are identical to the logic in your `spatial.py` training script to ensure the embeddings are generated in exactly the same way.
"""

import torch
import torch.nn as nn
import numpy as np
import joblib
from sklearn.cluster import MiniBatchKMeans

# --- Model component definitions, copied exactly from `spatial.py` ---

class PositionalMLP(nn.Module):
    """Projects 2D coordinates to model_dim and produces a learned positional bias."""
    def __init__(self, model_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim)
        )
    def forward(self, coords_norm):
        return self.mlp(coords_norm)

class PatchProjector(nn.Module):
    """Projects patch features to model_dim."""
    def __init__(self, feat_dim, model_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, model_dim), nn.GELU(), nn.LayerNorm(model_dim)
        )
    def forward(self, x):
        return self.proj(x)

class SpatialTransformerAggregator(nn.Module):
    """Transformer aggregator that uses a CLS token to generate a global embedding."""
    def __init__(self, model_dim=512, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=n_heads, dim_feedforward=model_dim*4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)

    def forward(self, patch_embs, attn_mask=None):
        b = patch_embs.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, patch_embs], dim=1)
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        return x[:, 0, :] # Return the CLS token embedding

def get_spatial_embedding(primary_tumor_wsi, lymph_node_wsi, resources_path, device="cpu"):
    """
    Generates a 512-d embedding from WSI patch data for a single patient.
    """
    print("Starting spatial data preprocessing...")

    # 1. Combine features from primary tumor and lymph node WSI files.
    all_feats, all_coords = [], []
    if primary_tumor_wsi and primary_tumor_wsi.get('features'):
        all_feats.append(np.array(primary_tumor_wsi['features'], dtype=np.float32))
        all_coords.append(np.array(primary_tumor_wsi['coords'], dtype=np.float32))
    if lymph_node_wsi and lymph_node_wsi.get('features'):
        all_feats.append(np.array(lymph_node_wsi['features'], dtype=np.float32))
        all_coords.append(np.array(lymph_node_wsi['coords'], dtype=np.float32))

    if not all_feats:
        print("Warning: No spatial WSI features found. Returning a zero vector.")
        return torch.zeros((1, 512))

    feats = np.concatenate(all_feats)
    coords = np.concatenate(all_coords)
    n_patches, feat_dim = feats.shape
    print(f"Total patches from all sources: {n_patches}")

    # 2. Normalize features (L2) and coordinates (Min-Max).
    fnorm = np.linalg.norm(feats, axis=1, keepdims=True); fnorm[fnorm == 0] = 1.0
    feats_norm = feats / fnorm
    
    c_min, c_max = coords.min(axis=0), coords.max(axis=0); span = c_max - c_min
    span[span == 0] = 1.0
    coords_norm = (coords - c_min) / span

    # 3. Cluster to reduce patches if necessary, matching the training logic.
    max_patches = 1024
    if n_patches > max_patches:
        print(f"Clustering {n_patches} patches down to {max_patches}...")
        cluster_data = np.concatenate([feats_norm, coords_norm], axis=1)
        kmeans = MiniBatchKMeans(n_clusters=max_patches, random_state=42, n_init='auto')
        kmeans.fit(cluster_data)
        sel_feats = kmeans.cluster_centers_[:, :feat_dim].astype(np.float32)
        sel_coords_norm = kmeans.cluster_centers_[:, feat_dim:].astype(np.float32)
    else:
        sel_feats, sel_coords_norm = feats_norm, coords_norm

    # 4. Load the trained models from the correct files in the resources directory.
    model_dim = 512
    pos_mlp = PositionalMLP(model_dim).to(device)
    aggregator = SpatialTransformerAggregator(model_dim=model_dim).to(device)
    
    pos_mlp.load_state_dict(torch.load(resources_path / "positional_mlp.pt", map_location=device))
    aggregator.load_state_dict(torch.load(resources_path / "spatial_transformer_aggregator.pt", map_location=device))

    # The PatchProjector is created dynamically based on the input feature dimension.
    projector = PatchProjector(feat_dim, model_dim).to(device)
    projector.eval() # No training for projector, just a transformation.
    pos_mlp.eval()
    
    # 5. Prepare tensors and inject spatial information.
    with torch.no_grad():
        patch_features_tensor = torch.from_numpy(sel_feats).float().to(device)
        projected_patches = projector(patch_features_tensor)
        
        coords_tensor = torch.from_numpy(sel_coords_norm).float().unsqueeze(0).to(device)
        positional_bias = pos_mlp(coords_tensor).squeeze(0)
        
        spatially_aware_patches = (projected_patches + positional_bias).unsqueeze(0)

    # 6. Perform Monte-Carlo dropout for a more robust embedding.
    mc_samples = 16
    aggregator.train() # Set to train mode to enable dropout.
    
    mc_embeddings = []
    with torch.no_grad():
        for _ in range(mc_samples):
            emb = aggregator(spatially_aware_patches)
            mc_embeddings.append(emb.cpu().numpy().flatten())
            
    # 7. Average the results to get the final, stable embedding.
    final_embedding_np = np.stack(mc_embeddings).mean(axis=0)
    final_embedding = torch.from_numpy(final_embedding_np).float().unsqueeze(0)

    print(f"Successfully generated spatial embedding with shape: {final_embedding.shape}")
    return final_embedding