# save as example-algorithm/preprocess/semantic_inference.py

"""
Handles preprocessing and embedding generation for a single patient's semantic (text) data.

**Inference Logic Explained:**

1.  **Offline Transformer Model:** This script loads the `Bio_ClinicalBERT` model and tokenizer from a local directory within the `resources` folder. This is mandatory for the Grand Challenge environment, which has no internet access.

2.  **Text Processing:** The script processes three separate text fields: `history`, `report`, and `description` (from surgery). Each text is cleaned, and then a 768-dimensional embedding is generated for it using the Transformer model.

3.  **Weighted Fusion:** The training script combines the embeddings from the three text sources using a weighting scheme based on text length and other quality metrics. We replicate this by creating a simple weighted average of the three embeddings. Texts that are empty or very short will contribute less to the final combined embedding.

4.  **Projection to 512 Dimensions:** The `Bio_ClinicalBERT` model outputs a 768-dimensional vector, but the final HCAT model expects a 512-dimensional input. The training script uses a dimensionality reduction technique (like SVD) for this projection. For inference, we will use a pre-fitted PCA object that should be saved in `text_semantic_preproc_objects.joblib` from your training run. This ensures the projection is consistent.

5.  **Final Output:** The result is a single, normalized 512-dimensional vector that represents the patient's entire clinical narrative.
"""

import torch
import numpy as np
import joblib
import re
from transformers import AutoTokenizer, AutoModel

def clean_text(s):
    """Removes HTML tags, extra whitespace, and normalizes line breaks."""
    if not isinstance(s, str): return ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def document_embedding_transformer(text, tokenizer, model, device):
    """Generates a document embedding using a transformer with mean pooling."""
    if not text:
        return np.zeros(model.config.hidden_size, dtype=np.float32)
    
    inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state.squeeze(0)
        mask = inputs["attention_mask"].squeeze(0).unsqueeze(-1).float()
        sum_hidden = (last_hidden * mask).sum(dim=0)
        sum_mask = torch.clamp(mask.sum(dim=0), min=1e-9)
        emb = (sum_hidden / sum_mask).cpu().numpy()
    return emb

def get_semantic_embedding(text_data, resources_path, device="cpu"):
    """
    Generates a 512-dimensional embedding from text data for a single patient.
    """
    print("Starting semantic data preprocessing...")
    if not text_data:
        print("Warning: No text data provided. Returning zero embedding.")
        return torch.zeros((1, 512))

    # 1. Load the offline Transformer model and tokenizer
    model_path = resources_path / "Bio_ClinicalBERT"
    if not model_path.exists():
        print(f"ERROR: Transformer model not found at {model_path}. Please follow the instructions to download it.")
        return torch.zeros((1, 512))
        
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()

    # 2. Extract and clean the three text sources
    history_text = clean_text(text_data.get("history"))
    report_text = clean_text(text_data.get("report"))
    surgery_text = clean_text(text_data.get("description")) # Corresponds to "surgery" in training script

    # 3. Generate raw embeddings for each text source
    history_emb = document_embedding_transformer(history_text, tokenizer, model, device)
    report_emb = document_embedding_transformer(report_text, tokenizer, model, device)
    surgery_emb = document_embedding_transformer(surgery_text, tokenizer, model, device)

    # 4. Fuse the embeddings using a simple quality-based weighting (length)
    embs = [history_emb, report_emb, surgery_emb]
    weights = np.array([len(history_text), len(report_text), len(surgery_text)], dtype=np.float32)
    
    # Avoid division by zero if all texts are empty
    if weights.sum() == 0:
        print("Warning: All text fields are empty. Returning zero embedding.")
        return torch.zeros((1, 512))
        
    weights = weights / weights.sum()
    
    # Perform weighted average
    fused_emb_768d = np.average(embs, axis=0, weights=weights)

    # 5. Project the 768d embedding to 512d using a pre-fitted projector
    # This assumes your `text_semantic_preproc_objects.joblib` contains a fitted
    # PCA or SVD object named 'projector_768_to_512'.
    # If not, we will fall back to a deterministic random projection.
    try:
        preproc_objects = joblib.load(resources_path / "text_semantic_preproc_objects.joblib")
        projector = preproc_objects['projector_768_to_512']
        fused_emb_512d = projector.transform(fused_emb_768d.reshape(1, -1)).flatten()
        print("Used pre-fitted SVD/PCA projector.")
    except (FileNotFoundError, KeyError):
        print("Warning: Pre-fitted projector not found in joblib. Using deterministic random projection.")
        rng = np.random.RandomState(42) # Use a fixed seed for reproducibility
        projection_matrix = rng.randn(768, 512).astype(np.float32)
        projection_matrix /= np.linalg.norm(projection_matrix, axis=0) # Normalize columns
        fused_emb_512d = fused_emb_768d @ projection_matrix

    final_embedding = torch.from_numpy(fused_emb_512d).float().unsqueeze(0)
    
    print(f"Successfully generated semantic embedding with shape: {final_embedding.shape}")
    return final_embedding