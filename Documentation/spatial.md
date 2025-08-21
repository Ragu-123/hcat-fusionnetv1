# Spatial Preprocessing Pipeline (`spatial_preproc_to_512.py`)

This document details the **spatial preprocessing pipeline** for whole slide image (WSI) patch embeddings into patient-level **512-dimensional embeddings**. The method leverages a **spatially-aware Transformer aggregator**, representative **patch clustering**, and **Monte-Carlo dropout** for uncertainty estimation.

---

## **1. Input Data**

* **Source:** Patch-level `.h5` files generated from WSIs.
* **Contents of each file:**

  * `features`: Patch-level embeddings `(n_patches, feat_dim)`.
  * `coords`: Corresponding spatial coordinates `(n_patches, 2)`.
* **Fallbacks:**

  * If feature/coordinate keys differ, script attempts alternatives (`feature`, `patch_features`, `embeddings`, etc.).
  * If coordinates are missing, dummy grid-based indices are generated.

---

## **2. File Discovery**

* Scans multiple WSI directories (`lymph`, `cup`, `larynx`, `oral`, `oroph1`, `oroph2`, `hypo`).
* Collects all `.h5` patch files into a processing list.

---

## **3. Preprocessing Steps**

### **3.1 Patch Feature Normalization**

* L2-normalization applied to each patch embedding:

  ```math
  f_i = \frac{f_i}{\|f_i\|_2}
  ```
* Ensures stability for attention-based models and clustering.

### **3.2 Coordinate Normalization**

* Patch coordinates are min-max normalized per-slide:

  ```math
  c_i^{norm} = \frac{c_i - c_{min}}{c_{max} - c_{min}}
  ```
* Scale is consistent across `[0, 1]`.

### **3.3 Patch Reduction via Clustering**

* If patches exceed `max_patches` (default = 1024):

  * Perform **MiniBatchKMeans** clustering on concatenated `[features + coords]`.
  * Representative cluster centers are selected as reduced patch set.
* Retains both **feature** and **spatial** similarity.

### **3.4 Quality Score Calculation**

```
Q = selected_patches / original_patches
```



## **4. Embedding Aggregation**

### **4.1 Feature Projection**

* Patch embeddings are projected into `model_dim=512` using:

  * `PatchProjector`: Linear + GELU + LayerNorm.

### **4.2 Spatial Bias Injection**

* Positional MLP maps normalized `(x,y)` coordinates → `512-dim bias`.
* Added directly to projected patch embeddings.

### **4.3 Transformer Aggregator**

* Spatial Transformer with CLS token:

  * Input: `(1, n, 512)` patch embeddings.
  * CLS token prepended.
  * Transformer Encoder layers (default: 4 layers, 8 heads).
  * Output: CLS token as **global WSI embedding**.

### **4.4 Monte-Carlo Dropout (Uncertainty Estimation)**

* During inference, dropout layers remain active.
* Perform `T` stochastic forward passes (default `T=16`).
* For each WSI:

  * Compute **mean embedding** across passes.
  * Compute **variance embedding** as uncertainty estimate.

### **4.5 CPU Fallback (No PyTorch)**

* Mean pooling + random projection to 512 dims.
* Provides approximate embeddings without transformer.

---

## **5. Outputs**

### **HDF5 (`spatial_embeddings_512.h5`)**

* `patient_id`: Patient identifiers.
* `embedding_mean_512`: Mean embedding vector.
* `embedding_var_512`: Variance (uncertainty) vector.
* `n_patches`: Original patch count.
* `quality_score`: Fraction of patches retained.
* `selected_patch_coords`: Retained patch coordinates.

### **JSON (`spatial_summary.json`)**

* Number of slides processed.
* Number of failed samples.
* Maximum patches retained.
* Transformer model parameters.
* Notes on methodology.

### **Joblib (`spatial_preproc_objects.joblib`)**

* Stores preprocessing arguments for reproducibility.

---

## **6. Key Notes**

* Clustering ensures computational efficiency without major information loss.
* Positional encoding enforces **spatial awareness**.
* Transformer CLS token provides robust global representation.
* Monte-Carlo dropout offers uncertainty-aware embeddings.

---

## **7. Example Workflow**

```bash
python spatial_preproc_to_512.py \
  --wsi_lymph ./dataset/lymph_h5 \
  --wsi_cup ./dataset/cup_h5 \
  --outdir ./training_dataset/spatial \
  --max_patches 1024 \
  --model_dim 512 \
  --n_layers 4 --n_heads 8 \
  --dropout 0.1 --mc_samples 16 --use_gpu
```

---

## **8. Applications**

* Patient-level spatial embedding generation.
* Downstream survival analysis, clustering, or prediction tasks.
* Uncertainty-aware WSI representations for robust clinical ML models.
