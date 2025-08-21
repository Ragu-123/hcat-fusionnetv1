# Documentation for `train.py`

This document provides a detailed explanation of the `train.py` script, which implements **Enhanced HCAT (Hierarchical Cross-modal Attention Transformer)** training with **Advanced Multi-Modal Imputation Techniques**.

---

## Overview

The script trains a multi-modal deep learning model that integrates heterogeneous biomedical data (clinical, semantic, temporal, pathological, and spatial). It addresses missing data using advanced **cross-modal imputation** strategies and produces predictions for two clinical tasks:

* **5-year survival prediction**
* **2-year recurrence prediction**

Key innovations include:

* **JAMIE-style Joint Variational Autoencoders (VAE)** for cross-modal imputation.
* **Multi-scale latent representations** with uncertainty quantification.
* **Attention-based cross-modal imputation**.
* **Iterative refinement using self-supervision**.
* **Enhanced transformer architecture** with modality dropout, contrastive learning, and uncertainty-aware weighting.

---

## Major Components

### 1. Data Handling Utilities

* **`normalize_pid`**: Normalizes patient identifiers across files.
* **`read_all_datasets_from_h5`**: Reads all datasets from HDF5 files.
* **`find_patientid_and_embedding`**: Extracts patient IDs and embeddings from HDF5 files.
* **`aggregate_spatial`**: Aggregates multiple patch-level embeddings into patient-level spatial embeddings.
* **`build_patient_embedding_map`**: Aligns embeddings with patient IDs for each modality.

These utilities ensure consistent patient-level data alignment across modalities.

---

### 2. Multi-Modal VAE (`MultiModalVAE`)

A joint variational autoencoder that:

* Encodes each modality at multiple scales.
* Uses **cross-modal attention** to share information between modalities.
* Learns a **joint latent space** for all modalities.
* Produces reconstructions, imputations for missing modalities, and **uncertainty estimates**.

Losses:

* **Reconstruction loss** (MSE for observed modalities).
* **KL divergence** for latent space regularization.

---

### 3. Iterative Imputer (`AdvancedIterativeImputer`)

* Wraps the **MultiModalVAE**.
* Performs **iterative refinement** with Transformer encoder layers.
* Updates imputations progressively.
* Uses **confidence heads** and **progressive uncertainty reduction**.
* Adds a **self-supervised consistency loss** to improve imputations.

---

### 4. Dataset Class (`MultiModalDataset`)

* Loads aligned embeddings from all modalities.
* Assembles a dictionary per patient containing:

  * Embeddings for all five modalities.
  * Quality scores.
  * Presence indicators (whether modality available).
  * Labels: survival and recurrence.

Returned sample structure:

```python
{
  "patient_id": str,
  "emb": np.ndarray (5, 512),
  "quality": np.ndarray (5,),
  "present": np.ndarray (5,),
  "targets": np.ndarray (2,)  # surv, rec labels
}
```

---

### 5. Enhanced HCAT Model (`EnhancedHCAT`)

A transformer-based model with advanced imputation:

* **Positional Encoding** (`ModalityPositionalEncoding`):
  Adds modality-specific embeddings.

* **Quality Gate** (`QualityGate`):
  Learns to weight embeddings based on modality quality.

* **Branch Transformers**:

  * Local temporal encoder (clinical + temporal).
  * Local spatial encoder (pathological + spatial).
  * Semantic projection encoder.

* **Global Transformer Encoder**: Fuses branch outputs.

* **Attention Pooling**: Aggregates token-level features.

* **Cross-modal Fusion**: Refines pooled branch embeddings.

* **Prediction Heads**:

  * `head_surv`: Binary survival prediction.
  * `head_rec`: Binary recurrence prediction.
  * Contrastive projection head (`cproj`): for representation learning.

* **Uncertainty-aware weighting**: Adjusts contributions of imputed modalities.

---

### 6. Loss Functions

* **Binary Cross-Entropy (BCE) with masking**.
* **Focal Loss**: Addresses class imbalance.
* **Contrastive Loss (InfoNCE)**: Improves representation quality.
* **KL Divergence**: Regularizes latent space of VAE.
* **Imputation Loss**: From iterative imputer.

Final loss is a weighted sum:

```
Loss = α_surv * survival_loss + α_rec * recurrence_loss
     + α_con * contrastive_loss + α_kl * KL_loss
     + α_impute * imputation_loss
```

---

### 7. Training Loop

* Splits dataset into **train/validation** sets.
* Supports **modality dropout** (regularization).
* Adds **embedding noise** for robustness.
* Optimizer: **AdamW** with **cosine annealing warm restarts**.
* Tracks metrics: Accuracy, F1-score, AUC.
* Implements **early stopping**.
* Saves best checkpoints for survival, recurrence, and average F1.
* Outputs a JSON summary with metrics and configuration.

---

### 8. Command-Line Interface (CLI)

Run the script with:

```bash
python train2.py \
  --clinical clinical.h5 \
  --semantic semantic.h5 \
  --temporal temporal.h5 \
  --pathological pathological.h5 \
  --spatial spatial.h5 \
  --outdir ./checkpoints \
  --epochs 50 --batch 32 --lr 5e-5
```

#### Important Arguments:

* **Data inputs**: `--clinical`, `--semantic`, `--temporal`, `--pathological`, `--spatial`.
* **Model params**: `--d_model`, `--n_heads`, `--n_global_layers`, `--dropout`.
* **Training params**: `--epochs`, `--batch`, `--lr`, `--device`, `--early_stop_patience`.
* **Imputation params**: `--use_advanced_imputation`, `--n_impute_iterations`.
* **Loss weights**: `--alpha_surv`, `--alpha_rec`, `--alpha_con`, `--alpha_kl`, `--alpha_impute`.
* **Augmentations**: `--moddrop`, `--moddrop_mode`, `--emb_noise`.

---

## Outputs

* **Checkpoints**: best models saved per-task and final model.
* **Training summary**: `enhanced_hcat_training_summary.json`.
* **Logs**: Validation F1/AUC printed per epoch.

---

## Summary

`train2.py` implements a state-of-the-art training pipeline for multi-modal biomedical data:

* Handles missing data with **probabilistic imputation (VAE)** + **iterative refinement**.
* Learns rich, uncertainty-aware patient embeddings.
* Predicts survival and recurrence with high robustness.
* Provides flexible CLI with configurable model/training parameters.

This makes it suitable for **clinical outcome prediction** and **multi-modal representation learning** in real-world datasets.
