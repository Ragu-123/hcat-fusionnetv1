# Pathological Data Preprocessing Pipeline Documentation

This document explains the functionality, workflow, and outputs of the script `pathological_preproc_novel.py`. The script is designed to preprocess pathological clinical data from a JSON file, apply advanced probabilistic imputations, perform graph smoothing, and generate high-dimensional patient embeddings. The final outputs are stored in structured formats (HDF5, JSON, Joblib).

---

## Overview

**Input:**

* A structured pathological dataset in JSON format (`pathological_data.json`).

**Process:**

1. Data loading and cleaning
2. Feature engineering (numeric + categorical)
3. Missing data imputation (KNN + Variational Autoencoder)
4. Graph-based refinement
5. Embedding generation (PCA or Autoencoder)

**Output:**

* Preprocessed features (HDF5)
* Pathological embeddings (512-dim HDF5)
* Preprocessing objects (Joblib)
* Trained VAE / Autoencoder weights (PyTorch)
* Summary metadata (JSON)

---

## Key Components

### 1. Data Loading

* Reads pathological data from JSON (`INPUT_JSON`).
* Ensures `patient_id` exists and is unique.
* Drops duplicates and indexes by `patient_id`.

### 2. Data Cleaning

* Handles numeric string anomalies (e.g., "<0.1" → 0.05).
* Coerces clinical values like infiltration depth and lymph node counts to numeric.
* Missing values are converted to `NaN`.

### 3. Feature Engineering

* **Numeric columns**: Automatically detected numeric features.
* **Categorical columns**: Encoded via frequency encoding.
* **Missingness indicators**: A binary indicator (`miss__col`) is created for each feature.

### 4. Imputation

Two approaches are used:

* **Baseline KNN Imputer** (5-nearest neighbors)
* **Probabilistic VAE Imputer** (if PyTorch available, unless `--no_vae` specified):

  * Learns latent representation of data with missing entries.
  * Samples multiple imputations (`M_IMPUTATIONS = 5`).

### 5. Graph Smoothing

* Constructs a patient similarity graph using the RBF kernel.
* Smooths imputed values over graph neighborhoods.
* Refines missing values via iterative averaging (`GRAPH_ALPHA = 0.6`).

### 6. Embedding Generation

Two modes available:

* **PCA Mode**: Reduces input matrix to 512 dimensions (pads if necessary).
* **Autoencoder Mode** (default if PyTorch available):

  * Denoising Autoencoder learns compact patient embeddings.
  * Produces 512-dimensional embeddings.

### 7. Outputs

* **Preprocessed Data (HDF5):**

  * Patient IDs
  * Feature names
  * Mean & variance-imputed feature matrix
  * Missingness mask
  * Imputation samples

* **Embeddings (HDF5):**

  * Patient IDs
  * 512-dimensional embeddings

* **Supporting Files:**

  * Preprocessing objects (`joblib` dump)
  * VAE/Autoencoder weights (`.pt` files)
  * Summary metadata (`summary.json`)

---

## Command-Line Arguments

```bash
python pathological_preproc_novel.py [options]
```

**Options:**

* `--no_vae` : Skip VAE-based imputation, fallback to KNN only.
* `--epochs` : Training epochs for VAE (default: 50).
* `--bs` : Batch size for VAE training (default: 64).
* `--lr` : Learning rate for VAE (default: 1e-3).
* `--mode {auto,pca}` : Embedding method (default: auto → Autoencoder if available, else PCA).
* `--embed_epochs` : Autoencoder training epochs (default: 40).
* `--embed_bs` : Autoencoder batch size (default: 64).
* `--embed_lr` : Autoencoder learning rate (default: 1e-3).

---

## Workflow Summary

1. **Load** pathological dataset.
2. **Clean & encode** features.
3. **Generate missingness indicators.**
4. **Impute missing values** using KNN + (optionally) VAE.
5. **Refine imputations** using patient graph smoothing.
6. **Aggregate imputations** (mean + variance).
7. **Save preprocessed dataset** in HDF5 format.
8. **Train embedding model** (PCA/Autoencoder).
9. **Save 512-d embeddings** in HDF5 format.
10. **Dump preprocessing objects** and metadata summary.

---

## Outputs in File System

The processed results are stored in:

```
C:\Users\haris\Downloads\project phase-1\training_dataset\pathological
```

Files include:

* `pathological_preprocessed_advanced.h5` → Preprocessed features
* `pathological_embedding_512.h5` → 512-d embeddings
* `pathological_preproc_objects.joblib` → Preprocessing models
* `pathological_vae.pt` → VAE weights (if trained)
* `pathological_embedding_ae.pt` → Autoencoder weights (if trained)
* `summary.json` → Metadata summary

---

## Notes

* If PyTorch is unavailable, the pipeline runs in PCA-only mode.
* Multiple imputations ensure robustness against missing data uncertainty.
* Graph smoothing stabilizes patient representations by leveraging neighborhood similarity.
* Embeddings can be used for downstream tasks such as survival prediction, clustering, or risk assessment.

---

## Example Run

```bash
# Run full pipeline with VAE + Autoencoder
python pathological_preproc_novel.py

# Run pipeline with only KNN + PCA
python pathological_preproc_novel.py --no_vae --mode pca
```

---

This preprocessing pipeline provides a structured, probabilistic, and embedding-ready representation of pathological clinical data, suitable for integration into machine learning and deep learning models.
