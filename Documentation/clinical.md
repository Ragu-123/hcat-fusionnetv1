# Documentation for Clinical Data Preprocessing and Embedding Pipelines

This document explains the functionality of two Python scripts:

1. **`clinical_pre.py`** – Performs advanced preprocessing and imputation on clinical JSON data, then saves processed features and metadata to HDF5.
2. **`clinical_make_embedding_512.py`** – Generates 512-dimensional embeddings from preprocessed features using either a denoising autoencoder (AE, if PyTorch is available) or PCA.

Both scripts are designed as part of a pipeline for **HANCOCK clinical data preprocessing and representation learning**.

---

## 1. clinical\_preproc.py

### Purpose

This script preprocesses a structured clinical JSON dataset by:

* Deriving clinical labels (5-year survival, 2-year recurrence).
* Performing advanced imputations (KNN, VAE, ensemble methods).
* Refining imputations with **graph smoothing**.
* Producing multiple imputed datasets, aggregating mean and variance.
* Saving results to an **HDF5 file** and **joblib preprocessing objects**.

### Input

* `clinical_data.json`: Raw structured clinical data.

### Outputs

* `clinical_preprocessed_advanced.h5`: Main processed feature dataset.
* `clinical_preproc_objects.joblib`: Preprocessing pipeline objects.
* `clinical_vae.pt`: (Optional) Trained VAE weights.

### Workflow

1. **Load JSON** → Read the structured clinical data.

2. **Derive labels** →

   * Survival label (`surv_5yr_label`): based on tumor-related mortality within or after 5 years.
   * Recurrence label (`rec_2yr_label`): based on recurrence events within 2 years.

3. **Feature selection** → Automatically classify features as numeric or categorical.

4. **Feature encoding** →

   * Numeric values kept as is.
   * Categorical values converted via **frequency encoding**.

5. **Missing value handling** →

   * Simple median imputation.
   * KNN imputation.
   * Variational Autoencoder (VAE)-based imputation (probabilistic, multiple draws).

6. **Graph smoothing** → Refines imputations using patient similarity graph.

7. **Ensemble aggregation** → Combine multiple imputations → compute mean & variance.

8. **Save outputs** →

   * Features (mean, variance, mask).
   * Labels (survival, recurrence).
   * Preprocessing objects (scaler, imputers, configs).

### Key Algorithms

* **KNNImputer** (from scikit-learn): fills missing values using nearest neighbors.
* **VAEImputer** (PyTorch): trains a generative model to probabilistically impute missing data.
* **Graph smoothing**: uses RBF kernel similarity between patients to refine missing values by averaging across similar patients.

### Command-Line Options

```
--epochs   (default: 60)   Number of training epochs for VAE.
--bs       (default: 64)   Batch size for VAE training.
--lr       (default: 1e-3) Learning rate for VAE.
--no_vae                Skip VAE step and only use KNN imputation.
```

---

## 2. clinical\_make\_embedding\_512.py

### Purpose

This script generates **512-dimensional embeddings** from the preprocessed HDF5 file. Embeddings summarize patient data into compact vectors useful for machine learning models.

### Input

* `clinical_preprocessed_advanced.h5`: Output from preprocessing.

### Outputs

* `clinical_embedding_512.h5`: Final patient embeddings.
* `clinical_embedding_preproc.joblib`: Preprocessing schema.
* `clinical_embedding_ae.pt`: (Optional) Trained autoencoder weights.

### Workflow

1. **Load preprocessed data** → Patient IDs, features (mean, var, mask), labels.

2. **Build input matrix** → Concatenate:

   * Feature means.
   * Feature variances.
   * Missingness mask.

3. **Embedding creation**:

   * **Autoencoder (preferred, if PyTorch available)**:

     * Denoising AE trained to reconstruct inputs with dropout noise.
     * Bottleneck layer outputs 512-d embedding.
   * **PCA fallback**:

     * Reduces input dimensionality to 512.
     * Pads if fewer than 512 dimensions are available.

4. **Evaluation** → Logistic Regression classifiers are trained on embeddings to predict:

   * Survival (5-year outcome).
   * Recurrence (2-year outcome).
     Performance is measured via **Accuracy** and **F1 score**.

### Key Algorithms

* **Denoising Autoencoder (AE)**: Learns embeddings by reconstructing corrupted inputs.
* **Principal Component Analysis (PCA)**: Linear dimensionality reduction.
* **Logistic Regression**: Simple classifier used to evaluate embedding quality.

---

## Technical Notes

* **Random Seed**: Both scripts fix random seeds (`SEED=42`) for reproducibility.
* **Torch optionality**: If PyTorch is not available, both VAE and AE steps are skipped, falling back to simpler models.
* **File format**: HDF5 stores numeric arrays efficiently; joblib stores Python objects (scalers, imputers).

---

## Example Usage

### Run preprocessing:

```bash
python clinical_preproc_novel.py --epochs 50 --bs 128
```

### Run embedding creation:

```bash
python clinical_make_embedding_512.py
```

---

## Summary

Together, these scripts form a robust preprocessing and representation pipeline for structured clinical data:

* Missing data is carefully imputed using **ensemble + VAE + graph refinement**.
* Patient data is represented as **512-dimensional embeddings**, either with AE or PCA.
* Outputs are ready for downstream tasks like survival prediction, recurrence risk assessment, or integration into multi-modal pipelines.
