# Temporal Preprocessing Steps for Clinical Blood Data

This document explains the **detailed preprocessing workflow** implemented in `temporal_preproc_advanced.py`. The goal is to convert irregular, sparse, and noisy clinical blood test measurements into **patient-level 512-dimensional embeddings**.

---

## 1. Dataframe Construction

* Raw records are parsed into a **structured DataFrame**.
* Standardizations applied:

  * `patient_id` → normalized as zero-padded string identifiers.
  * `days_before_first_treatment` → converted to integers.
  * `analyte_name` → normalized (whitespace stripped, consistent casing).
  * `value` → converted to numeric, invalid entries set as NaN.

This ensures consistency in identifiers and analyte values across the dataset.

---

## 2. Reference Range Integration

* Reference ranges (`male_min`, `male_max`, `female_min`, `female_max`) are loaded.
* Each analyte is mapped to its **physiological normal ranges**.
* These ranges act as a clinical baseline for **physiology-aware imputation**.

---

## 3. Analyte Selection

* Counts how many patients have each analyte recorded.
* Selects:

  * Analytes present in at least 3 patients.
  * Additional analytes listed in reference ranges.
* Final analyte set balances **statistical robustness** and **clinical relevance**.

---

## 4. Temporal Binning

* Defines a **time window of 30 days** before first treatment.
* Splits this window into **16 equal bins**.
* For each analyte:

  * Measurements are assigned to bins using their day offset.
  * If multiple values fall in the same bin → averaged.

This produces a **fixed-length temporal sequence per patient**.

---

## 5. Per-Patient Matrices

* Each patient is represented as a **matrix of shape (n\_analytes × seq\_len)**.
* Missing entries remain as **NaN** for later imputation.
* Observation counts are tracked for quality assessment.

---

## 6. Quality Scoring

* For each patient:

  * Compute fraction of observed entries vs. total possible entries.
* Produces a **quality score (0–1)**, later stored in metadata.

---

## 7. Physiology-Aware Filling

* For analytes with **no data**:

  * Fill with the **midpoint of clinical reference ranges** (if available).
* For analytes with **partial missing data**:

  * Apply **1D interpolation across the time axis**.

This step grounds imputations in **clinical plausibility** instead of purely statistical methods.

---

## 8. Cohort-Level KNN Imputation

* After physiology fill, a **second imputation layer** is applied:

  * Flatten per-patient matrices.
  * Use **KNNImputer (k=8)** across patients to borrow patterns from similar cohorts.
* If NaNs remain, fallback to **column medians**.

This leverages **population-level patterns** to refine patient-specific imputations.

---

## 9. Standardization

* Flatten patient matrices into feature vectors.
* Apply **StandardScaler**:

  * Zero-mean, unit-variance scaling.
  * Ensures analytes are comparable across patients.

---

## 10. Embedding Creation

Two alternative embedding strategies:

### a) **Denoising LSTM Encoder (default if PyTorch available)**

* Input: patient temporal matrix → reshaped as (seq\_len × n\_analytes).
* LSTM encoder produces a **512-d embedding**.
* Decoder reconstructs the original sequence.
* Training objective: **denoising autoencoding**

  * Randomly mask 10% of inputs.
  * Reconstruct the complete sequence.
* Produces robust embeddings that encode both **temporal dynamics** and **analyte interactions**.

### b) **PCA Fallback**

* If PyTorch unavailable or `--mode=pca` selected:

  * Flatten + scale features.
  * Apply **PCA → 512 components**.
  * If fewer than 512 dimensions are available, pad with zeros.

---

## 11. Saving Outputs

* Results saved in HDF5 + JSON + Joblib formats:

  * **Embeddings (512-d per patient)**
  * **Analyte list** and **time bin centers**
  * **Quality scores**
  * **Missingness statistics**
  * **Reference to preprocessing objects** (scaler, KNN imputer)

---

## 12. Summary Generation

* Produces a JSON summary with:

  * Number of patients and analytes
  * Embedding method used (LSTM or PCA)
  * Distribution of quality scores
  * Missingness fraction per analyte
  * Notes on preprocessing pipeline

This ensures **reproducibility** and **interpretability** for downstream analysis.

---

#  Key Innovations

1. **Physiology-aware imputation** grounded in medical knowledge.
2. **Two-stage imputation** (reference-fill → cohort KNN).
3. **Temporal alignment** across patients via binning.
4. **Embedding learning** with a denoising LSTM encoder.

---

# Final Output

Each patient is represented by:

* A **512-dimensional vector** embedding.
* Metadata: quality score, missingness stats, analyte/time structure.

This transforms heterogeneous, irregular clinical data into a **uniform, information-rich representation** for integration into multimodal models (HCAT).
