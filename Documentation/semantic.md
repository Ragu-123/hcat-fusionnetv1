# Text Semantic Preprocessing Pipeline (`text_preproc_to_512_semantic.py`)

This document details the **semantic text preprocessing pipeline** for converting clinical narratives (histories, reports, surgery descriptions) into patient-level **512-dimensional embeddings**. The pipeline supports both **transformer-based embeddings (Bio\_ClinicalBERT)** and lightweight **TF-IDF + TruncatedSVD** representations for environments without GPUs.

---

## **1. Input Data**

* **Source:** Plain-text `.txt` files from three modalities:

  * Histories
  * Reports
  * Surgical descriptions
* **File-to-patient mapping:**

  * Patient IDs inferred from filenames (suffix- or underscore-based numbering).
* **Optional Alignment:**

  * If an HDF5 clinical dataset is provided, script aligns patient IDs across modalities.

---

## **2. Preprocessing Steps**

### **2.1 Text Cleaning**

* Removes HTML/XML-like tags.
* Collapses whitespace and normalizes line breaks.

### **2.2 Sentence Splitting**

* Splits documents into sentences to localize entity context.

### **2.3 Entity Extraction**

* Regex-based identification of clinical terms:

  * Tumor, metastasis, lymph nodes.
  * Resection margins (R0, R1).
  * Tracheostomy, PEG, flap.
  * Complications.

### **2.4 Negation Detection**

* Detects negations within local context:

  * Terms like `no`, `without`, `negative`, `denied`.
* Flags entities as absent if preceded by negation.

---

## **3. Embedding Generation**

### **3.1 Transformer Mode (Default if available)**

* Uses **Bio\_ClinicalBERT** (HuggingFace Transformers).
* Token embeddings aggregated with **mean pooling weighted by attention mask**.
* Produces contextual, domain-specific representations.

### **3.2 TF-IDF + SVD Mode (Fallback)**

* Builds sparse TF-IDF vectors from text corpus.
* Dimensionality reduced to ≤512 with **TruncatedSVD**.
* Outputs normalized embeddings suitable for downstream ML.

---

## **4. Aggregation per Patient**

### **4.1 Document-Level Aggregation**

* Embeddings weighted by:

  * Document length (log-scaled).
  * Entity density (count of detected entities).
  * Negation penalties.

### **4.2 Modality-Level Aggregation**

* Histories, reports, and surgery texts each reduced to a **512-dim modality embedding**.

### **4.3 Cross-Modality Fusion**

* Combines modality embeddings using **soft attention weights** based on modality quality.
* Produces final **patient embedding (512-dim)**.

---

## **5. Outputs**

### **HDF5 (`text_semantic_embeddings_512.h5`)**

* `patient_id`: Patient identifiers.
* `histories_embedding_512`: History-level embedding.
* `reports_embedding_512`: Report-level embedding.
* `surgery_embedding_512`: Surgery-level embedding.
* `text_combined_embedding_512`: Final patient-level embedding.
* `entity_names`: List of entity categories.
* `entity_counts`: Per-patient entity counts.
* `file_counts`: Number of files processed per modality.
* `modality_quality`: Modality confidence scores.
* `metadata`: Notes on preprocessing and embedding mode.

### **Joblib (`text_semantic_preproc_objects.joblib`)**

* Stores fitted TF-IDF/SVD models or transformer configs.

### **JSON (`text_semantic_summary.json`)**

* Human-readable summary with statistics and methodology notes.

---

## **6. Key Notes**

* Supports **multi-modality text fusion**.
* Provides dual modes: **contextual embeddings (BERT)** or **lightweight embeddings (TF-IDF)**.
* Built-in **entity extraction** and **negation handling**.
* Outputs are **plug-and-play** for ML pipelines.
* Modular design ensures flexibility in computational environments.

---

## **7. Example Workflow**

```bash
python text_preproc_to_512_semantic.py \
  --histories ./dataset/TextData/histories_english \
  --reports ./dataset/TextData/reports_english \
  --surgery ./dataset/TextData/surgery_descriptions_english \
  --outdir ./training_dataset/semantic \
  --use_transformer
```

---

## **8. Applications**

* Patient-level text embedding generation.
* Survival analysis and prognosis prediction.
* Clinical decision support via semantic retrieval.
* Multimodal fusion with imaging and genomic embeddings.
* Exploratory analysis of entity distributions across cohorts.
