# HCAT-FusionNet:  Multimodal Preprocessing and Fusion for Survival and Recurrence Prediction,using variational autoencoders and cross-modal attention for holistic healthcare outcome prediction.

This repository contains preprocessing pipelines and training framework for the **HANCOCK (Head and Neck Cancer Cohort)** dataset used in the **Hancothon25 Challenge** at MICCAI 2025. The challenge focuses on predicting **5-year survival** and **2-year recurrence** using multimodal patient data (clinical, pathological, semantic text, spatial histopathology, temporal blood tests).

Our solution introduces **novel preprocessing, imputation, and fusion strategies** to extract robust 512-dimensional embeddings from heterogeneous modalities, followed by **advanced multi-modal training**.

## Preprocessed_h5_files and Model Weights
* **Huggingface**: [H5](https://huggingface.co/ragunath-ravi/hcat-fusionnet/tree/main/preprocessed_h5_files)
* **HuggingFace**: [Model Weights](https://huggingface.co/ragunath-ravi/hcat-fusionnet/tree/main/model/hcat_checkpoints_v_improved)

---

---

## Features

* **Clinical Data Preprocessing**: Advanced imputation ensemble + VAE-based handling of missing data.
* **Pathological Data Preprocessing**: Probabilistic imputation with graph smoothing and 512-d embeddings.
* **Semantic Text Processing**: ClinicalBERT / TF-IDF + SVD pipelines for histories, reports, and surgery descriptions.
* **Spatial Histopathology Aggregation**: Transformer-based aggregation of patch-level features with spatial awareness.
* **Temporal Blood Data**: Physiology-aware normalization, KNN refinement, and LSTM encoder for sequential signals.
* **Fusion Training**: Multi-modal VAE with attention-based cross-modal imputation, joint latent space learning, and uncertainty quantification.
* **Evaluation**: Binary classification of survival and recurrence, reporting accuracy and F1-score.

---



## Documentation

Detailed explanations of each pipeline are available here:

* [Clinical Preprocessing](https://github.com/Ragu-123/hcat-fusionnet/blob/main/Documentation/clinical.md)
* [Pathological Preprocessing](https://github.com/Ragu-123/hcat-fusionnet/blob/main/Documentation/pathological.md)
* [Semantic Text Processing](https://github.com/Ragu-123/hcat-fusionnet/blob/main/Documentation/semantic.md)
* [Spatial Histopathology](https://github.com/Ragu-123/hcat-fusionnet/blob/main/Documentation/spatial.md)
* [Temporal Blood Data](https://github.com/Ragu-123/hcat-fusionnet/blob/main/Documentation/temporal.md)
* [Training Pipeline](https://github.com/Ragu-123/hcat-fusionnet/blob/main/Documentation/train.md)

---

## Results

From the enhanced training pipeline (`train2.py`), the system achieved:

* **5-year Survival F1-score**: **0.80**
* **2-year Recurrence F1-score**: **0.95**
* **Average F1-score**: **0.875**

*(See [`enhanced_hcat_training_summary.json`](enhanced_hcat_training_summary.json) for full training logs and config.)*

---

## Methods Summary

* **Imputation**: Multi-modal VAE, KNN, PCA, and graph smoothing.
* **Embeddings**: Standardized 512-d representations across modalities.
* **Fusion**: Attention-based cross-modal integration with uncertainty weighting.
* **Classification**: Binary prediction of survival and recurrence with robust evaluation.

---

## Challenge Context

This work addresses the **HANCOCK multimodal dataset** provided for **Hancothon25 (MICCAI 2025)**. The dataset includes **763 patients** with modalities:

* Clinical structured data
* Pathology structured data
* Histopathology WSIs & TMAs
* Tabular blood test data
* Free-text clinical/surgery reports

Our framework is designed for **precision oncology**, enabling predictive modeling for treatment planning and follow-up.

---

## Performance & Insights

* The system demonstrates strong **generalization across modalities**.
* Temporal and pathological modalities improved recurrence prediction.
* Clinical and semantic features boosted survival classification.
* Fusion strategies with uncertainty modeling ensured robustness under missing modalities.



---

## Contact

For questions or collaborations:

* **Author**: [Harish G](https://github.com/Harish2404lll), [Ragunath R](https://github.com/Ragu-123), [Sanjay S](https://github.com/22002102)


