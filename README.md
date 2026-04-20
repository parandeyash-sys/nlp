# Paraphrase Detection & Semantic Textual Similarity (STS)

This project implements a high-performance NLP system to detect paraphrases and calculate semantic similarity using state-of-the-art Transformer models. It focuses on evaluating model robustness across various structural complexities.

## 🚀 Key Features

*   **Models**: Powered by **Sentence-BERT (SBERT)** using the `all-MiniLM-L6-v2` architecture and enhanced with **NLI Cross-Encoders** for contradiction handling.
*   **Tasks**:
    *   **Paraphrase Detection**: Binary classification (Yes/No) using an optimized similarity threshold of **0.75**.
    *   **Semantic Textual Similarity (STS)**: Regression task mapping similarity to a 0.0–1.0 scale (normalized from human 0–5 scores).
*   **Datasets**: Evaluated on the industry-standard **GLUE Benchmark** datasets:
    *   **MRPC** (Microsoft Research Paraphrase Corpus): ~5,400 pairs.
    *   **STSb** (Semantic Textual Similarity Benchmark): ~8,600 pairs.

## 📊 Documentation & Analysis Reports

This repository contains a comprehensive suite of professional documentation designed for a viva presentation, located in the root and the `project_analysis/` folder:

### 📄 Core Project Info
*   **`project_info.docx`**: Advanced overview of the project architecture, justification for the 0.75 similarity threshold, and the scientific rationale behind dataset selection and standard GLUE splits.

### 📈 Structural & Statistical Reports
*   **`project_analysis/project_analysis_2.docx`**: High-resolution performance table grouping sentence pairs by their structural profiles (sentence/word count mismatch). It provides mean results for accurate statistical representation.
*   **`project_analysis/accuracy_analysis.docx`**: A granular histogram based on the **full test dataset (3,225 pairs)**. It identifies how accuracy scales with word count mismatch and explains the **"Hard vs. Easy Negatives" paradox**.
*   **`project_analysis/sts_correlation_analysis.docx`**: Scatter plot visualizing the correlation between human judgment and model predictions.
*   **`project_analysis/error_distribution_analysis.docx`**: Histogram showing the distribution of absolute error magnitudes (MAE).

## 🛠 Project Structure

```text
.
├── main.py                # Main execution script
├── src/                   # Model and data loading logic
│   ├── sbert_model.py     # SBERT and Cross-Encoder implementation
│   └── data_loader.py     # GLUE dataset integration
├── project_info.docx      # Comprehensive Project Documentation
└── project_analysis       # Analytical Visualizations Folder
    ├── project_analysis_2.docx
    ├── accuracy_analysis.docx
    ├── sts_correlation_analysis.docx
    └── error_distribution_analysis.docx
```

## 🏆 Performance Summary
*   **Global Accuracy (Paraphrase)**: High stability across varying sentence lengths.
*   **Mean Absolute Error (STS)**: Demonstrates close alignment with human annotated "gold standard" similarity scores.
