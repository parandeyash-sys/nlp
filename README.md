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

## 🏆 Project Achievements & Outcomes

Through rigorous structural analysis and full-dataset evaluation (3,225 samples), this project has achieved the following:

*   **Verified Robustness**: Demonstrated that the SBERT model maintains semantic integrity even when faced with significant word count mismatches.
*   **Identification of the "Hard Negative" Paradox**: Scientifically documented through granular histograms why small structural changes are more challenging than large ones.
*   **Industrial Benchmark Alignment**: Successfully mapped standard GLUE evaluation metrics into a professional reporting suite adapted for viva-level scrutiny.

## 📊 Performance Summary (Final Evaluation)

*   **Global Accuracy (Paraphrase Detection)**: **83.67%**
    *   This score reflects the model's peak reliability in detecting semantic equivalence across the validated dataset.
*   **Mean Absolute Error (STS)**: **0.1148**
    *   This demonstrates a high degree of precision, showing that on average, our model's similarity score is within 0.11 of human-annotated values.
*   **Threshold Efficiency**: The **0.75 threshold** was empirically validated as the optimal balance for achieving this high-precision paraphrase detection.
