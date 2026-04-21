# Paraphrase Detection & Semantic Textual Similarity (STS)
This project implemented a high-performance NLP evaluation suite to benchmark state-of-the-art Transformer models for detecting paraphrases and calculating semantic similarity. It evaluates model robustness through granular structural profiling and identifies the core challenges in semantic mapping, such as the "Hard Negative Paradox" and "Cosine Similarity Bias."

🚀 **Key Features**
*   **Multi-Model Architecture**: Features a comparison between three specialized transformer architectures:
    *   **MiniLM (all-MiniLM-L6-v2)**: Optimized for speed and low-latency deployment.
    *   **MPNet (all-mpnet-base-v2)**: High-quality bi-encoder with a balanced accuracy/speed trade-off.
    *   **Elite (DeBERTa-V3 Large)**: A maximum-precision Cross-Encoder for deep semantic reasoning.
*   **Contradiction Handling**: Integrated **NLI Cross-Encoder Refinement** that penalizes similarity scores when logical contradictions are detected.
*   **Tasks**:
    *   **Paraphrase Detection**: Binary classification (Yes/No) using an optimized 0.75 threshold.
    *   **Semantic Textual Similarity (STS)**: Depth evaluation mapping similarity to a 0.0–1.0 scale.
*   **Datasets**: Evaluated on the industry-standard **GLUE Benchmarks** (MRPC and STSb), involving over 12,000 sentence pairs.

📊 **Documentation & Analysis Reports**
The project includes four primary analytical documents located in the `project_analysis/` folder:

1.  **[Model Analysis for 300 test samples.docx](project_analysis/Model_analysis_for_300-test%20samples.docx)**
    A side-by-side comparative table of all three models on a representative subset of 300 cases (150 MRPC / 150 STSb).
    *   **Consolidated Metrics (300 Samples)**:
        *   MiniLM: 78.33% Acc | 0.144 MAE
        *   MPNet: 80.30% Acc | 0.109 MAE
        *   Elite: 82.67% Acc | 0.102 MAE

2.  **[accuracy_analysis.docx](project_analysis/accuracy_analysis.docx)**
    Identifies how accuracy scales with structural mismatch and explains the **Hard Negative Paradox**.
    *   **Insight**: Model accuracy often improves at higher word count mismatches (>12 words) because the structural difference is obvious. Accuracy is lowest at small mismatches (0–4 words) where subtle semantic changes negate meaning (hard negatives). Elite excels here due to full cross-attention.

3.  **[error_distribution_analysis.docx](project_analysis/error_distribution_analysis.docx)**
    Compares error profiles across models to determine score calibration and reliability.
    *   **Insight**: Elite maintains the narrowest error distribution (MAE 0.095) with its peak closest to zero, confirming the most calibrated and reliable similarity scores among the ensemble.

4.  **[sts_correlation_analysis.docx](project_analysis/sts_correlation_analysis.docx)**
    Features scatter plots and Pearson Correlation coefficients to measure rank-order alignment with human judgment.
    *   **Strong Correlations**: Elite (r=0.9123) > MPNet (r=0.8734) > MiniLM (r=0.8421).
    *   **Technical Finding**: All models show an "upward bias" (over-prediction) due to Cosine Similarity Bias and Topic vs. Semantic conflation. This is a structural feature of dense embeddings, which Elite reduces significantly through joint processing.

🛠 **Project Structure**
```text
.
├── main.py                # Main execution script
├── src/                   # Source logic (SBERT, Data Loading, Visualization)
├── project_info.docx      # Core project rationale and architecture documentation
└── project_analysis       # Analytical Visualizations Folder
    ├── Model_analysis_for_300-test samples.docx
    ├── accuracy_analysis.docx
    ├── sts_correlation_analysis.docx
    └── error_distribution_analysis.docx
```

📊 **Final Performance Summary (Full Dataset)**

| Model | Overall Accuracy | F1-Score | Pearson r | MAE |
| :--- | :--- | :--- | :--- | :--- |
| **Elite** | 80.5% | 0.8512 | 0.9123 | 0.095 |
| **MPNet** | 78.1% | 0.8291 | 0.8734 | 0.118 |
| **MiniLM** | 76.7% | 0.8134 | 0.8421 | 0.134 |
