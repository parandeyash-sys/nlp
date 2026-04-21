# 🧠 Paraphrase Detection & Semantic Textual Similarity (STS)

This project implemented a high-performance NLP evaluation suite to benchmark state-of-the-art Transformer models for detecting paraphrases and calculating semantic similarity. It evaluates model robustness through granular structural profiling and identifies the core challenges in semantic mapping, such as the "Hard Negative Paradox" and "Cosine Similarity Bias."

## 🚀 Key Features

*   **Multi-Model Architecture**: Features a comparison between three specialized transformer architectures:
    *   **MiniLM (all-MiniLM-L6-v2)**: Optimized for speed and low-latency deployment.
    *   **MPNet (all-mpnet-base-v2)**: High-quality bi-encoder with a balanced accuracy/speed trade-off.
    *   **Elite (DeBERTa-V3 Large)**: A maximum-precision Cross-Encoder for deep semantic reasoning.
*   **Contradiction Handling**: Integrated **NLI Cross-Encoder Refinement** that penalizes similarity scores when logical contradictions are detected.
*   **Tasks**:
    *   **Paraphrase Detection**: Binary classification (Yes/No) using an optimized 0.75 threshold.
    *   **Semantic Textual Similarity (STS)**: Depth evaluation mapping similarity to a 0.0–1.0 scale.
*   **Datasets**: Evaluated on the industry-standard **GLUE Benchmarks** (MRPC and STSb), involving over 12,000 sentence pairs.

## 📊 Documentation & Analysis Reports

The project includes four primary analytical documents located in the `project_analysis/` folder:

1.  **[Model Analysis for 300 test samples.docx](project_analysis/Model_analysis_for_300-test%20samples.docx)**: A side-by-side comparative table of all three models on a representative subset of 300 cases (150 MRPC / 150 STSb).
    *   **Consolidated Metrics (300 Samples)**:
        *   MiniLM: 78.33% Acc | 0.144 MAE
        *   MPNet: 80.30% Acc | 0.109 MAE
        *   Elite: 82.67% Acc | 0.102 MAE

2.  **[accuracy_analysis.docx](project_analysis/accuracy_analysis.docx)**: Identifies how accuracy scales with structural mismatch and explains the **Hard Negative Paradox**. Accuracy is lowest at small mismatches where subtle semantic changes negate meaning (hard negatives).

3.  **[error_distribution_analysis.docx](project_analysis/error_distribution_analysis.docx)**: Compares error profiles across models to determine score calibration. Elite maintains the narrowest error distribution, confirming higher calibrated reliability.

4.  **[sts_correlation_analysis.docx](project_analysis/sts_correlation_analysis.docx)**: Features scatter plots and Pearson r alignment (Elite: 0.9123). Documents the **upward bias** (over-prediction) phenomenon in dense embeddings.

## 🛠 Project Structure
```text
.
├── main.py                # Main benchmark execution script
├── app.py                 # Streamlit Interactive UI
├── src/                   # Model and data loading logic
├── project_info.docx      # Core project rationale and architecture documentation
└── project_analysis/      # Analytical Visualizations Folder
```

## 🏆 Project Achievements & Outcomes
Through rigorous structural analysis and full-dataset evaluation (over 12,000 samples), this project has achieved the following:

*   **Verified Robustness**: Demonstrated that the transformer models maintain semantic integrity even when faced with significant word count mismatches.
*   **Identification of the "Hard Negative" Paradox**: Scientifically documented through granular profiling why small structural changes are significantly more challenging for models than large ones.
*   **Industrial Benchmark Alignment**: Successfully mapped standard GLUE evaluation metrics into a professional reporting suite adapted for viva-level scrutiny.

## 🚀 Getting Started

### 1. Installation
Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```

### 2. Run Interactive UI
Check similarity between custom sentence pairs using the Streamlit dashboard:
```bash
streamlit run app.py
```

### 3. Run Benchmark Scripts
Evaluate all models on the full MRPC and STSb datasets:
```bash
python3 main.py
```

## 📊 Final Performance Summary (Full Dataset)

| Model | Overall Accuracy | F1-Score | Pearson r | MAE |
| :--- | :--- | :--- | :--- | :--- |
| **Elite** | **80.5%** | **0.8512** | **0.9123** | **0.095** |
| **MPNet** | 78.1% | 0.8291 | 0.8734 | 0.118 |
| **MiniLM** | 76.7% | 0.8134 | 0.8421 | 0.134 |
