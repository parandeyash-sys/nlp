# 🧠 Multi-Model Paraphrase Detection & Semantic Textual Similarity (STS)

This project implements a state-of-the-art NLP system designed to detect paraphrases and calculate semantic similarity. By leveraging multiple Transformer architectures and advanced linguistic heuristics (NLI, Negation, and Antonyms), it provides a robust framework for understanding textual relationships.

## 🚀 Key Features

*   **Triple Model Support**: Choose between three optimized architectures based on your latency and precision requirements.
*   **Advanced Linguistic Heuristics**:
    *   **NLI Contradiction Detection**: Automatically zeros out similarity scores if a logical contradiction is detected (e.g., "The sun is hot" vs "The sun is cold").
    *   **Negation Handling**: Intelligently identifies mismatched negations to prevent false positives.
    *   **Antonym Sensitivity**: Penalizes similarity when opposite word pairs (e.g., "male" vs "female") are present.
*   **Interactive UI**: A sleek Streamlit-powered dashboard for real-time testing.
*   **Benchmarking Suite**: Built-in support for evaluating models on industry-standard **GLUE** datasets (MRPC and STS).

---

## 🏗 Model Comparison

The system supports three distinct model configurations:

| Model Identity | HuggingFace Architecture | Type | Primary Strength |
| :--- | :--- | :--- | :--- |
| **MiniLM** | `all-MiniLM-L6-v2` | Bi-Encoder | **Speed & Efficiency**: Ideal for real-time applications. |
| **MPNet** | `all-mpnet-base-v2` | Bi-Encoder | **Balanced Performance**: High quality with reasonable latency. |
| **Elite (DeBERTa)** | `stsb-roberta-large` | Cross-Encoder | **Maximum Precision**: Highest accuracy by comparing pairs directly. |

---

## 🔬 Architecture & Logic

Our `SBERTModel` class integrates several layers of validation to ensure semantic integrity:

1.  **Similarity Calculation**:
    *   **Bi-Encoders (MiniLM/MPNet)**: Use Cosine Similarity between dense vector embeddings.
    *   **Cross-Encoders (Elite)**: Process sentence pairs simultaneously for fine-grained attention.
2.  **Linguistic Filters**:
    *   **NLI Check**: Uses `cross-encoder/nli-deberta-v3-base` to ensure that similarity scores respect logical entailment.
    *   **Negation Penalty**: Reduces similarity by 50% if negation presence differs between sentences.
    *   **Opposite Check**: Reduces similarity by 70% if explicit antonyms are detected.

---

## 📊 Documentation & Analysis

The repository includes detailed analytical reports generated through rigorous testing:

*   [project_info.docx](file:///home/yash/nlp/suchir_nlp_3models/project_info.docx): Advanced overview and justification for the **0.75 similarity threshold**.
*   **Project Analysis Folder**:
    *   `accuracy_analysis.docx`: Granular histogram analyzing robustness against word count mismatches.
    *   `sts_correlation_analysis.docx`: Visualization of Pearson correlation between model output and human judgment.
    *   `error_distribution_analysis.docx`: Mapping the absolute error magnitudes across datasets.

---

## 🛠 Installation & Usage

### 1. Requirements
Ensure you have Python 3.8+ installed. Install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Interactive UI
The Streamlit app allows you to select models and test custom sentence pairs.
```bash
streamlit run app.py
```

### 3. Run Benchmark Evaluation
Execute the batch script to see comparative performance on MRPC and STS datasets.
```bash
python3 main.py
```

---

## 🏆 Project Structure

```text
.
├── main.py                # Main benchmark execution script
├── app.py                 # Streamlit web application
├── src/                   # Core implementation
│   ├── sbert_model.py     # Multi-model selection and heuristic logic
│   ├── data_loader.py     # GLUE dataset integration
│   └── evaluation.py      # Metric calculation (Accuracy, F1, Pearsonr)
├── project_info.docx      # Comprehensive documentation
└── project_analysis/      # Generated performance visualizations
```

---

## 🏅 Performance Outcomes

Through extensive evaluation on the **GLUE Benchmark**, this project maintains:
- **High Semantic Integrity**: Handling "Hard Negatives" where structural similarity masks semantic differences.
- **Empirical Validation**: Validated similarity thresholds for real-world paraphrase identification.
- **Multi-Architecture Comparison**: Clear documentation of trade-offs between speed (MiniLM) and depth (Elite).
