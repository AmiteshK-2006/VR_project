# Baseline Evaluation for Zero-Shot VQA

This folder contains the baseline evaluation results for a zero-shot Visual Question Answering (VQA) model.

## Contents

### 1. `vqa_evaluation_results.csv.zip`

This compressed CSV file contains the **per-question evaluation results** for the zero-shot VQA setup.  
Each row corresponds to a question-answer (QA) pair with associated metric outputs (e.g., BERTScore, Exact Match, etc.).

### 2. `Results/`

This folder includes:

- `vqa_metrics.npz`: A compressed NumPy archive containing metric arrays (Exact Match, BERT Score components, Sentence Similarity, etc.) for all QA pairs.
- `main.py` : A Python script that loads the `.npz` file and calculates **average performance metrics** across the entire evaluation set.

# Datasets Information

## 1. Zero-Shot Dataset

**Link:** [Kaggle - Zero-Shot Dataset](https://www.kaggle.com/datasets/transyltoonia/zero-shot-dataset)

**Description:**  
This dataset contains **25,000 JSON files**. Each file corresponds to one image and includes **15 Question-Answer (QA) pairs**.  
The dataset is designed to evaluate the **zero-shot performance** of VQA models—models that haven't seen this dataset during training.

**Use Case:**  
Ideal for baseline benchmarking of zero-shot VQA models such as BLIP.  
It contains diverse QA pairs across various question types, making it valuable for in-depth evaluation.

---

## 2. Berkley Dataset

**Link:** [Kaggle - Berkley Dataset](https://www.kaggle.com/datasets/transyltoonia/berkley-dataset)

**Description:**  
An adapted version of the Berkley VQA dataset. It features structured visual scenes and paired questions that are suitable for robust model evaluation.

**Use Case:**  
Useful for both zero-shot and fine-tuned settings. Provides a consistent ground for evaluating generalization and model robustness across tasks.


