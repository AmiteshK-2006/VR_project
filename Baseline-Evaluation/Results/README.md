# VQA Baseline Evaluation Summary

This script computes and prints summary statistics for a zero-shot Visual Question Answering (VQA) baseline evaluation using pre-computed metrics stored in a `.npz` file.

## Contents

- `vqa_metrics.npz`: A NumPy archive containing evaluation metrics for each VQA prediction.
- `evaluate_metrics.py`: Python script to load and summarize the evaluation metrics.

## Baseline Model

These results correspond to the performance of the **BLIP (Salesforce/blip-vqa-base)** model in a **zero-shot** setting, without any fine-tuning on the task-specific dataset. The evaluation covers standard token-level and semantic similarity metrics.

## Metrics Included

- **Exact Match**: Proportion of predictions that exactly match the ground truth.
- **Token-level F1**: Overlap of tokens between prediction and ground truth.
- **BERT Precision / Recall / F1**: Semantic similarity measured using BERT embeddings.
- **Sentence Similarity**: Sentence-level similarity score.
- **Cosine Similarity**: Embedding-based similarity between prediction and ground truth.

## How to Run

1. Ensure you have Python and NumPy installed:

```bash
pip install numpy
```

2. Place the vqa_metrics.npz file in the same directory as the script.


Run the script:
```bash
python evaluate_metrics.py
```

## Evaluation Summary (Means over all samples)

| Metric               | Score   |
|----------------------|---------|
| Exact Match          | 0.2713  |
| Token-level F1       | 0.9352  |
| BERT Precision       | 0.9476  |
| BERT Recall          | 0.9252  |
| BERT F1              | 0.9352  |
| Sentence Similarity  | 0.5873  |
| Cosine Similarity    | 0.5873  |

