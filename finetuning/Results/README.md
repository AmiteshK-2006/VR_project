# Evaluation Of Model after Finetuning

This script computes and prints summary statistics of the model after finetuning using pre-computed metrics stored in a `.npz` file.

## Contents

- `vqa_metrics.npz`: A NumPy archive containing evaluation metrics for each VQA prediction.
- `evaluate_metrics.py`: Python script to load and summarize the evaluation metrics.


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
| Exact Match          | 0.5441  | 
| Token-level F1       | 0.5441  |
| BERT Precision       | 0.9784  |
| BERT Recall          | 0.9740  |
| BERT F1              | 0.9758  |
| Sentence Similarity  | 0.7535  |
| Cosine Similarity    | 0.7535  |

