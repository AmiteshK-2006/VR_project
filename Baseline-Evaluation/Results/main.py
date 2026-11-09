import numpy as np

# Load the file
data = np.load('vqa_metrics.npz')

# Summary statistics
print("Evaluation Summary (Means over all samples):\n")

print(f"Exact Match: {np.mean(data['exact_match']):.4f}")
print(f"Token-level F1: {np.mean(data['f1_token']):.4f}")
print(f"BERT Precision: {np.mean(data['bert_p']):.4f}")
print(f"BERT Recall: {np.mean(data['bert_r']):.4f}")
print(f"BERT F1: {np.mean(data['bert_f1']):.4f}")
print(f"Sentence Similarity: {np.mean(data['sent_sim']):.4f}")
print(f"Cosine Similarity: {np.mean(data['cos_sim']):.4f}")
