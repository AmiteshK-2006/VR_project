# BLIP-VQA: Efficient Fine-Tuning with LoRA

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Hugging Face](https://img.shields.io/badge/🤗-Transformers-yellow.svg)

Efficient fine-tuning of BLIP (Bootstrapping Language-Image Pre-training) for Visual Question Answering using Parameter-Efficient techniques on large-scale image-text datasets.

## Overview

This repository contains the code for fine-tuning the BLIP-VQA model on custom datasets using Low-Rank Adaptation (LoRA), enabling efficient training even on consumer hardware. We implement various memory optimization techniques to allow training of this large-scale vision-language model with limited GPU resources.

## Features

- 🚀 Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning
- 💾 8-bit quantization for reduced memory footprint
- ⚡️ KV-Cache optimization for efficient autoregressive decoding
- 💻 Gradient checkpointing for memory-efficient backpropagation
- 📊 Comprehensive logging and checkpointing system
- 🔄 Robust dataset processing with error handling
- 🎛️ Configurable training parameters
- 📱 Multi-GPU support via Accelerate

## Installation

```bash
# Clone repository
git clone git@github.com:Tanish-pat/vr_project.git
cd vr_project/finetuning/

# Set up environment
conda create -n blip-vqa python=3.12
conda activate blip-vqa

# Install dependencies
pip install -r requirements.txt

# Run the Finetuning File
python Updated_Finetuning_with_LORA.py
```


## Requirements

* Python 3.8+
* PyTorch 2.0+
* `transformers`
* `peft`
* `bitsandbytes`
* `accelerate`
* `Pillow`
* `tqdm`
* `numpy`

---

## Usage

### Data Preparation

Prepare your dataset in the following JSON format for each image:

```json
[
  {"path": "relative/path/to/image.jpg"},
  {"questions": [
    {"question": "What color is the sky?", "answer": "Blue"},
    {"question": "How many people are in the image?", "answer": "Two"}
  ]}
]
```
You can add any number of questions in the json file.

### Configuration

Edit the `Config` class in the script to customize training parameters:

```python
class Config:
    # Paths
    TRAIN_JSON_DIR = "/path/to/train/jsons"
    VAL_JSON_DIR = "/path/to/val/jsons"
    IMAGE_PREFIX = "/path/to/images"
    PRETRAINED_DIR = "/path/to/pretrained"
    CHECKPOINT_DIR = "/path/to/checkpoints"

    # Training parameters
    BATCH_SIZE = 8
    NUM_WORKERS = 2
    NUM_EPOCHS = 3
    LEARNING_RATE = 2e-4

    # LoRA parameters
    LORA_R = 4
    LORA_ALPHA = 8
    LORA_DROPOUT = 0.1
    TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]

    # ... more parameters ...
```


### Inference

```python
from inference import VQAPredictor

# Initialize predictor with checkpoint path
predictor = VQAPredictor('checkpoints/best_model')

# Ask questions about an image
answers = predictor.predict_answers('path/to/image.jpg',
                                   ['What is in the image?',
                                    'What color is the car?'])
print(answers)
```

---

## Model Architecture

We fine-tune the `Salesforce/blip-vqa-base` model which consists of:

* Vision Transformer (ViT) for image encoding
* Text encoder based on BERT
* Fusion layers for multimodal reasoning
* Text decoder for answer generation

Our approach modifies only a small subset of parameters using LoRA, drastically reducing training memory requirements while maintaining model quality.

---

## Memory Optimization Techniques

| Technique              | Description                            | Memory Impact                       |
| ---------------------- | -------------------------------------- | ----------------------------------- |
| LoRA                   | Fine-tunes only small adapter matrices | Reduces trainable params to 0.32%   |
| 8-bit Optimizer        | Uses bitsandbytes 8-bit AdamW          | \~60% optimizer state reduction     |
| Gradient Checkpointing | Trades computation for memory          | \~20% activation memory reduction   |
| KV-Cache               | Caches key-value projections           | Optimizes autoregressive decoding   |
| Mixed Precision        | Uses FP16 where appropriate            | Reduces compute memory requirements |

---

## Training Results

Training on a dataset of 1.26M QA pairs from 84,504 unique images showed consistent convergence across all 3 epochs:

| Epoch | Train Loss    | Val Loss     | Learning Rate (start)  |
| ----- | ----------    | --------     | ---------------------  |
| 1     | 8.2099        | 8.1336       | 1.48×10⁻⁴              |
| 2     | 8.1265        | 8.1249       | 7.41×10⁻⁵              |
| 3     | 8.1200        | 8.1214       | 0.1×10⁻⁵               |

The final model achieved strong performance on our validation set with minimal overfitting.

---

## Evaluation

Evaluated on a validation set of 375,000 QA pairs from 25,000 unique images:

* **Final Validation Loss**: 8.1214
* **Accuracy (Exact Match)**: 0.5441
* **BERT F1**: 0.9757
* **Token Level F1**: 0.5441
* **Cosine Similarity**: 0.7535

### Question Types Handled

* Object identification
* Spatial relationships
* Color recognition
* Counting
* Action recognition

---

## Checkpointing Strategy

The implementation uses a dual-trigger checkpoint strategy:

* **Image-based**: Every 2,048 images processed
* **Epoch-based**: At the end of each epoch
* **Best-Model**: When validation loss improves

Each checkpoint contains comprehensive metadata for easy resumption of training.


## License

See the LICENSE file for details.

---
