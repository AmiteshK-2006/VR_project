#!/usr/bin/env python3
"""
BLIP-VQA Interactive Web Application
Demonstrates live Visual Question Answering with the fine-tuned LoRA model.
"""

import os
import sys
import torch
from PIL import Image

# Check for required packages
try:
    import gradio as gr
    from transformers import BlipProcessor, BlipForQuestionAnswering
    from peft import PeftModel
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Please ensure your environment has: gradio transformers peft torch pillow")
    sys.exit(1)

# Device selection (CUDA for NVIDIA GPU, CPU for Mac / Windows)
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"🚀 Initializing BLIP-VQA on device: [{device.upper()}]...")

BASE_MODEL_NAME = "Salesforce/blip-vqa-base"
LORA_PATH = os.path.join(os.path.dirname(__file__), "finetuning", "pretrained")

print(f"📦 Loading base processor and model ({BASE_MODEL_NAME})...")
processor = BlipProcessor.from_pretrained(BASE_MODEL_NAME)
base_model = BlipForQuestionAnswering.from_pretrained(BASE_MODEL_NAME).to(device)
base_model.eval()

# Load LoRA adapter if present
lora_model = None
if os.path.exists(LORA_PATH) and os.path.exists(os.path.join(LORA_PATH, "adapter_config.json")):
    print(f"🔧 Attaching LoRA fine-tuned adapter from '{LORA_PATH}'...")
    lora_model = PeftModel.from_pretrained(base_model, LORA_PATH).to(device)
    lora_model.eval()
    print("✅ LoRA adapter attached successfully!")
else:
    print(f"⚠️ Warning: LoRA weights not found at '{LORA_PATH}'. Using base model only.")

def answer_question(image, question, use_lora=True):
    if image is None:
        return "Please upload an image.", ""
    if not question or not question.strip():
        return "Please enter a question.", ""

    # Ensure RGB
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert("RGB")
    else:
        image = image.convert("RGB")

    # Select active model
    active_model = lora_model if (use_lora and lora_model is not None) else base_model

    # Run inference
    inputs = processor(images=image, text=question.strip(), return_tensors="pt").to(device)
    with torch.no_grad():
        out = active_model.generate(**inputs, max_length=16)
    
    predicted_answer = processor.decode(out[0], skip_special_tokens=True).strip()

    # If LoRA is active, optionally also run base model to show comparison
    base_comparison = ""
    if lora_model is not None:
        with torch.no_grad():
            base_out = base_model.generate(**inputs, max_length=16)
        base_ans = processor.decode(base_out[0], skip_special_tokens=True).strip()
        base_comparison = f"Base Model (Zero-Shot): '{base_ans}'\nLoRA Fine-Tuned: '{predicted_answer}'"

    return predicted_answer, base_comparison

# Build Gradio UI
with gr.Blocks(title="BLIP-VQA Live Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔍 BLIP-VQA: Visual Question Answering Live Demonstration
        ### Fine-Tuned with LoRA (Low-Rank Adaptation)
        Upload an image, type a question about its attributes (color, material, shape, texture, features), and get the predicted answer!
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="1. Upload or Select Image")
            question_input = gr.Textbox(
                label="2. Question",
                placeholder="e.g., What color is this? What material is the object? What shape is it?",
                lines=2
            )
            use_lora_checkbox = gr.Checkbox(label="Use Fine-Tuned LoRA Adapter", value=True)
            submit_btn = gr.Button("🔮 Ask Model", variant="primary")
            
        with gr.Column(scale=1):
            answer_output = gr.Textbox(label="Predicted Answer", lines=2)
            comparison_output = gr.Textbox(label="Side-by-Side Comparison (Zero-Shot vs Fine-Tuned)", lines=4)

            gr.Markdown(
                """
                ### 💡 Example Questions to Try:
                - *What is the primary material of the item?*
                - *What color is the main object?*
                - *Describe the shape of the product.*
                - *What is the texture of the surface?*
                """
            )

    submit_btn.click(
        fn=answer_question,
        inputs=[image_input, question_input, use_lora_checkbox],
        outputs=[answer_output, comparison_output]
    )
    question_input.submit(
        fn=answer_question,
        inputs=[image_input, question_input, use_lora_checkbox],
        outputs=[answer_output, comparison_output]
    )

if __name__ == "__main__":
    print("🌐 Launching Gradio web interface at http://127.0.0.1:7860 ...")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
