#!/usr/bin/env python3
"""
BLIP-VQA Inference Engine with LoRA
Loads the Salesforce/blip-vqa-base model and attaches trained LoRA adapters.
Works across CUDA (NVIDIA), MPS (Apple Silicon Mac), and CPU.
"""

import os
import sys
import argparse

def get_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # Note: PyTorch BLIP text generation currently has a known MPSEvent issue on macOS Metal,
        # so CPU is the most stable and fast device for BLIP inference on Mac.
        return "cpu"
    except ImportError:
        return "cpu"

class VQAPredictor:
    def __init__(self, checkpoint_path=None, base_model_name="Salesforce/blip-vqa-base"):
        try:
            import torch
            from PIL import Image
            from transformers import BlipProcessor, BlipForQuestionAnswering
            from peft import PeftModel
        except ImportError as e:
            print(f"❌ Missing required ML library: {e}")
            print("👉 Run: pip install torch torchvision transformers peft pillow")
            sys.exit(1)

        self.device = get_device()
        print(f"⚙️ Initializing VQAPredictor on device: [{self.device.upper()}]...")

        # Default checkpoint path
        if checkpoint_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoint_path = os.path.join(script_dir, "finetuning", "pretrained")

        print(f"📦 Loading base model '{base_model_name}'...")
        self.processor = BlipProcessor.from_pretrained(base_model_name)
        base_model = BlipForQuestionAnswering.from_pretrained(base_model_name)

        if os.path.exists(checkpoint_path) and os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
            print(f"🔧 Attaching fine-tuned LoRA adapter from '{checkpoint_path}'...")
            self.model = PeftModel.from_pretrained(base_model, checkpoint_path)
        else:
            print(f"⚠️ Warning: Checkpoint path '{checkpoint_path}' not found. Using base model.")
            self.model = base_model

        self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded and ready for inference!")

    def predict_answers(self, image_path, questions):
        """
        Takes an image path and a question (str) or list of questions (list[str]).
        Returns answers as a list of strings (or single string if single question passed).
        """
        import torch
        from PIL import Image

        if isinstance(questions, str):
            single_mode = True
            question_list = [questions]
        else:
            single_mode = False
            question_list = questions

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")

        image = Image.open(image_path).convert("RGB")
        answers = []

        for q in question_list:
            inputs = self.processor(images=image, text=q, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=16)
            answer = self.processor.decode(out[0], skip_special_tokens=True).strip()
            answers.append(answer)

        return answers[0] if single_mode else answers

def main():
    parser = argparse.ArgumentParser(description="BLIP-VQA LoRA Inference CLI")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--question", type=str, help="Question to ask about the image")
    parser.add_argument("--checkpoint", type=str, default="finetuning/pretrained", help="Path to LoRA checkpoint")
    args = parser.parse_args()

    if not args.image or not args.question:
        print("Usage: python inference.py --image <path/to/image.jpg> --question \"What color is this?\"")
        print("\nTesting dependency check:")
        predictor = VQAPredictor(checkpoint_path=args.checkpoint)
        print("Model initialized successfully!")
        return

    predictor = VQAPredictor(checkpoint_path=args.checkpoint)
    answer = predictor.predict_answers(args.image, args.question)
    print(f"\nImage:    {args.image}")
    print(f"Question: {args.question}")
    print(f"Answer:   {answer}")

if __name__ == "__main__":
    main()
