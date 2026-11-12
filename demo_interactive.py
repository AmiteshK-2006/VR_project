#!/usr/bin/env python3
"""
BLIP-VQA Interactive CLI Demo
Demonstrates Visual Question Answering directly in the terminal.
"""

import os
import sys

def main():
    print("=" * 70)
    print("   BLIP-VQA: LIVE TERMINAL DEMONSTRATION")
    print("   Fine-Tuned with LoRA (Low-Rank Adaptation)")
    print("=" * 70)

    try:
        from inference import VQAPredictor
    except ImportError as e:
        print(f"❌ Error importing inference module: {e}")
        sys.exit(1)

    checkpoint_dir = os.path.join(os.path.dirname(__file__), "finetuning", "pretrained")
    predictor = VQAPredictor(checkpoint_path=checkpoint_dir)

    print("\n💡 Tip: You can drag and drop any image file from your Mac into this terminal!")
    print("Type 'exit' or 'q' at any prompt to quit.\n")

    while True:
        try:
            image_path = input("\n📷 Enter path to image (or 'q' to quit): ").strip().strip("'\"")
            if image_path.lower() in ('q', 'quit', 'exit'):
                print("Exiting demo. Good luck with your presentation!")
                break

            if not os.path.exists(image_path):
                print(f"❌ File not found at: '{image_path}'. Please try again.")
                continue

            print(f"✅ Loaded image: {os.path.basename(image_path)}")

            while True:
                question = input("\n❓ Ask a question about this image (or 'new' for another image): ").strip()
                if question.lower() in ('new', 'n'):
                    break
                if question.lower() in ('q', 'quit', 'exit'):
                    print("Exiting demo. Good luck with your presentation!")
                    return
                if not question:
                    continue

                answer = predictor.predict_answers(image_path, question)
                print(f"👉 Predicted Answer: \033[1;32m{answer}\033[0m")

        except (KeyboardInterrupt, EOFError):
            print("\nDemo interrupted. Bye!")
            break

if __name__ == "__main__":
    main()
