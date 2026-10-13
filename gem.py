# FULL WORKING WITH DYNAMIC MAIN_ID AND OTHER_ID AND SAVING WITH IMAGE PATH AND 15 QUESTIONS
import csv
import os
import random
import json
import google.generativeai as genai
from PIL import Image
import time
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# ========== CONFIGURATION ==========
API_FILE = "API.json"
csv_file_path = 'images/metadata/images.csv'
json_directory = 'listings/metadata/'
images_folder = 'images/small/'
output_folder = 'vqa_dataset/'

# ========== API KEY MANAGEMENT ==========
def load_api_keys():
    with open(API_FILE, "r") as f:
        return json.load(f)

def save_api_keys(api_keys):
    with open(API_FILE, "w") as f:
        json.dump(api_keys, f, indent=2)

def rotate_api_key(api_keys):
    failed_key = api_keys.pop(0)
    api_keys.append(failed_key)
    save_api_keys(api_keys)
    print(f"🔁 Rotated API key. Using new key: {api_keys[0]}")
    return api_keys

api_keys = load_api_keys()
if not api_keys:
    print("❌ No API keys found in API.json.")
    exit(1)

genai.configure(api_key=api_keys[0])
safety_settings = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
}

model = genai.GenerativeModel(
    model_name="models/gemini-1.5-flash",
    safety_settings=safety_settings,
)

# ========== VQA GENERATION ==========
def generate_vqa(image_path, metadata, output_json_path):
    global api_keys, model
    with Image.open(image_path) as img:
        
        prompt = (
            "You are provided with an image of a product and its associated metadata from an e-commerce listing.\n\n"
            "Follow these instructions precisely:\n\n"
            "1. **First**, perform a detailed visual analysis of the image. Extract observable product attributes such as color, shape, material, design elements, and functional features.\n"
            "2. **Then**, refer to the metadata only as a secondary source to confirm or supplement visual evidence. Do not rely solely on metadata.\n"
            "3. Based solely on what can be visually confirmed (and optionally supported by metadata), generate **15 one-word Visual Question Answer (VQA) pairs**.\n"
            "4. Only include **relevant, non-trivial** questions. If the product is simple or minimal, produce fewer but high-quality pairs.\n"
            "5. List the questions in **decreasing order of importance**, prioritizing the most distinctive or defining characteristics.\n"
            "6. Each question must be detailed (approx. 15 words) and clearly refer to something visually evident in the product.\n"
            "7. Each answer must be **a single word**, strictly grounded in the image or metadata.\n"
            "8. Avoid any hallucinations, guesses, generic filler, or duplicate questions.\n"
            "9. Cover diverse aspects such as color, material, texture, design, components, shape, or markings.\n"
            "10. Format your response as a JSON array:\n"
            "    [\n"
            "      {\"question\": \"...\", \"answer\": \"...\"},\n"
            "      ...\n"
            "    ]\n"
            "11. **Return only the JSON array above — no prose, no comments, no formatting tags.**\n"
        )

        while True:
            try:
                response = model.generate_content([prompt, img, json.dumps(metadata or {})], stream=False)
                result = response.text
                break
            except Exception as e:
                if "429" in str(e):
                    print("⚠️ Rate limit hit. Rotating API key...")
                    api_keys = rotate_api_key(api_keys)
                    genai.configure(api_key=api_keys[0])
                    model = genai.GenerativeModel(
                        model_name="models/gemini-1.5-flash",
                        safety_settings=safety_settings,
                    )
                elif "400" in str(e):
                    print("⚠️ API key Expired. Rotating API key...")
                    api_keys = rotate_api_key(api_keys)
                    genai.configure(api_key=api_keys[0])
                    model = genai.GenerativeModel(
                        model_name="models/gemini-1.5-flash",
                        safety_settings=safety_settings,
                    )
                else:
                    print("❌ Error during generation:", str(e))
                    return

        try:
            cleaned_result = result.strip().lstrip('```json').rstrip('```').strip()
            qa_pairs = json.loads(cleaned_result)

            # Create the final structure as a list with two elements
            final_output = [
                {"path": image_path},  # 0th element contains the path
                {"questions": qa_pairs}  # 1st element contains the questions
            ]

            with open(output_json_path, "w", encoding="utf-8") as out_f:
                json.dump(final_output, out_f, indent=2)
            print(f"✅ Image path is {image_path}")
            print(f"✅ Saved VQA to {output_json_path}")
        except json.JSONDecodeError:
            print("❌ GPT output was not valid JSON.")
            print(result)

# ========== JSON LOOKUP ==========
def search_json_entry(directory, image_id, check_main_only):
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    for line_num, line in enumerate(lines, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            if not isinstance(entry, dict):
                                continue

                            main_match = entry.get("main_image_id") == image_id
                            other_match = not check_main_only and image_id in entry.get("other_image_ids", [])

                            if main_match or other_match:
                                print(f"✅ Found {image_id} in {filename} (line {line_num})")
                                return entry
                        except json.JSONDecodeError:
                            print(f"⚠️ Skipping malformed JSON at {filename} line {line_num}")
            except Exception as e:
                print(f"❌ Error reading {filename}: {e}")
    return None

# ========== MAIN LOOP ==========
def main(check_main_image_id=0):
    os.makedirs(output_folder, exist_ok=True)

    loop = 1000
    while loop > 0:
        with open(csv_file_path, mode='r') as file:
            rows = list(csv.DictReader(file))
            image_metadata = random.choice(rows)

        image_id = image_metadata['image_id']
        image_path = os.path.join(images_folder, image_metadata['path'])
        print(f"\n🔍 Checking image_id: {image_id}")

        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            continue

        metadata = None
        if check_main_image_id == 1:
            metadata = search_json_entry(json_directory, image_id, check_main_only=True)
            if not metadata:
                print(f"❌ image_id {image_id} not found as main_image_id.")
                continue
        elif check_main_image_id == 0:
            metadata = None  # skip lookup entirely

        print(f"✅ current api key is {api_keys[0]}")
        print(f"✅ Image {'and metadata ' if metadata else ''}found. Proceeding to VQA generation.")
        output_json_path = os.path.join(output_folder, f"{image_id}.json")
        generate_vqa(image_path, metadata, output_json_path)

        time.sleep(1)
        loop -= 1

# ========== ENTRY POINT ==========
if __name__ == "__main__":
    main(check_main_image_id=0)  # 0: no check, 1: main_image_id only