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
        # # prompt1
        # prompt = (
        #     "You are provided with an image of a product along with its metadata from an e-commerce listing. Please adhere to the following instructions:\n\n"
        #     "1. Analyze the image and metadata thoroughly. The image is a representation of the product, and the metadata provides descriptive attributes.\n\n"
        #     "2. Based ONLY on the information visible in the image and described in the metadata, generate up to **15 simple, one-word Visual Question Answer (VQA) pairs**, but only include **relevant** and **non-trivial** questions..\n\n"
        #     "3. Do not hallucinate or generate answers that are not supported by the image or metadata. All answers should be directly inferable from the provided content.\n\n"
        #     "4. If the product is minimal, generate only a few **important** questions.\n"
        #     "5. Questions must be in **decreasing order of importance**, i.e., the most significant characteristics should come first.\n"
        #     "6. Each answer must be one word, directly inferred from the image or metadata. No guesses.\n"
        #     "7. The output must strictly adhere to the following JSON format:\n"
        #     "   [{'question': '...', 'answer': '...'}, ...]\n\n"
        #     "8. Ensure the questions cover diverse aspects of the product, such as its color, material, features, and other relevant attributes.\n\n"
        #     "9. Avoid duplicates, overly generic, or filler questions.\n"
        #     "10. Make sure questions are detailed, about 15 words, and provide enough context.\n"
        #     "11. AND ONLY RETURN OUTPUT IN THE JSON FORMAT AS TOLD\n"
        # )

        # prompt2
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







# # FULL WORKING
# import csv
# import os
# import random
# import json
# import google.generativeai as genai
# from PIL import Image
# import time
# from google.generativeai.types import HarmCategory, HarmBlockThreshold

# # ========== CONFIGURATION ==========
# API_FILE = "API.json"
# csv_file_path = 'images/metadata/images.csv'
# json_directory = 'listings/metadata/'
# images_folder = 'images/small/'
# output_folder = 'vqa_dataset/'

# # ========== API KEY MANAGEMENT ==========
# def load_api_keys():
#     with open(API_FILE, "r") as f:
#         return json.load(f)

# def save_api_keys(api_keys):
#     with open(API_FILE, "w") as f:
#         json.dump(api_keys, f, indent=2)

# def rotate_api_key(api_keys):
#     failed_key = api_keys.pop(0)
#     api_keys.append(failed_key)
#     save_api_keys(api_keys)
#     print(f"🔁 Rotated API key. Using new key: {api_keys[0]}")
#     return api_keys

# api_keys = load_api_keys()
# if not api_keys:
#     print("❌ No API keys found in API.json.")
#     exit(1)

# genai.configure(api_key=api_keys[0])
# safety_settings = {
#     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
# }

# model = genai.GenerativeModel(
#     model_name="models/gemini-1.5-flash",
#     safety_settings=safety_settings,
# )

# # ========== VQA GENERATION ==========
# def generate_vqa(image_path, metadata, output_json_path):
#     global api_keys, model
#     with Image.open(image_path) as img:
#         prompt = (
#             "You are provided with an image of a product along with its metadata from an e-commerce listing. Please adhere to the following instructions:\n\n"
#             "1. Analyze the image and metadata thoroughly. The image is a representation of the product, and the metadata provides descriptive attributes.\n\n"
#             "2. Based ONLY on the information visible in the image and described in the metadata, generate **15 simple, one-word Visual Question Answer (VQA) pairs**.\n\n"
#             "3. Do not hallucinate or generate answers that are not supported by the image or metadata. All answers should be directly inferable from the provided content.\n\n"
#             "4. The output must strictly adhere to the following JSON format:\n"
#             "   [{'question': '...', 'answer': '...'}, ...]\n\n"
#             "5. Ensure the questions cover diverse aspects of the product, such as its color, material, features, and other relevant attributes. Avoid repetition.\n\n"
#             "6. Make sure questions are detailed, about 10 words, and provide enough context.\n"
#             "7. AND ONLY RETURN OUTPUT IN THE JSON FORMAT AS TOLD\n"
#         )

#         while True:
#             try:
#                 response = model.generate_content([prompt, img, json.dumps(metadata or {})], stream=False)
#                 result = response.text
#                 break
#             except Exception as e:
#                 if "429" in str(e):
#                     print("⚠️ Rate limit hit. Rotating API key...")
#                     api_keys = rotate_api_key(api_keys)
#                     genai.configure(api_key=api_keys[0])
#                     model = genai.GenerativeModel(
#                         model_name="models/gemini-1.5-flash",
#                         safety_settings=safety_settings,
#                     )
#                 else:
#                     print("❌ Error during generation:", str(e))
#                     return

#         try:
#             cleaned_result = result.strip().lstrip('```json').rstrip('```').strip()
#             qa_pairs = json.loads(cleaned_result)
#             with open(output_json_path, "w", encoding="utf-8") as out_f:
#                 json.dump(qa_pairs, out_f, indent=2)
#             print(f"✅ Saved VQA to {output_json_path}")
#         except json.JSONDecodeError:
#             print("❌ GPT output was not valid JSON.")
#             print(result)

# # ========== JSON LOOKUP ==========
# def search_json_entry(directory, image_id, check_main_only):
#     for filename in os.listdir(directory):
#         if filename.endswith(".json"):
#             filepath = os.path.join(directory, filename)
#             try:
#                 with open(filepath, 'r', encoding='utf-8') as file:
#                     lines = file.readlines()
#                     for line_num, line in enumerate(lines, start=1):
#                         line = line.strip()
#                         if not line:
#                             continue
#                         try:
#                             entry = json.loads(line)
#                             if not isinstance(entry, dict):
#                                 continue

#                             main_match = entry.get("main_image_id") == image_id
#                             other_match = not check_main_only and image_id in entry.get("other_image_ids", [])

#                             if main_match or other_match:
#                                 print(f"✅ Found {image_id} in {filename} (line {line_num})")
#                                 return entry
#                         except json.JSONDecodeError:
#                             print(f"⚠️ Skipping malformed JSON at {filename} line {line_num}")
#             except Exception as e:
#                 print(f"❌ Error reading {filename}: {e}")
#     return None

# # ========== MAIN LOOP ==========
# def main(check_main_image_id=0):
#     os.makedirs(output_folder, exist_ok=True)

#     loop = 1000
#     while loop > 0:
#         with open(csv_file_path, mode='r') as file:
#             rows = list(csv.DictReader(file))
#             image_metadata = random.choice(rows)

#         image_id = image_metadata['image_id']
#         image_path = os.path.join(images_folder, image_metadata['path'])
#         print(f"\n🔍 Checking image_id: {image_id}")

#         if not os.path.exists(image_path):
#             print(f"❌ Image not found: {image_path}")
#             continue

#         metadata = None
#         if check_main_image_id == 1:
#             metadata = search_json_entry(json_directory, image_id, check_main_only=True)
#             if not metadata:
#                 print(f"❌ image_id {image_id} not found as main_image_id.")
#                 continue
#         elif check_main_image_id == 0:
#             metadata = None  # skip lookup entirely

#         print(f"✅ Image {'and metadata ' if metadata else ''}found. Proceeding to VQA generation.")
#         output_json_path = os.path.join(output_folder, f"{image_id}.json")
#         generate_vqa(image_path, metadata, output_json_path)

#         time.sleep(5)
#         loop -= 1

# # ========== ENTRY POINT ==========
# if __name__ == "__main__":
#     main(check_main_image_id=1)  # 0: no check, 1: main_image_id only













# import csv
# import os
# import random
# import json
# import google.generativeai as genai
# from PIL import Image
# import time
# from google.generativeai.types import HarmCategory, HarmBlockThreshold

# # MULTI API KEY MANAGEMENT
# API_FILE = "API.json"

# def load_api_keys():
#     with open(API_FILE, "r") as f:
#         return json.load(f)

# def save_api_keys(api_keys):
#     with open(API_FILE, "w") as f:
#         json.dump(api_keys, f, indent=2)

# def rotate_api_key(api_keys):
#     failed_key = api_keys.pop(0)
#     api_keys.append(failed_key)
#     save_api_keys(api_keys)
#     print(f"🔁 Rotated API key. Using new key: {api_keys[0]}")
#     return api_keys

# # Load and configure first key
# api_keys = load_api_keys()
# if not api_keys:
#     print("❌ No API keys found in API.json.")
#     exit(1)

# # Configure Gemini with first key
# genai.configure(api_key=api_keys[0])

# # Setup model
# safety_settings = {
#     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
# }

# model = genai.GenerativeModel(
#     model_name="models/gemini-1.5-flash",
#     safety_settings=safety_settings,
# )



# def generate_vqa(image_path, metadata, output_json_path):
#     global api_keys, model  # To allow reassignment after rotation
#     with Image.open(image_path) as img:
#         prompt = (
#             "You are provided with an image of a product along with its metadata from an e-commerce listing. Please adhere to the following instructions:\n\n"
#             "1. Analyze the image and metadata thoroughly. The image is a representation of the product, and the metadata provides descriptive attributes.\n\n"
#             "2. Based ONLY on the information visible in the image and described in the metadata, generate **15 simple, one-word Visual Question Answer (VQA) pairs**.\n\n"
#             "3. Do not hallucinate or generate answers that are not supported by the image or metadata. All answers should be directly inferable from the provided content.\n\n"
#             "4. The output must strictly adhere to the following JSON format:\n"
#             "   [{'question': '...', 'answer': '...'}, ...]\n\n"
#             "5. Ensure the questions cover diverse aspects of the product, such as its color, material, features, and other relevant attributes. Avoid repetition.\n\n"
#             "Please ensure the answers are accurate and concise, and DO NOT EXCEED ONE WORD PER ANSWER.\n"
#             "6. Make sure questions are detailed, about 10 words, and provide enough context.\n"
#             "7. AND ONLY RETURN OUTPUT IN THE JSON FOMRAT AS TOLD\n"
#         )

#         while True:
#             try:
#                 response = model.generate_content([prompt, img, json.dumps(metadata)], stream=False)
#                 result = response.text
#                 break  # success
#             except Exception as e:
#                 if "429" in str(e):
#                     print("⚠️ Rate limit hit. Rotating API key...")
#                     api_keys = rotate_api_key(api_keys)
#                     genai.configure(api_key=api_keys[0])
#                     model = genai.GenerativeModel(
#                         model_name="models/gemini-1.5-flash",
#                         safety_settings=safety_settings,
#                     )
#                     continue
#                 else:
#                     print("❌ Error during generation:", str(e))
#                     return

#         try:
#             cleaned_result = result.strip().lstrip('```json').rstrip('```').strip()
#             qa_pairs = json.loads(cleaned_result)
#             with open(output_json_path, "w", encoding="utf-8") as out_f:
#                 json.dump(qa_pairs, out_f, indent=2)
#             print(f"✅ Saved VQA to {output_json_path}")
#         except json.JSONDecodeError:
#             print("❌ GPT output was not valid JSON.")
#             print(result)


# def search_strings_in_files(directory, strings_to_search):
#     """Searches for each string in strings_to_search across all JSON files in the directory."""
#     found_lines = []
#     # Ensure the directory exists
#     if not os.path.isdir(directory):
#         print(f"❌ The directory {directory} does not exist or is not a directory.")
#         return found_lines

#     # Loop through all files in the directory
#     for filename in os.listdir(directory):
#         if filename.endswith(".json"):
#             filepath = os.path.join(directory, filename)
#             try:
#                 with open(filepath, 'r', encoding='utf-8') as file:
#                     lines = file.readlines()
#                     for search_string in strings_to_search:
#                         for line_num, line in enumerate(lines, start=1):
#                             if search_string in line:
#                                 found_lines.append(f"Found '{search_string}' in {filename} (Line {line_num}): {line.strip()}")
#             except Exception as e:
#                 print(f"❌ Error reading {filename}: {e}")
#     return found_lines

# # def search_json_entry(directory, image_id):
# #     """Search for the image_id in all JSON files in the directory."""
# #     strings_to_search = [f'"main_image_id": "{image_id}"']  # format search string with image_id
# #     found_lines = search_strings_in_files(directory, strings_to_search)

# #     if found_lines:
# #         # Return the first match found for image_id under 'main_image_id'
# #         for line in found_lines:
# #             print(line)
# #             return line  # You can modify to return a parsed JSON object if needed
# #     return None

# def search_json_entry(directory, image_id, check_main_only):
#     """Search for image_id in JSON files. If check_main_only is False, also check 'other_image_id'."""
#     found_lines = []

#     for filename in os.listdir(directory):
#         if filename.endswith(".json"):
#             filepath = os.path.join(directory, filename)
#             try:
#                 with open(filepath, 'r', encoding='utf-8') as file:
#                     content = json.load(file)
#                     main_match = content.get("main_image_id") == image_id
#                     other_match = not check_main_only and image_id in content.get("other_image_ids", [])

#                     if main_match or other_match:
#                         print(f"✅ Found {image_id} in {filename}")
#                         return content
#             except Exception as e:
#                 print(f"❌ Error reading {filename}: {e}")

#     return None

# # Example usage:
# # In your `main` function, update the path for JSON file directory:
# def main(check_main_image_id=1):

#     csv_file_path = 'images/metadata/images.csv'
#     json_directory = 'listings/metadata/'  # Folder containing JSON files
#     images_folder = 'images/small/'
#     output_folder = 'vqa_dataset/'

#     os.makedirs(output_folder, exist_ok=True)

#     loop = 1000
#     while loop > 0:
#         with open(csv_file_path, mode='r') as file:
#             rows = list(csv.DictReader(file))
#             image_metadata = random.choice(rows)

#         image_id = image_metadata['image_id']
#         image_path = os.path.join(images_folder, image_metadata['path'])
#         print(f"\n🔍 Checking image_id: {image_id}")

#         if not os.path.exists(image_path):
#             print(f"❌ Image not found: {image_path}")
#             continue

#         # Conditional check based on the parameter 'check_main_image_id'
#         if check_main_image_id == 2:
#             # Check if image_id is in main_image_id (not other_image_id)
#             # metadata = search_json_entry(json_directory, image_id)
#             metadata = search_json_entry(json_directory, image_id, check_main_image_id == 2)
#             if not metadata:
#                 print(f"❌ image_id {image_id} not found as main_image_id.")
#                 continue
#         else:
#             metadata = search_json_entry(json_directory, image_id)
#             if not metadata:
#                 print(f"❌ image_id {image_id} not found.")
#                 continue

#         print(f"✅ Image found and valid for VQA generation.")
#         output_json_path = os.path.join(output_folder, f"{image_id}.json")
#         generate_vqa(image_path, metadata, output_json_path)
#         sec = 5
#         print(f"please wait for {sec} seconds to stay under the rate limit")
#         time.sleep(sec)  # stay under the rate limit
#         loop -= 1

# if __name__ == "__main__":
#     main(check_main_image_id=1)
#     # If check_main_image_id == 1, the script proceeds with the image without validating it against the main_image_id.
#     # If check_main_image_id == 2, the script checks for the main_image_id using the search_json_entry function.















# # WORKS PERFECTLY WITH ONLY ONE API KEY
# import csv
# import os
# import random
# import json
# import google.generativeai as genai
# from PIL import Image
# import time
# from google.generativeai.types import HarmCategory, HarmBlockThreshold

# safety_settings = {
#     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
# }

# # Setup Gemini API
# genai.configure(api_key="AIzaSyA_DP_WqEsK-W7Sa5j7a7HmjtWULjgWlzs")
# # for m in genai.list_models():
# #     print(m.name)

# model = genai.GenerativeModel(
#     model_name="models/gemini-1.5-flash", # faster responses with slightly cheaper cost
#     safety_settings=safety_settings,
# )


# def generate_vqa(image_path, metadata, output_json_path):
#     with Image.open(image_path) as img:
#         prompt = (
#             "You are provided with an image of a product along with its metadata from an e-commerce listing. Please adhere to the following instructions:\n\n"
#             "1. Analyze the image and metadata thoroughly. The image is a representation of the product, and the metadata provides descriptive attributes.\n\n"
#             "2. Based ONLY on the information visible in the image and described in the metadata, generate **15 simple, one-word Visual Question Answer (VQA) pairs**.\n\n"
#             "3. Do not hallucinate or generate answers that are not supported by the image or metadata. All answers should be directly inferable from the provided content.\n\n"
#             "4. The output must strictly adhere to the following JSON format:\n"
#             "   [{'question': '...', 'answer': '...'}, ...]\n\n"
#             "5. Ensure the questions cover diverse aspects of the product, such as its color, material, features, and other relevant attributes. Avoid repetition.\n\n"
#             "Please ensure the answers are accurate and concise, and DO NOT EXCEED ONE WORD PER ANSWER.\n"
#             "6. Make sure questions are detailed, about 10 words, and provide enough context.\n"
#             "7. AND ONLY RETURN OUTPUT IN THE JSON FOMRAT AS TOLD\n"
#         )

#         response = model.generate_content([prompt, img, json.dumps(metadata)], stream=False)
#         result = response.text

#         # Clean the result to remove unwanted markdown markers and parse it
#         try:
#             # Remove the markdown block markers
#             cleaned_result = result.strip().lstrip('```json').rstrip('```').strip()
#             print("cleaned_result:", cleaned_result)
#             # Parse the cleaned result into a JSON object
#             qa_pairs = json.loads(cleaned_result)

#             # Save output to .json
#             with open(output_json_path, "w", encoding="utf-8") as out_f:
#                 json.dump(qa_pairs, out_f, indent=2)
#             print(f"✅ Saved VQA to {output_json_path}")
#         except json.JSONDecodeError:
#             print("❌ GPT output was not valid JSON.")
#             print(result)

# def search_strings_in_files(directory, strings_to_search):
#     """Searches for each string in strings_to_search across all JSON files in the directory."""
#     found_lines = []
#     # Ensure the directory exists
#     if not os.path.isdir(directory):
#         print(f"❌ The directory {directory} does not exist or is not a directory.")
#         return found_lines

#     # Loop through all files in the directory
#     for filename in os.listdir(directory):
#         if filename.endswith(".json"):
#             filepath = os.path.join(directory, filename)
#             try:
#                 with open(filepath, 'r', encoding='utf-8') as file:
#                     lines = file.readlines()
#                     for search_string in strings_to_search:
#                         for line_num, line in enumerate(lines, start=1):
#                             if search_string in line:
#                                 found_lines.append(f"Found '{search_string}' in {filename} (Line {line_num}): {line.strip()}")
#             except Exception as e:
#                 print(f"❌ Error reading {filename}: {e}")
#     return found_lines

# def search_json_entry(directory, image_id):
#     """Search for the image_id in all JSON files in the directory."""
#     strings_to_search = [f'"main_image_id": "{image_id}"']  # format search string with image_id
#     found_lines = search_strings_in_files(directory, strings_to_search)

#     if found_lines:
#         # Return the first match found for image_id under 'main_image_id'
#         for line in found_lines:
#             print(line)
#             return line  # You can modify to return a parsed JSON object if needed
#     return None

# # Example usage:
# # In your `main` function, update the path for JSON file directory:
# def main(check_main_image_id=2):

#     csv_file_path = 'images/metadata/images.csv'
#     json_directory = 'listings/metadata/'  # Folder containing JSON files
#     images_folder = 'images/small/'
#     output_folder = 'vqa_dataset/'

#     os.makedirs(output_folder, exist_ok=True)

#     loop = 1
#     while loop > 0:
#         with open(csv_file_path, mode='r') as file:
#             rows = list(csv.DictReader(file))
#             image_metadata = random.choice(rows)

#         image_id = image_metadata['image_id']
#         image_path = os.path.join(images_folder, image_metadata['path'])
#         print(f"\n🔍 Checking image_id: {image_id}")

#         if not os.path.exists(image_path):
#             print(f"❌ Image not found: {image_path}")
#             continue

#         # Conditional check based on the parameter 'check_main_image_id'
#         if check_main_image_id == 2:
#             # Check if image_id is in main_image_id (not other_image_id)
#             metadata = search_json_entry(json_directory, image_id)
#             if not metadata:
#                 print(f"❌ image_id {image_id} not found as main_image_id.")
#                 continue
#         else:
#             metadata = search_json_entry(json_directory, image_id)
#             if not metadata:
#                 print(f"❌ image_id {image_id} not found.")
#                 continue

#         print(f"✅ Image found and valid for VQA generation.")
#         output_json_path = os.path.join(output_folder, f"{image_id}.json")
#         generate_vqa(image_path, metadata, output_json_path)
#         sec = 5
#         print(f"please wait for {sec} seconds to stay under the rate limit")
#         time.sleep(sec)  # stay under the rate limit
#         loop -= 1

# if __name__ == "__main__":
#     main(check_main_image_id=2)
#     # If check_main_image_id == 1, the script proceeds with the image without validating it against the main_image_id.
#     # If check_main_image_id == 2, the script checks for the main_image_id using the search_json_entry function.















# import csv
# import os
# import random
# import json
# import google.generativeai as genai
# from PIL import Image
# import time
# from google.generativeai.types import HarmCategory, HarmBlockThreshold

# safety_settings = {
#     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
# }

# # Setup Gemini API
# genai.configure(api_key="AIzaSyA_DP_WqEsK-W7Sa5j7a7HmjtWULjgWlzs")
# for m in genai.list_models():
#     print(m.name)

# model = genai.GenerativeModel(
#     model_name="models/gemini-1.5-flash",
#     safety_settings=safety_settings,
# )

# # model = genai.GenerativeModel("gemini-1.5-flash") # faster responses with slightly cheaper cost

# def generate_vqa(image_path, metadata, output_json_path):
#     with Image.open(image_path) as img:
#         prompt = (
#             "You are provided with an image of a product along with its metadata from an e-commerce listing. Please adhere to the following instructions:\n\n"
#             "1. Analyze the image and metadata thoroughly. The image is a representation of the product, and the metadata provides descriptive attributes.\n\n"
#             "2. Based ONLY on the information visible in the image and described in the metadata, generate **15 simple, one-word Visual Question Answer (VQA) pairs**.\n\n"
#             "3. Do not hallucinate or generate answers that are not supported by the image or metadata. All answers should be directly inferable from the provided content.\n\n"
#             "4. The output must strictly adhere to the following JSON format:\n"
#             "   [{'question': '...', 'answer': '...'}, ...]\n\n"
#             "5. Ensure the questions cover diverse aspects of the product, such as its color, material, features, and other relevant attributes. Avoid repetition.\n\n"
#             "Please ensure the answers are accurate and concise, and DO NOT EXCEED ONE WORD PER ANSWER.\n"
#             "6. AND ONLY RETURN OUTPUT IN THE JSON FOMRAT AS TOLD\n"
#         )

#         response = model.generate_content([prompt, img, json.dumps(metadata)], stream=False)
#         result = response.text
#         # print("response:", response)
#         # print("result:", result)

#         # Clean the result to remove unwanted markdown markers and parse it
#         try:
#             # Remove the markdown block markers
#             cleaned_result = result.strip().lstrip('```json').rstrip('```').strip()
#             print("cleaned_result:", cleaned_result)
#             # Parse the cleaned result into a JSON object
#             qa_pairs = json.loads(cleaned_result)

#             # Save output to .json
#             with open(output_json_path, "w", encoding="utf-8") as out_f:
#                 json.dump(qa_pairs, out_f, indent=2)
#             print(f"✅ Saved VQA to {output_json_path}")
#         except json.JSONDecodeError:
#             print("❌ GPT output was not valid JSON.")
#             print(result)

# def search_json_entry(json_file, image_id):
#     """Extract full product metadata entry containing the given main_image_id"""
#     with open(json_file, 'r', encoding='utf-8') as file:
#         for line in file:
#             if '"main_image_id":' in line and image_id in line:
#                 try:
#                     return json.loads(line.strip().rstrip(','))
#                 except:
#                     return None
#     return None

# def main():
#     csv_file_path = 'images/metadata/images.csv'
#     json_file = 'listings/metadata/listings_5.json'
#     images_folder = 'images/small/'
#     output_folder = 'vqa_dataset/'
#     os.makedirs(output_folder, exist_ok=True)

#     loop = 1
#     while loop > 0:
#         with open(csv_file_path, mode='r') as file:
#             rows = list(csv.DictReader(file))
#             image_metadata = random.choice(rows)

#         image_id = image_metadata['image_id']
#         image_path = os.path.join(images_folder, image_metadata['path'])
#         print(f"\n🔍 Checking image_id: {image_id}")

#         if not os.path.exists(image_path):
#             print(f"❌ Image not found: {image_path}")
#             continue

#         # Check if image_id is in main_image_id (not other_image_id)
#         metadata = search_json_entry(json_file, image_id)
#         if not metadata:
#             print(f"❌ image_id {image_id} not found as main_image_id.")
#             continue

#         print(f"✅ Image found and valid for VQA generation.")
#         output_json_path = os.path.join(output_folder, f"{image_id}.json")
#         generate_vqa(image_path, metadata, output_json_path)
#         sec = 5
#         print(f"please wait for {sec} seconds to stay under the rate limit")
#         time.sleep(sec) # stay under the rate limit
#         loop -= 1

# if __name__ == "__main__":
#     main()














# import csv
# import os
# import random
# import json
# import google.generativeai as genai
# from PIL import Image
# import time
# from google.generativeai.types import HarmCategory, HarmBlockThreshold

# safety_settings = {
#     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
# }


# # Setup Gemini API
# genai.configure(api_key="AIzaSyA_DP_WqEsK-W7Sa5j7a7HmjtWULjgWlzs")
# # # print the models in the list
# # for m in genai.list_models():
# #     print(m.name)

# model = genai.GenerativeModel(
#     model_name="models/gemini-1.5-flash",
#     safety_settings=safety_settings,
# )

# # model = genai.GenerativeModel("gemini-1.5-flash") # faster responses with slightly cheaper cost

# def generate_vqa(image_path, metadata, output_json_path):
#     with Image.open(image_path) as img:
#         prompt = (
#             "You are provided with an image of a product along with its metadata from an e-commerce listing. Please adhere to the following instructions:\n\n"
#             "1. Analyze the image and metadata thoroughly. The image is a representation of the product, and the metadata provides descriptive attributes.\n\n"
#             "2. Based ONLY on the information visible in the image and described in the metadata, generate **15 simple, one-word Visual Question Answer (VQA) pairs**.\n\n"
#             "3. Do not hallucinate or generate answers that are not supported by the image or metadata. All answers should be directly inferable from the provided content.\n\n"
#             "4. The output must strictly adhere to the following JSON format:\n"
#             "   [{'question': '...', 'answer': '...'}, ...]\n\n"
#             "5. Ensure the questions cover diverse aspects of the product, such as its color, material, features, and other relevant attributes. Avoid repetition.\n\n"
#             "Please ensure the answers are accurate and concise, and DO NOT EXCEED ONE WORD PER ANSWER.\n"
#             "6. AND ONLY RETURN OUTPUT IN THE JSON FOMRAT AS TOLD\n"
#         )


#         response = model.generate_content([prompt, img, json.dumps(metadata)], stream=False)

#         result = response.text
#         # print("response:", response)
#         # print("result:", result)
#         # Save output to .json
#         try:
#             cleaned_result = result.strip("```json").strip("```").strip()
#             cleaned_result = result.strip("```")
#             print("cleaned_result:", cleaned_result)
#             qa_pairs = json.loads(cleaned_result)
#             with open(output_json_path, "w", encoding="utf-8") as out_f:
#                 json.dump(qa_pairs, out_f, indent=2)
#             print(f"✅ Saved VQA to {output_json_path}")
#         except json.JSONDecodeError:
#             print("❌ GPT output was not valid JSON.")
#             print(cleaned_result)

# def search_json_entry(json_file, image_id):
#     """Extract full product metadata entry containing the given main_image_id"""
#     with open(json_file, 'r', encoding='utf-8') as file:
#         for line in file:
#             if '"main_image_id":' in line and image_id in line:
#                 try:
#                     return json.loads(line.strip().rstrip(','))
#                 except:
#                     return None
#     return None

# def main():
#     csv_file_path = 'images/metadata/images.csv'
#     json_file = 'listings/metadata/listings_5.json'
#     images_folder = 'images/small/'
#     output_folder = 'vqa_dataset/'
#     os.makedirs(output_folder, exist_ok=True)

#     loop = 1
#     while loop > 0:
#         with open(csv_file_path, mode='r') as file:
#             rows = list(csv.DictReader(file))
#             image_metadata = random.choice(rows)

#         image_id = image_metadata['image_id']
#         image_path = os.path.join(images_folder, image_metadata['path'])
#         print(f"\n🔍 Checking image_id: {image_id}")

#         if not os.path.exists(image_path):
#             print(f"❌ Image not found: {image_path}")
#             continue

#         # Check if image_id is in main_image_id (not other_image_id)
#         metadata = search_json_entry(json_file, image_id)
#         if not metadata:
#             print(f"❌ image_id {image_id} not found as main_image_id.")
#             continue

#         print(f"✅ Image found and valid for VQA generation.")
#         output_json_path = os.path.join(output_folder, f"{image_id}.json")
#         generate_vqa(image_path, metadata, output_json_path)
#         sec = 0
#         print(f"please wait for {sec} seconds to stay under the rate limit")
#         time.sleep(sec) # stay under the rate limit
#         loop -= 1

# if __name__ == "__main__":
#     main()
