import os
import json

directory = "vqa_dataset"
invalid_files = []
os.makedirs("invalid_files", exist_ok=True)

for filename in os.listdir(directory):
    if filename.endswith(".json"):
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if (
                isinstance(data, list) and len(data) == 2 and
                isinstance(data[0], dict) and "path" in data[0] and
                isinstance(data[1], dict) and "questions" in data[1]
            ):
                questions = data[1]["questions"]
                if (
                    isinstance(questions, list) and len(questions) == 15 and
                    all(isinstance(q, dict) and "question" in q and "answer" in q for q in questions)
                ):
                    continue  # Valid file
                else:
                    invalid_files.append(filename)
            else:
                invalid_files.append(filename)

        except Exception:
            invalid_files.append(filename)

# Report
print(f"Total files checked: {len(os.listdir(directory))}")
print(f"Invalid files: {len(invalid_files)}")
for file in invalid_files:
    print(f" - {file}")
    os.rename(os.path.join(directory, file), os.path.join("invalid_files", file))
