# THIS ONE ONLY CHECKS IF THE IMAGE IN IN THE MAIN_IMAGE_ID
import csv
import os
import random

def search_strings_in_files(directory, strings_to_search):
    """Searches for each string in strings_to_search across all JSON files in the directory."""
    found_lines = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                for search_string in strings_to_search:
                    for line_num, line in enumerate(lines, start=1):
                        if search_string in line:
                            found_lines.append(f"Found '{search_string}' in {filename} (Line {line_num}): {line.strip()}")
    return found_lines

def get_image_path(image_path, images_folder):
    """Constructs and returns the image file path from the images folder based on image_path."""
    image_path = os.path.join(images_folder, image_path)
    print(f"Constructed image path: {image_path}")
    return image_path

def main():
    csv_file_path = 'images/metadata/images.csv'
    json_folder = 'listings/metadata/'
    images_folder = 'images/small/'
    loop = 5
    while(loop > 0):
        with open(csv_file_path, mode='r') as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            # Randomly select a row from the list
            image_metadata = random.choice(rows)

        image_id = image_metadata['image_id']
        image_path = image_metadata['path']
        print(f"Searching for image_id: {image_id}")

        # Check if image_id is in main_image_id, if it is in other_image_id, skip this one
        found_in_main = False
        with open(json_folder + 'listings_5.json', 'r', encoding='utf-8') as file:
            data = file.readlines()
            for line in data:
                if '"main_image_id":' in line and image_id in line:
                    found_in_main = True
                    break
                elif '"other_image_id":' in line and image_id in line:
                    found_in_main = False
                    break

        if not found_in_main:
            print(f"Image id {image_id} is not in main_image_id, skipping.")
            continue

        # Proceed with searching in the files if found in main_image_id
        found_lines = search_strings_in_files(json_folder, [image_id])

        if found_lines:
            print(f"Found the following results in JSON files:")
            for line in found_lines:
                print(line)
        else:
            print(f"No JSON data found for image_id: {image_id}")
            return

        image_path = get_image_path(image_path, images_folder)

        if os.path.exists(image_path):
            print(f"Image found at: {image_path}")
        else:
            print(f"Image not found for image_id: {image_id}")
        loop -= 1

if __name__ == "__main__":
    main()












# THIS ONE DOES NOT CHECK IF IT IS IN THE MAIN_IMAGE_ID, BRINGS THE RESPECTIVE RESULT
import csv
import os
import random

def search_strings_in_files(directory, strings_to_search):
    """Searches for each string in strings_to_search across all JSON files in the directory."""
    found_lines = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                for search_string in strings_to_search:
                    for line_num, line in enumerate(lines, start=1):
                        if search_string in line:
                            found_lines.append(f"Found '{search_string}' in {filename} (Line {line_num}): {line.strip()}")
    return found_lines

def get_image_path(image_path, images_folder):
    """Constructs and returns the image file path from the images folder based on image_path."""
    image_path = os.path.join(images_folder, image_path)
    print(f"Constructed image path: {image_path}")
    return image_path

def main():
    csv_file_path = 'images/metadata/images.csv'
    json_folder = 'listings/metadata/'
    images_folder = 'images/small/'
    loop = 1
    while(loop>0):
        with open(csv_file_path, mode='r') as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            # Randomly select a row from the list
            image_metadata = random.choice(rows)

        image_id = image_metadata['image_id']
        image_path = image_metadata['path']
        print(f"Searching for image_id: {image_id}")

        found_lines = search_strings_in_files(json_folder, [image_id])

        if found_lines:
            print(f"Found the following results in JSON files:")
            for line in found_lines:
                print(line)
        else:
            print(f"No JSON data found for image_id: {image_id}")
            return

        image_path = get_image_path(image_path, images_folder)

        if os.path.exists(image_path):
            print(f"Image found at: {image_path}")
        else:
            print(f"Image not found for image_id: {image_id}")
        loop -= 1

if __name__ == "__main__":
    main()

