import json
from datasets import load_dataset, get_dataset_split_names
from lingua import LanguageDetectorBuilder, Language

# Login using e.g. `huggingface-cli login` to access this dataset
dataset_name = "nvidia/Llama-Nemotron-Post-Training-Dataset"

# Get and print the available splits
print(f"Available splits in {dataset_name}:")
splits = get_dataset_split_names(dataset_name)
print(splits)
print("-" * 20) # Separator

# Initialize language detector for English
detector = LanguageDetectorBuilder.from_languages(Language.ENGLISH).with_preloaded_language_models().build()

en_content_list = []
not_en_content_list = []

# Iterate through splits, skipping 'safety'
for split_name in splits:
    if split_name == 'safety':
        continue

    num_samples = 50
    print(f"Loading {num_samples} samples from split: {split_name}")
    # Load the current split
    ds = load_dataset(dataset_name, split=split_name)

    # Get the first num_samples rows
    # Ensure we don't request more samples than available in the split
    actual_samples = min(num_samples, len(ds))
    if actual_samples < num_samples:
        print(f"  Warning: Split '{split_name}' only has {actual_samples} rows. Taking all of them.")
    sample = ds.select(range(actual_samples))

    # Collect the 'content' from the 'input' field for this split's sample
    for row in sample:
        # Assuming 'input' is a list containing one dictionary
        if row['input'] and isinstance(row['input'], list) and len(row['input']) > 0:
            input_dict = row['input'][0]
            if 'content' in input_dict:
                original_content = input_dict['content']
                # Normalize content (convert to lowercase)
                normalized_content = original_content.lower()
                # Check if content is not empty or just whitespace before detection
                if normalized_content and not normalized_content.isspace():
                    detected_language = detector.detect_language_of(normalized_content)
                    if detected_language == Language.ENGLISH:
                        # Save the original, non-normalized content
                        en_content_list.append(original_content)
                    else:
                        # Save the original, non-normalized content
                        not_en_content_list.append(original_content)
                else:
                    # Handle empty or whitespace-only content if necessary, e.g., skip or add to a specific list
                    # print(f"Skipping empty content in split {split_name}")
                    pass # Currently skipping

# Save the lists to separate JSON files
output_filename_en = "lnptd_prompts_en.json"
output_filename_not_en = "lnptd_prompts_noten.json"

print("-" * 20) # Separator
print(f"Saving English content to {output_filename_en}...")
with open(output_filename_en, 'w', encoding='utf-8') as f:
    json.dump(en_content_list, f, indent=2, ensure_ascii=False)
print(f"Successfully saved {len(en_content_list)} English prompts to {output_filename_en}.")

print(f"Saving non-English content to {output_filename_not_en}...")
with open(output_filename_not_en, 'w', encoding='utf-8') as f:
    json.dump(not_en_content_list, f, indent=2, ensure_ascii=False)
print(f"Successfully saved {len(not_en_content_list)} non-English prompts to {output_filename_not_en}.")
