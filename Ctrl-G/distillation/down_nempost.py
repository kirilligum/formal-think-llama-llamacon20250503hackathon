import json
from datasets import load_dataset, get_dataset_split_names

# Login using e.g. `huggingface-cli login` to access this dataset
dataset_name = "nvidia/Llama-Nemotron-Post-Training-Dataset"

# Get and print the available splits
print(f"Available splits in {dataset_name}:")
splits = get_dataset_split_names(dataset_name)
print(splits)
print("-" * 20) # Separator

all_content_list = []

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
                all_content_list.append(input_dict['content'])

# Save the combined list to a JSON file
output_filename = "lnptd_prompts_en.json"
print("-" * 20) # Separator
print(f"Saving combined content to {output_filename}...")
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(all_content_list, f, indent=2, ensure_ascii=False)

print(f"Successfully saved {len(all_content_list)} prompts to {output_filename}.")
