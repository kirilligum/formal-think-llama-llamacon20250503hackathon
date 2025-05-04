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

    print(f"Loading 2 samples from split: {split_name}")
    # Load the current split
    ds = load_dataset(dataset_name, split=split_name)

    # Get the first 2 rows
    sample = ds.select(range(2))

    # Collect the 'content' from the 'input' field for this split's sample
    for row in sample:
        # Assuming 'input' is a list containing one dictionary
        if row['input'] and isinstance(row['input'], list) and len(row['input']) > 0:
            input_dict = row['input'][0]
            if 'content' in input_dict:
                all_content_list.append(input_dict['content'])

# Print the combined list as a JSON array
print("-" * 20) # Separator
print("Combined content from all sampled splits:")
print(json.dumps(all_content_list, indent=2)) # Using indent for readability
