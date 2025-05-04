import json
from datasets import load_dataset, get_dataset_split_names

# Login using e.g. `huggingface-cli login` to access this dataset
dataset_name = "nvidia/Llama-Nemotron-Post-Training-Dataset"

# Get and print the available splits
print(f"Available splits in {dataset_name}:")
splits = get_dataset_split_names(dataset_name)
print(splits)
print("-" * 20) # Separator

# Load the desired split
ds = load_dataset(dataset_name, split='chat')

# Get the first 2 rows
sample = ds.select(range(2))

# Collect the 'content' from the 'input' field
content_list = []
for row in sample:
    # Assuming 'input' is a list containing one dictionary
    if row['input'] and isinstance(row['input'], list) and len(row['input']) > 0:
        input_dict = row['input'][0]
        if 'content' in input_dict:
            content_list.append(input_dict['content'])

# Print the list as a JSON array
print(json.dumps(content_list, indent=2)) # Using indent for readability
