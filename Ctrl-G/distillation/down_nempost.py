import json
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset", split='chat')

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
