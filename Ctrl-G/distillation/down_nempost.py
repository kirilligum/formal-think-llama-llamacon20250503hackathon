from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset", split='chat')

# Get the first 2 rows
sample = ds.select(range(2))

# Print the 'content' from the 'input' field
for row in sample:
    # Assuming 'input' is a list containing one dictionary
    if row['input'] and isinstance(row['input'], list) and len(row['input']) > 0:
        input_dict = row['input'][0]
        if 'content' in input_dict:
            print(input_dict['content'])
