from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset", split='train')

# Get the first 2 rows
sample = ds.select(range(2))

# Print the sample
for row in sample:
    print(row)
