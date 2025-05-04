BASE_MODEL_PATH = 'meta-llama/Llama-3.2-1B-Instruct'

# a list of prompts used to sample data from the given LLM, 
# please change according to the LLM & your use cases.
INPUT_FILE = './workspace/inference_data/distillation_prompts.json' 

DATASET = 'Llama-3.2-1B-Instruct' # Updated dataset name
DATA_PATH = f'./workspace/hmm_data/{DATASET}'
LVD_SIZE = 100000
CHUNK_SIZE = 100000
DEV_SIZE = 20000
TOTAL_CHUNKS = 100
SEQUENCE_LEN = 32

import os


CUDA_CORES = '0' # Set based on nvidia-smi output
BATCH_SIZE = 32 # Reduced batch size further to prevent OOM


# create data path
os.system(f'mkdir -p {DATA_PATH}')


# sample LVD_SIZE examples for initializing hmm parameters via latent variable distillation (LVD)
cmd = f'CUDA_VISIBLE_DEVICES={CUDA_CORES} torchrun --standalone --nproc_per_node=gpu \
    sample_data.py \
    --model_name_or_path {BASE_MODEL_PATH} \
    --tokenizer_name_or_path {BASE_MODEL_PATH} \
    --input_file {INPUT_FILE} --chunk_size {LVD_SIZE} \
    --batch_size {BATCH_SIZE} --max_new_tokens {SEQUENCE_LEN} \
    --save_embeddings --output_file {DATA_PATH}/{DATASET}.lvd'.strip()
print(cmd)


# sample TOTAL_CHUNKS chunks of training examples as the training set
cmd = f'CUDA_VISIBLE_DEVICES={CUDA_CORES} torchrun --standalone --nproc_per_node=gpu \
    sample_data.py \
    --model_name_or_path {BASE_MODEL_PATH} \
    --tokenizer_name_or_path {BASE_MODEL_PATH} \
    --input_file {INPUT_FILE} --chunk_size {CHUNK_SIZE} --total_chunks {TOTAL_CHUNKS} \
    --batch_size {BATCH_SIZE} --max_new_tokens {SEQUENCE_LEN} \
    --output_file {DATA_PATH}/{DATASET}.train'.strip()
print(cmd)


# sample DEV_SIZE examples as the dev set
cmd = f'CUDA_VISIBLE_DEVICES={CUDA_CORES} torchrun --standalone --nproc_per_node=gpu \
    sample_data.py \
    --model_name_or_path {BASE_MODEL_PATH} \
    --tokenizer_name_or_path {BASE_MODEL_PATH} \
    --input_file {INPUT_FILE} --chunk_size {DEV_SIZE} \
    --batch_size {BATCH_SIZE} --max_new_tokens {SEQUENCE_LEN} \
    --output_file {DATA_PATH}/{DATASET}.dev'.strip()
print(cmd)



import os
from transformers import AutoTokenizer

# specify the HMM size
HIDDEN_STATES = 4096

# get vocab_size and eos_token_id; might vary for different models #
__tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
VOCAB_SIZE = __tokenizer.vocab_size
EOS_TOKEN_ID = __tokenizer.eos_token_id
####################################################################

HMM_MODEL_ID = f'hmm_{DATASET}_{HIDDEN_STATES}'
HMM_MODEL_PATH = f'./workspace/models/{HMM_MODEL_ID}/'

_ = os.system(f'mkdir -p {HMM_MODEL_PATH}')


import os

CUDA_CORES = '0,1,2,3,4,5'
SEQUENCES_FILE = f'{DATA_PATH}/{DATASET}.lvd'
EMEBEDDINGS_FILE = f'{DATA_PATH}/{DATASET}.lvd.embeddings'

# latent variable distillation
cmd = f'CUDA_VISIBLE_DEVICES={CUDA_CORES} python lvd_hmm.py \
    --sequences_file {SEQUENCES_FILE} --embeddings_file {EMEBEDDINGS_FILE} \
    --hidden_states {HIDDEN_STATES} --vocab_size {VOCAB_SIZE} --eos_token_id {EOS_TOKEN_ID} \
    --kmeans_iterations 100 --pseudocount 0.001 \
    --output_file {HMM_MODEL_PATH}/checkpoint-0'
print(cmd)


import os

os.system('mkdir -p ./workspace/logs')
LOG_FILE=f'./workspace/logs/{HMM_MODEL_ID}_log.txt'

CUDA_CORES = '0,1,2,3,4,5'
BATCH_SIZE = 256
SAVE_PER_STEP = 10
DROPOUT = 0.01

# EM training schedule:
# 1. train for 10 EM steps, each step using 1 chunk of data
# 2. train for 5 EM steps, each step using 2 chunks of data
# 3. train for 4 EM steps, each step using 5 chunks of data
# 4. train for 4 EM steps, each step using 10 chunks of data
# 5. train for 4 EM steps, each step using 20 chunks of data
# 6. train for 1 EM steps, each step using 40 chunks of data
EM_SCHEDULE = "\"10,1;5,2;4,5;4,10;4,20;1,40\""

cmd = f'CUDA_VISIBLE_DEVICES={CUDA_CORES} torchrun --standalone --nproc_per_node=gpu train_hmm.py \
    --model_path {HMM_MODEL_PATH} --checkpoint 0 --save_per_step {SAVE_PER_STEP} \
    --data_path {DATA_PATH} --dataset {DATASET} --total_chunks {TOTAL_CHUNKS} --batch_size {BATCH_SIZE} \
    --em_schedule {EM_SCHEDULE} --dropout {DROPOUT} --log_file {LOG_FILE}'.strip()
print(cmd)
