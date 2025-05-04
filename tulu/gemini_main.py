import os

device = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # set your cuda device
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

import ctrlg

BASE_MODEL_PATH = f"ctrlg/gpt2-large_common-gen"  # a gpt2-large checkpoint domain adapted to the common-gen corpus
HMM_MODEL_PATH = f"ctrlg/hmm_gpt2-large_common-gen_4096"  # alternatively 'ctrlg/hmm_gpt2-large_common-gen_32768' for better quality

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH).to(device)
base_model.eval()
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
hmm_model = ctrlg.HMM.from_pretrained(HMM_MODEL_PATH).to(device)


# --- Define Reasoning Task ---
problem_statement = "Question: John has 5 apples. He gives 2 apples to Mary. How many apples does John have left?\nAnswer:"
prefix = problem_statement  # Start generation right after the prompt
suffix = "<|endoftext|>"  # Must end with EOS token

prefix_ids = tokenizer.encode(prefix)
suffix_ids = tokenizer.encode(suffix)
# Prompt includes the prefix for the base model context
prompt_ids = tokenizer.encode(f"<|endoftext|> {prefix}")

# Adjust token limits for reasoning steps
min_new_tokens = 200  # Minimum tokens for reasoning + answer
max_new_tokens = 240  # Maximum tokens for reasoning + answer

vocab_size = hmm_model.vocab_size
eos_token_id = hmm_model.eos_token_id

# --- DFA Definition for Structured Reasoning Sequence ---
print("Defining DFA for structured reasoning sequence...")
kmp_builder = ctrlg.KMPBuilder(vocab_size)
eos_builder = ctrlg.EOSBuilder(vocab_size, eos_token_id)

# Define the reasoning steps/phrases
analyze_phrase = " Problem Analysis:"  # Leading space often helps tokenization
# hypothesize_phrase = " Hypothesis:"
plan_phrase = " Plan:"
execute_phrase = " Execution:"  # Added execution step
final_answer_phrase = " Final Answer:"

# Encode the phrases
analyze_ids = tokenizer.encode(analyze_phrase)
# hypothesize_ids = tokenizer.encode(hypothesize_phrase)
plan_ids = tokenizer.encode(plan_phrase)
execute_ids = tokenizer.encode(execute_phrase)
final_answer_ids = tokenizer.encode(final_answer_phrase)

# Build DFAs for each phrase
dfa_analyze = kmp_builder.build(analyze_ids)
# dfa_hypothesize = kmp_builder.build(hypothesize_ids)
dfa_plan = kmp_builder.build(plan_ids)
dfa_execute = kmp_builder.build(execute_ids)
dfa_final_answer = kmp_builder.build(final_answer_ids)

# Concatenate DFAs to enforce sequence
dfa_reasoning_sequence_graph = ctrlg.DFA_concatenate(
    [dfa_plan, dfa_final_answer]
    # [dfa_analyze, dfa_hypothesize, dfa_plan, dfa_execute, dfa_final_answer]
)
print("Reasoning Sequence DFA defined.")
print("State Count:", ctrlg.DFA_size(dfa_reasoning_sequence_graph)[0])
print("Edge Count:", ctrlg.DFA_size(dfa_reasoning_sequence_graph)[1])

# --- Combine Reasoning DFA with EOS Constraint ---
print("Combining Reasoning DFA with EOS constraint...")
dfa_eos_graph = eos_builder.build()
dfa_graph = ctrlg.DFA_prod(
    [dfa_reasoning_sequence_graph, dfa_eos_graph],
    mode="intersection",  # Enforces both constraints
)
print("Combined DFA defined.")
print("Final State Count:", ctrlg.DFA_size(dfa_graph)[0])
print("Final Edge Count:", ctrlg.DFA_size(dfa_graph)[1])

# Minimize the DFA for potentially better performance (optional but recommended)
print("Minimizing the final DFA graph...")
dfa_graph = ctrlg.DFA_minimize(dfa_graph)
print("Minimized DFA State Count:", ctrlg.DFA_size(dfa_graph)[0])
print("Minimized DFA Edge Count:", ctrlg.DFA_size(dfa_graph)[1])
dfa_model = ctrlg.DFAModel(dfa_graph, vocab_size).to(
    device
)  # compile for GPU inference


# initialze the constraints logits processor & pre-computes conditional probabilities
constraint_logits_processor = ctrlg.ConstraintLogitsProcessor(
    hmm_model,
    dfa_model,
    min_new_tokens,
    max_new_tokens,
    prompt_ids,
    prefix_ids=prefix_ids,
    suffix_ids=suffix_ids,
)


# set the hmm_batch_size depending on the resource available;
beam_size = 16
constraint_logits_processor.hmm_batch_size = beam_size

# generate with beam search
input_ids = torch.tensor([prompt_ids], device=device)
outputs = base_model.generate(
    input_ids=input_ids,
    do_sample=False,
    num_beams=beam_size,
    num_return_sequences=beam_size,
    min_new_tokens=min_new_tokens,
    max_new_tokens=max_new_tokens,
    logits_processor=LogitsProcessorList([constraint_logits_processor]),
    pad_token_id=tokenizer.eos_token_id,
)


# extract the generated ids;
generated_ids = ctrlg.extract_generated_ids(
    outputs.tolist(), prompt_ids, suffix_ids, eos_token_id
)

# rank the generated ids by the base_model probability (using default length penalty)
generated_ids = ctrlg.rank_generated_ids(
    base_model, generated_ids, prompt_ids, suffix_ids, length_penalty=0.2
)

# print top 5 outputs
print(f"\n--- Generated Outputs (Top 5) ---")
print(f"Prompt: {tokenizer.decode(prefix_ids, skip_special_tokens=True)}")
for idx, generated in enumerate(generated_ids[:1]):
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    suffix_text = tokenizer.decode(suffix_ids, skip_special_tokens=True)
    # Only print the generated part clearly marked
    print(f"\n--- Output {idx+1} ---")
    print(f"\033[1m{generated_text}\033[0m")
    # print(f"Suffix (expected): {suffix_text}") # Optional: print suffix for verification
