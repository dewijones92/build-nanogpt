import os
import multiprocessing as mp
import numpy as np
import tiktoken
from tqdm import tqdm

# ------------------------------------------
local_dir = "edu_fineweb10B"
shard_size = int(1e8)  # 100M tokens per shard
train_data_folder = "train_data"  # folder containing the txt files

# create the cache local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]  # end of text token

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split the content into paragraphs
    paragraphs = content.split('\n\n')
    
    # Group paragraphs into sections of 4
    sections = [paragraphs[i:i+4] for i in range(0, len(paragraphs), 4)]
    
    tokenized_sections = []
    for section in sections:
        # Join the 4 paragraphs back into a single string
        section_text = '\n\n'.join(section)
        # Tokenize the section
        tokens = [eot]  # Start with the EOT token
        tokens.extend(enc.encode(section_text))
        tokenized_sections.append(np.array(tokens, dtype=np.uint16))
    
    return tokenized_sections

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# Get all txt files from the train_data folder
txt_files = [os.path.join(train_data_folder, f) for f in os.listdir(train_data_folder) if f.endswith('.txt')]

# Process all files and tokenize sections
nprocs = max(1, os.cpu_count()//2)
with mp.Pool(nprocs) as pool:
    all_sections = []
    for sections in tqdm(pool.imap(process_file, txt_files), total=len(txt_files), desc="Processing files"):
        all_sections.extend(sections)

# Shuffle the sections
np.random.shuffle(all_sections)

# Split into train and val (90:10)
split_idx = int(len(all_sections) * 0.9)
train_sections = all_sections[:split_idx]
val_sections = all_sections[split_idx:]

# Function to write shards
def write_shards(sections, prefix):
    shard_index = 0
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    
    for tokens in sections:
        if token_count + len(tokens) < shard_size:
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"{prefix} Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{prefix}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder
    
    if token_count != 0:
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{prefix}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])

# Write train and val shards
write_shards(train_sections, "train")
write_shards(val_sections, "val")

print("Processing complete. Shards saved in", DATA_CACHE_DIR)