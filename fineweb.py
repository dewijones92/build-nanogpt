import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import argparse
import sys
import glob
import random

# ------------------------------------------
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8)  # 100M tokens per shard, total of 100 shards
RANDOM_SEED = 42  # Set a fixed random seed for reproducibility
LINES_PER_DOCUMENT = 1000  # Number of lines to group into one document

# Create the cache and local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']  # end of text token

def tokenize(doc):
    # Tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = enc.encode_ordinary(doc["text"] if isinstance(doc, dict) else doc)
    tokens.append(eot)  # Add the <|endoftext|> token at the end of the document
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)




def process_data(source):
    if source == 1:
        # Download the dataset
        fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")
        data_iterator = fw
        process_and_write_shards(data_iterator)
    elif source == 2:
        # Load all files in train_data/
        files = glob.glob("train_data/*")
        all_lines = []
        for file in files:
            try:
                with open(file, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                    all_lines.extend(lines)
            except Exception as e:
                print(f"Warning: Error reading file {file}: {str(e)}")
                print("Skipping this file and continuing with the next one.")
                continue
        
        # Set random seed for reproducibility
        random.seed(RANDOM_SEED)
        
        # Randomly select validation set (10% of data)
        val_size = int(len(all_lines) * 0.1)
        val_indices = set(random.sample(range(len(all_lines)), val_size))
        
        # Split data into train and validation sets while maintaining original order
        train_lines = [line for i, line in enumerate(all_lines) if i not in val_indices]
        val_lines = [line for i, line in enumerate(all_lines) if i in val_indices]
        
        # Group lines into documents
        train_docs = ['\n'.join(train_lines[i:i+LINES_PER_DOCUMENT]) for i in range(0, len(train_lines), LINES_PER_DOCUMENT)]
        val_docs = ['\n'.join(val_lines[i:i+LINES_PER_DOCUMENT]) for i in range(0, len(val_lines), LINES_PER_DOCUMENT)]
        
        # Process train data
        process_and_write_shards(train_docs, split="train")
        
        # Process validation data
        process_and_write_shards(val_docs, split="val")
    else:
        raise ValueError("Invalid source specified")



def process_and_write_shards(data_iterator, split=None):
    nprocs = max(1, os.cpu_count()//2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # Preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenize, data_iterator, chunksize=16):
            # Is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                # Simply append tokens to current shard
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                # Update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # Write the current shard and start a new one
                current_split = split if split else ("val" if shard_index == 0 else "train")
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{current_split}_{shard_index:06d}")
                # Split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # Populate the next shard with the leftovers of the current doc
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder
        # Write any remaining tokens as the last shard
        if token_count != 0:
            current_split = split if split else ("val" if shard_index == 0 else "train")
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{current_split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])

def parse_args():
    parser = argparse.ArgumentParser(description="Process FineWeb-Edu dataset")
    parser.add_argument("--source", type=int, choices=[1, 2],
                        help="1: Use HuggingFace dataset, 2: Load files from train_data/")
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_data(args.source)