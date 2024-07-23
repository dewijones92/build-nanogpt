import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples
from line_profiler import profile
from time import sleep

# -----------------------------------------------------------------------------

@profile
def get_best_float_config():
    if torch.cuda.is_available():
        # Check for CUDA GPUs
        device = torch.device("cuda")
        if torch.cuda.get_device_capability(0)[0] >= 8:
            # Ampere or newer architecture supports efficient bfloat16
            return torch.bfloat16
        elif torch.cuda.is_bf16_supported():
            # Older GPUs might support bfloat16 but less efficiently
            return torch.bfloat16
        else:
            # Fall back to float16 for older GPUs
            return torch.float16
    elif hasattr(torch, 'xla') and torch.xla.is_available():
        # TPU support
        import torch_xla.core.xla_model as xm
        if xm.xla_device().type == 'TPU':
            return torch.bfloat16  # TPUs are optimized for bfloat16
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Apple Silicon (M1/M2) GPUs
        return torch.float16  # bfloat16 not fully supported on MPS as of now
    elif torch.backends.mkldnn.is_available():
        # Intel CPUs with MKL-DNN (now OneDNN) support
        return torch.bfloat16
    else:
        # Default to float32 for other cases
        return torch.float32

# Example usage
best_dtype = get_best_float_config()
print(f"The recommended dtype for your hardware is: {best_dtype}")


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        print(f"Configuring optimizer with weight_decay={weight_decay}, learning_rate={learning_rate}, device_type={device_type}")
        
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        print(f"Total number of named parameters: {len(param_dict)}")
        
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        print(f"Number of parameters requiring gradients: {len(param_dict)}")
        
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        print(f"Number of parameters to be decayed: {len(decay_params)}")
        
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        print(f"Number of parameters not to be decayed: {len(nodecay_params)}")
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        print(f"Created {len(optim_groups)} optimizer groups")
        
        num_decay_params = sum(p.numel() for p in decay_params)
        print(f"Total number of parameters to be decayed: {num_decay_params:,}")
        
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Total number of parameters not to be decayed: {num_nodecay_params:,}")
        
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        print(f"Fused AdamW available: {fused_available}")
        
        use_fused = fused_available and device_type == "cuda"
        print(f"Using fused AdamW: {use_fused}")
        
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        print(f"Created AdamW optimizer with learning rate {learning_rate}, betas=(0.9, 0.95), eps=1e-8, fused={use_fused}")
        
        print("Optimizer configuration complete")
        return optimizer

# -----------------------------------------------------------------------------
import tiktoken
import numpy as np

@profile
def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        while True:
            if self.current_position + B*T+1 > len(self.tokens):
                # We've reached the end of the current shard
                self.current_shard = (self.current_shard + 1) % len(self.shards)
                self.tokens = load_tokens(self.shards[self.current_shard])
                self.current_position = B * T * self.process_rank
                continue

            buf = self.tokens[self.current_position : self.current_position+B*T+1]
            if len(buf) < B*T+1:
                # Not enough data in this shard, move to next
                self.current_shard = (self.current_shard + 1) % len(self.shards)
                self.tokens = load_tokens(self.shards[self.current_shard])
                self.current_position = B * T * self.process_rank
                continue

            x = buf[:-1].view(B, T)
            y = buf[1:].view(B, T)
            self.current_position += B * T * self.num_processes
            return x, y


# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

@profile
def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")


import torch
import math
from line_profiler import profile

@profile
def optimize_training_params(model, min_micro_batch_size=1, min_seq_length=64, max_seq_length=2048):
    def get_gpu_memory():
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory, torch.cuda.memory_allocated(0)
        else:
            return 0, 0

    def estimate_model_memory():
        return sum(p.numel() * p.element_size() for p in model.parameters())

    def estimate_sample_memory(seq_length):
        # This is a rough estimate and may need to be adjusted based on your specific model architecture
        return seq_length * model.config.n_embd * 4 * 2  # *2 for forward and backward pass

    total_memory, allocated_memory = get_gpu_memory()
    free_memory = total_memory - allocated_memory
    model_memory = estimate_model_memory()
    
    # Reserve some memory for CUDA kernels and other overhead
    available_memory = free_memory - model_memory - 1e9  # Reserve 1GB for overhead
    
    if available_memory <= 0:
        raise ValueError("Not enough GPU memory available")

    best_micro_batch_size = min_micro_batch_size
    best_seq_length = min_seq_length
    best_grad_acc_steps = 1

    for seq_length in range(min_seq_length, max_seq_length + 1, 64):
        sample_memory = estimate_sample_memory(seq_length)
        max_batch_size = available_memory // sample_memory

        if max_batch_size < min_micro_batch_size:
            break

        # We'll use gradient accumulation steps to increase effective batch size
        grad_acc_steps = max(1, min(32, math.ceil(1024 / max_batch_size)))  # Limit to max 32 grad acc steps

        current_micro_batch_size = min(max_batch_size, 1024 // grad_acc_steps)

        if current_micro_batch_size > best_micro_batch_size or (current_micro_batch_size == best_micro_batch_size and seq_length > best_seq_length):
            best_micro_batch_size = current_micro_batch_size
            best_seq_length = seq_length
            best_grad_acc_steps = grad_acc_steps

    if best_micro_batch_size == min_micro_batch_size and best_seq_length == min_seq_length:
        raise ValueError("Could not find suitable parameters. Consider reducing min_micro_batch_size or min_seq_length.")

    actual_batch_size = best_micro_batch_size * best_seq_length * best_grad_acc_steps

    return {
        "micro_batch_size": best_micro_batch_size,
        "sequence_length": best_seq_length,
        "gradient_accumulation_steps": best_grad_acc_steps,
        "actual_batch_size": actual_batch_size
    }


model = GPT(GPTConfig(vocab_size=50304))
try:
    params = optimize_training_params(model)
    print(f"Optimized parameters: {params}")
except ValueError as e:
    print(f"Error: {e}")
    exit;

# After getting the optimized parameters
B = params["micro_batch_size"]
T = params["sequence_length"]
grad_accum_steps = params["gradient_accumulation_steps"]
actual_batch_size = params["actual_batch_size"]

print(f"Micro batch size: {B}")
print(f"Sequence length: {T}")
print(f"Gradient accumulation steps: {grad_accum_steps}")
print(f"Actual batch size: {actual_batch_size}")

assert actual_batch_size == B * T * grad_accum_steps * ddp_world_size, "Inconsistency in batch size calculation"

if master_process:
    print(f"Effective total batch size: {actual_batch_size}")
    print(f"=> gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

# ... rest of your training loop ...

torch.set_float32_matmul_precision('high')


# create model
# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
model.to(device)
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens

@profile
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

@profile
def optimize(): 
    # optimize!
    import logging

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)
    print(f"Optimizer configured with weight_decay=0.1, learning_rate=6e-4, device_type={device_type}")

    # create the log directory we will write checkpoints to and log to
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    print(f"Log directory created: {log_dir}")
    with open(log_file, "w") as f: # open for writing to clear the file
        pass
    print(f"Log file cleared: {log_file}")

    for step in range(max_steps):
        print(f"Starting step {step}/{max_steps}")
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # once in a while evaluate our validation loss
        if step % 250 == 0 or last_step:
            print("Evaluating validation loss")
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for i in range(val_loss_steps):
                    logger.debug(f"Validation step {i+1}/{val_loss_steps}")
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=best_dtype):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                logger.debug("Reducing validation loss across processes")
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"Validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                if step > 0 and (step % 5000 == 0 or last_step):
                    print("Saving model checkpoint")
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'step': step,
                        'val_loss': val_loss_accum.item()
                    }
                    torch.save(checkpoint, checkpoint_path)
                    print(f"Checkpoint saved to {checkpoint_path}")

        # once in a while evaluate hellaswag
        if (step % 250 == 0 or last_step) and (not use_compile):
            print("Evaluating HellaSwag")
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val")):
                if i % ddp_world_size != ddp_rank:
                    continue
                logger.debug(f"Processing HellaSwag example {i}")
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=best_dtype):
                        logits, loss = model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            if ddp:
                logger.debug("Reducing HellaSwag results across processes")
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
            acc_norm = num_correct_norm / num_total
            if master_process:
                print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} hella {acc_norm:.4f}\n")

        # once in a while generate from the model (except step 0, which is noise)
        if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
            print("Generating text from the model")
            model.eval()
            num_return_sequences = 4
            max_length = 32
            tokens = enc.encode("Hello, I'm a language model,")
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)
            while xgen.size(1) < max_length:
                logger.debug(f"Generating token {xgen.size(1)}/{max_length}")
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=best_dtype):
                        logits, loss = model(xgen)
                    logits = logits[:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                    xcol = torch.gather(topk_indices, -1, ix)
                    xgen = torch.cat((xgen, xcol), dim=1)
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                print(f"rank {ddp_rank} sample {i}: {decoded}")

        # do one step of the optimization
        print("Starting optimization step")
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            logger.debug(f"Micro-step {micro_step+1}/{grad_accum_steps}")
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            with torch.autocast(device_type=device_type, dtype=best_dtype):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
            logger.debug(f"Micro-step loss: {loss.item():.6f}")
        if ddp:
            logger.debug("Reducing loss across processes")
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        if master_process:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")
    if ddp:
        destroy_process_group()


optimize()