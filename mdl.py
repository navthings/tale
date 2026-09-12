# im[orts
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import threading
import queue

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import LlamaConfig, LlamaForCausalLM, GPT2TokenizerFast


# device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print("using:", device)


use_autocast = device in ("mps", "cuda")
autocast_dtype = torch.bfloat16

# settings
steps = int(input("steps: "))
resume_from = input("resume from checkpoint dir (blank for fresh model): ").strip()
block_size = 384
batch_size = 16  


peak_lr = 3e-4
min_lr = 3e-5
warmup_steps = 200


min_buffer_tokens = 2_000_000
val_buffer_tokens = 200_000
max_buffer_tokens = min_buffer_tokens * 4
refill_every = 50  

# tokenizer I CANT BE BOTHERED TO TRAIN ONE FROM SCRATCH so using gpt 2 idk gonna have to add eos token smw
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size


def encode(s):
    return tokenizer.encode(s)


def decode(ids):
    return tokenizer.decode(ids, skip_special_tokens=True)


# dataset
dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
TOKENIZE_BATCH_SIZE = 256
token_queue = queue.Queue(maxsize=1024)
stop_signal = object()
# eos
eos_ids = [tokenizer.eos_token_id]


def tokenizer_worker():
    texts = []

    def flush():
        if not texts:
            return
        encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
        for ids in encoded:
            token_queue.put(ids + eos_ids)
        texts.clear()

    for example in dataset:
        texts.append(example["text"])
        if len(texts) >= TOKENIZE_BATCH_SIZE:
            flush()
    flush()
    token_queue.put(stop_signal)


worker = threading.Thread(target=tokenizer_worker, daemon=True)
worker.start()

# training buffer
train_capacity = max_buffer_tokens
train_buf = torch.empty(train_capacity, dtype=torch.long)
train_len = 0
val_buf = None
stream_exhausted = False


def _pull_available(max_tokens):
    global stream_exhausted
    pooled = []
    pooled_len = 0
    while pooled_len < max_tokens:
        try:
            ids = token_queue.get_nowait()
        except queue.Empty:
            break
        if ids is stop_signal:
            stream_exhausted = True
            break
        pooled.extend(ids)
        pooled_len += len(ids)
    if not pooled:
        return torch.empty(0, dtype=torch.long)
    return torch.tensor(pooled, dtype=torch.long)


def _pull_blocking(target_len):
    global stream_exhausted
    pooled = []
    pooled_len = 0
    while pooled_len < target_len:
        try:
            ids = token_queue.get(timeout=0.01)
        except queue.Empty:
            if stream_exhausted:
                break
            continue
        if ids is stop_signal:
            stream_exhausted = True
            break
        pooled.extend(ids)
        pooled_len += len(ids)
    if not pooled:
        return torch.empty(0, dtype=torch.long)
    return torch.tensor(pooled, dtype=torch.long)


def _write_into_train_buf(chunk):
    global train_len
    n = chunk.numel()
    if n == 0:
        return
    if n > train_capacity:
        chunk = chunk[-train_capacity:]
        n = chunk.numel()
        train_len = 0
    if train_len + n > train_capacity:
        overflow = train_len + n - train_capacity
        train_buf[: train_len - overflow] = train_buf[overflow:train_len].clone()
        train_len -= overflow
    train_buf[train_len: train_len + n] = chunk
    train_len += n


def fill_val_buffer():
    global val_buf
    val_buf = _pull_blocking(val_buffer_tokens)
    if val_buf.numel() < val_buffer_tokens:
        print("warning: stream exhausted while filling validation buffer")
    print("validation buffer ready:", val_buf.numel(), "tokens")


def fill_train_buffer_blocking(min_tokens):
    needed = min_tokens - train_len
    if needed > 0:
        _write_into_train_buf(_pull_blocking(needed))


def refill_train_buffer_nonblocking():
    _write_into_train_buf(_pull_available(train_capacity - train_len))


print("filling validation buffer...")
fill_val_buffer()

print("filling initial training buffer...")
fill_train_buffer_blocking(min_buffer_tokens)
print("training buffer ready:", train_len, "tokens")
print("vocab:", vocab_size)

# training batches
_offsets = torch.arange(block_size)


def get_batch():
    starts = torch.randint(0, train_len - block_size - 1, (batch_size,))
    idx = starts[:, None] + _offsets
    x = train_buf[idx]
    y = train_buf[idx + 1]
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


# validation batches
def get_val_batch():
    starts = torch.randint(0, val_buf.numel() - block_size - 1, (batch_size,))
    idx = starts[:, None] + _offsets
    x = val_buf[idx]
    y = val_buf[idx + 1]
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


# model (should be 50m parameters idk)
if resume_from:
    model = LlamaForCausalLM.from_pretrained(resume_from).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(resume_from)
    print("resumed model from:", resume_from)
else:
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=384,
        intermediate_size=1536,
        num_hidden_layers=13,
        num_attention_heads=6,
        num_key_value_heads=2,
        max_position_embeddings=block_size,
        rms_norm_eps=1e-5,
        rope_theta=10000,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(config).to(device)
print("parameters:", sum(p.numel() for p in model.parameters()))

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, foreach=True)


def lr_at(step):
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    cosine = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.141592653589793)))
    return min_lr + (peak_lr - min_lr) * cosine.item()


# validation loss
@torch.no_grad()
def validation_loss():
    model.eval()
    x, y = get_val_batch()
    with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=use_autocast):
        logits = model(x).logits
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
    model.train()
    return loss.item()


# train
model.train()
best_val_loss = float("inf")
for step in range(steps):
    lr = lr_at(step)
    for group in optimizer.param_groups:
        group["lr"] = lr

    if step % refill_every == 0:
        refill_train_buffer_nonblocking()

    x, y = get_batch()

    with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=use_autocast):
        logits = model(x).logits
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    #train loss only works half the time i HAVE NO FUCKING IDEA WHY
    print(f"step {step} | train loss {loss.item():.3f} | lr {lr:.2e} | buffer {train_len} tokens")

    # val loss
    if step % 100 == 0:
        val_loss = validation_loss()
        print(f"step {step} | val loss {val_loss:.3f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained("lilstoryteller-best")
            tokenizer.save_pretrained("lilstoryteller-best")
            print(f"new best val loss {val_loss:.3f}, saved to lilstoryteller-best")

    if step % 2000 == 0 and step > 0:
        model.save_pretrained("lilstoryteller-checkpoint")
        tokenizer.save_pretrained("lilstoryteller-checkpoint")
        print(f"checkpoint saved at step {step}")


# generation
@torch.no_grad()
def generate(prompt, length=500, temperature=0.8):
    model.eval()
    tokens = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

    # cap at block_size since positions beyond it are out of distribution
    prompt_len = tokens.shape[1]
    max_new_tokens = min(length, max(block_size - prompt_len, 0))
    if max_new_tokens < length:
        print(
            f"note: prompt uses {prompt_len} tokens, so generation is capped "
            f"at {max_new_tokens} new tokens (block_size={block_size})"
        )

    past_key_values = None
    next_input = tokens

    for _ in range(max_new_tokens):
        with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=use_autocast):
            out = model(next_input, past_key_values=past_key_values, use_cache=True)
        logits = out.logits
        past_key_values = out.past_key_values

        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)

        tokens = torch.cat([tokens, next_token], dim=1)
        next_input = next_token  # only the newest token needs feeding in now

        if next_token.item() == tokenizer.eos_token_id:
            break  # model chose to end the story

    return decode(tokens[0].tolist())


# save
model.save_pretrained("lilstoryteller")
tokenizer.save_pretrained("lilstoryteller")
print("\nmodel saved")

# test IF THIS DOESNT WORK AFTER I TRIAN FOR 20 HOURS IM GONNA EXPLODE
print("\nmodel output\n")
print(generate("Once upon a time ", 500, temperature=0.8))
