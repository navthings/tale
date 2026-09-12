# train a small llama-arch model ("lilstoryteller") on tinystories, streamed
# and tokenized on the fly. optimized for macOS / apple silicon (mps).

import os

# must be set before transformers/tokenizers is imported, or the fast
# tokenizer's rust thread pool may be disabled by default
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

# bf16 autocast speeds up fwd/bwd on mps/cuda at negligible quality cost
use_autocast = device in ("mps", "cuda")
autocast_dtype = torch.bfloat16

# settings
steps = int(input("steps: "))
resume_from = input("resume from checkpoint dir (blank for fresh model): ").strip()
block_size = 384
batch_size = 16  # 32 OOM'd on MPS backward(); try 8 if this still OOMs

# lr schedule: short warmup then cosine decay down to min_lr. dropping
# peak lr from the earlier flat 0.001 run — the val loss plateau there
# is the classic sign that lr is too high to make progress past the
# easy early gains.
peak_lr = 3e-4
min_lr = 3e-5
warmup_steps = 200

# tokens buffered before training starts, and held-out tokens for validation
min_buffer_tokens = 2_000_000
val_buffer_tokens = 200_000
max_buffer_tokens = min_buffer_tokens * 4
refill_every = 50  # top up train buffer every N steps, not every step

# tokenizer (GPT-2 BPE)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size


def encode(s):
    return tokenizer.encode(s)


def decode(ids):
    return tokenizer.decode(ids, skip_special_tokens=True)


# dataset (100% of train split, streamed)
dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

# background tokenizer thread — batches examples before encoding so the
# rust tokenizer backend can actually parallelize across cores (a single
# tokenizer.encode(s) call never gets that benefit)
TOKENIZE_BATCH_SIZE = 256
token_queue = queue.Queue(maxsize=1024)
stop_signal = object()
# eos separates stories so the model learns "story over -> emit eos",
# which is what lets generate() stop on its own instead of running to
# max_new_tokens every time
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

# training buffer: preallocated fixed-capacity tensor + fill pointer, so
# refills are in-place writes instead of a torch.cat that copies the
# whole buffer every call
train_capacity = max_buffer_tokens
train_buf = torch.empty(train_capacity, dtype=torch.long)
train_len = 0
val_buf = None
stream_exhausted = False


def _pull_available(max_tokens):
    # non-blocking: grabs whatever's already queued, up to max_tokens
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
    # blocks (with polling) until target_len tokens arrive or stream ends
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
    # in-place append; slides the live window left if capacity is hit
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
    # startup only: block until the buffer has at least min_tokens
    needed = min_tokens - train_len
    if needed > 0:
        _write_into_train_buf(_pull_blocking(needed))


def refill_train_buffer_nonblocking():
    # called periodically from the training loop, never from get_batch()
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
    # pure read: no refill here, so this is just indices + a gather
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


# model (~50M params, tied embeddings, GQA with 6 query / 2 kv heads)
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

# optimizer — foreach=True forces the batched update path on MPS
# (fused=True isn't supported there)
optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, foreach=True)


def lr_at(step):
    # linear warmup, then cosine decay from peak_lr to min_lr
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

    # train loss every step
    print(f"step {step} | train loss {loss.item():.3f} | lr {lr:.2e} | buffer {train_len} tokens")

    # val loss every 100 steps; save separately whenever it's a new best
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


# generation — uses KV cache instead of re-running the full context
# through every layer on each step
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

# output
print("\nmodel output\n")
print(generate("Once upon a time ", 500, temperature=0.8))
