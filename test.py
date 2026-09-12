# generate a sample from a saved checkpoint. run this from the same
# directory as mdl.py (so lilstoryteller-checkpoint/ is right there), e.g:
#   python3 generate_from_checkpoint.py

import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM, GPT2TokenizerFast

checkpoint_dir = "lilstoryteller-checkpoint"

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print("using:", device)

use_autocast = device in ("mps", "cuda")
autocast_dtype = torch.bfloat16
block_size = 384

print(f"loading checkpoint from {checkpoint_dir} ...")
tokenizer = GPT2TokenizerFast.from_pretrained(checkpoint_dir)
model = LlamaForCausalLM.from_pretrained(checkpoint_dir).to(device)
model.eval()
print("loaded.")


def encode(s):
    return tokenizer.encode(s)


def decode(ids):
    return tokenizer.decode(ids, skip_special_tokens=True)


@torch.no_grad()
def generate(prompt, length=500, temperature=0.8):
    tokens = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

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
        next_input = next_token

        if next_token.item() == tokenizer.eos_token_id:
            break

    return decode(tokens[0].tolist())


print("\ngenerating from step-2000 checkpoint...\n")
print(generate("Once upon a time ", 500, temperature=0.8))
