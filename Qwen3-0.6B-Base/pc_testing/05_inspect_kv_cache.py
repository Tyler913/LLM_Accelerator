from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

MODEL_DIR = Path(__file__).resolve().parents[1]

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

config = AutoConfig.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    dtype=dtype,
    local_files_only=True
).to(device)

model.eval()

prompt = "The future of FPGA is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

print("Prompt:", prompt)
print("Input token ids:", input_ids[0].tolist())
print("Number of prompt tokens:", input_ids.shape[1])
print()

print("Model config:")
print("num_hidden_layers     =", config.num_hidden_layers)
print("num_attention_heads   =", config.num_attention_heads)
print("num_key_value_heads   =", config.num_key_value_heads)
print("head_dim              =", config.head_dim)
print("hidden_size           =", config.hidden_size)
print("vocab_size            =", config.vocab_size)
print()


def extract_kv_pairs(past_key_values):
    """
    Support multiple Transformers cache layouts:
    1. Older versions return past_key_values as a tuple/list.
    2. Newer versions may return DynamicCache with key_cache/value_cache.
    """
    if isinstance(past_key_values, (tuple, list)):
        return [(layer_kv[0], layer_kv[1]) for layer_kv in past_key_values]

    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return list(zip(past_key_values.key_cache, past_key_values.value_cache))

    if hasattr(past_key_values, "layers"):
        pairs = []
        for layer in past_key_values.layers:
            k = None
            v = None

            for k_name in ["keys", "key_cache", "k_cache"]:
                if hasattr(layer, k_name):
                    k = getattr(layer, k_name)
                    break

            for v_name in ["values", "value_cache", "v_cache"]:
                if hasattr(layer, v_name):
                    v = getattr(layer, v_name)
                    break

            if k is not None and v is not None:
                pairs.append((k, v))

        if pairs:
            return pairs

    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)}")


past_key_values = None

with torch.no_grad():
    for pos in range(input_ids.shape[1]):
        token = input_ids[:, pos:pos + 1]

        outputs = model(
            input_ids=token,
            past_key_values=past_key_values,
            use_cache=True
        )

        past_key_values = outputs.past_key_values
        kv_pairs = extract_kv_pairs(past_key_values)

        token_id = token.item()
        token_text = tokenizer.decode([token_id])

        print("=" * 80)
        print(f"After processing prompt token position {pos}")
        print("token id   =", token_id)
        print("token text =", repr(token_text))
        print("logits shape =", list(outputs.logits.shape))
        print("number of KV layers =", len(kv_pairs))

        # Print only a few layers to keep the output manageable.
        for layer_id in [0, 1, 27]:
            k, v = kv_pairs[layer_id]
            print(f"Layer {layer_id}:")
            print("  K shape =", list(k.shape), "dtype =", k.dtype)
            print("  V shape =", list(v.shape), "dtype =", v.dtype)

    print("=" * 80)
    print("Generate first new token from last prompt token logits")

    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    print("next token id =", next_token.item())
    print("next token text =", repr(tokenizer.decode([next_token.item()])))

    outputs = model(
        input_ids=next_token,
        past_key_values=past_key_values,
        use_cache=True
    )

    past_key_values = outputs.past_key_values
    kv_pairs = extract_kv_pairs(past_key_values)

    print()
    print("After feeding first generated token back into model:")
    print("logits shape =", list(outputs.logits.shape))
    print("number of KV layers =", len(kv_pairs))

    for layer_id in [0, 1, 27]:
        k, v = kv_pairs[layer_id]
        print(f"Layer {layer_id}:")
        print("  K shape =", list(k.shape), "dtype =", k.dtype)
        print("  V shape =", list(v.shape), "dtype =", v.dtype)


print()
print("=" * 80)
print("Expected KV cache memory estimate:")

num_layers = config.num_hidden_layers
num_kv_heads = config.num_key_value_heads
head_dim = config.head_dim

kv_values_per_token_per_layer = 2 * num_kv_heads * head_dim
kv_values_per_token_all_layers = num_layers * kv_values_per_token_per_layer

print("KV values per token per layer =", kv_values_per_token_per_layer)
print("KV values per token all layers =", kv_values_per_token_all_layers)

for context_len in [128, 256, 512, 1024]:
    fp16_bytes = context_len * kv_values_per_token_all_layers * 2
    int8_bytes = context_len * kv_values_per_token_all_layers * 1

    print(f"context_len={context_len}:")
    print(f"  fp16/bf16 KV cache ≈ {fp16_bytes / 1024 / 1024:.2f} MB")
    print(f"  int8 KV cache      ≈ {int8_bytes / 1024 / 1024:.2f} MB")
