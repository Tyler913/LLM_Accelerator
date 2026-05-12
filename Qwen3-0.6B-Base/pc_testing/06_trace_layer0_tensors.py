from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = Path(__file__).resolve().parents[1]

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    dtype=dtype,
    local_files_only=True
).to(device)

model.eval()

# Qwen3ForCausalLM usually stores the actual backbone under model.model.
backbone = model.model if hasattr(model, "model") else model

records = []


def tensor_info(x):
    if torch.is_tensor(x):
        return {
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "device": str(x.device),
        }

    if isinstance(x, (tuple, list)):
        return [tensor_info(v) for v in x]

    if isinstance(x, dict):
        return {k: tensor_info(v) for k, v in x.items()}

    return str(type(x))


def make_hook(name):
    def hook(module, inputs, output):
        records.append({
            "name": name,
            "input": tensor_info(inputs),
            "output": tensor_info(output),
        })
    return hook


handles = []

# Trace only Layer 0 to keep the output manageable.
layer0 = backbone.layers[0]
attn0 = layer0.self_attn
mlp0 = layer0.mlp

modules_to_hook = []

modules_to_hook.append(("embed_tokens", backbone.embed_tokens))

modules_to_hook.append(("layer0.input_layernorm", layer0.input_layernorm))

modules_to_hook.append(("layer0.self_attn.q_proj", attn0.q_proj))
modules_to_hook.append(("layer0.self_attn.k_proj", attn0.k_proj))
modules_to_hook.append(("layer0.self_attn.v_proj", attn0.v_proj))

if hasattr(attn0, "q_norm"):
    modules_to_hook.append(("layer0.self_attn.q_norm", attn0.q_norm))

if hasattr(attn0, "k_norm"):
    modules_to_hook.append(("layer0.self_attn.k_norm", attn0.k_norm))

modules_to_hook.append(("layer0.self_attn.o_proj", attn0.o_proj))

modules_to_hook.append(("layer0.post_attention_layernorm", layer0.post_attention_layernorm))

modules_to_hook.append(("layer0.mlp.gate_proj", mlp0.gate_proj))
modules_to_hook.append(("layer0.mlp.up_proj", mlp0.up_proj))
modules_to_hook.append(("layer0.mlp.down_proj", mlp0.down_proj))

if hasattr(backbone, "norm"):
    modules_to_hook.append(("final_norm", backbone.norm))

if hasattr(model, "lm_head"):
    modules_to_hook.append(("lm_head", model.lm_head))

for name, module in modules_to_hook:
    handles.append(module.register_forward_hook(make_hook(name)))


prompt = "The future of FPGA is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

print("Prompt:", prompt)
print("Input ids:", input_ids[0].tolist())
print()

past_key_values = None

with torch.no_grad():
    # Run the first token to observe the no-cache case.
    token0 = input_ids[:, 0:1]

    print("=" * 80)
    print("Run token position 0")
    print("token id:", token0.item())
    print("token text:", repr(tokenizer.decode([token0.item()])))

    records.clear()

    outputs = model(
        input_ids=token0,
        past_key_values=past_key_values,
        use_cache=True
    )

    past_key_values = outputs.past_key_values

    print("logits shape:", list(outputs.logits.shape))
    print()

    for r in records:
        print("-" * 80)
        print(r["name"])
        print("  input :", r["input"])
        print("  output:", r["output"])

    # Run the second token to observe the cached case.
    token1 = input_ids[:, 1:2]

    print()
    print("=" * 80)
    print("Run token position 1 with past_key_values")
    print("token id:", token1.item())
    print("token text:", repr(tokenizer.decode([token1.item()])))

    records.clear()

    outputs = model(
        input_ids=token1,
        past_key_values=past_key_values,
        use_cache=True
    )

    past_key_values = outputs.past_key_values

    print("logits shape:", list(outputs.logits.shape))
    print()

    for r in records:
        print("-" * 80)
        print(r["name"])
        print("  input :", r["input"])
        print("  output:", r["output"])


for h in handles:
    h.remove()
