import json
from transformers import AutoConfig

MODEL_DIR = "./Qwen3-0.6B-Base"

config = AutoConfig.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

cfg = config.to_dict()

important_keys = [
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "rope_theta",
    "rms_norm_eps",
    "tie_word_embeddings",
    "hidden_act",
    "max_position_embeddings"
]

print("Important config:")
for k in important_keys:
    print(f"{k}: {cfg.get(k)}")

print("\nFull config:")
print(json.dumps(cfg, indent=2, ensure_ascii=False))