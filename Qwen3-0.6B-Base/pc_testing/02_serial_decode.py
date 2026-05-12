import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "./Qwen3-0.6B-Base"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=dtype,
    local_files_only=True
).to(device)

model.eval()

prompt = "The future of FPGA is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

max_new_tokens = 32

past_key_values = None

generated_ids = []

with torch.no_grad():
    # Serial prefill: feed prompt tokens one at a time.
    for i in range(input_ids.shape[1]):
        token = input_ids[:, i:i+1]

        outputs = model(
            input_ids=token,
            past_key_values=past_key_values,
            use_cache=True
        )

        past_key_values = outputs.past_key_values

    # After the prompt, select the next token from the final prompt logits.
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    generated_ids.append(next_token.item())

    # Decode: feed one token at a time after prefill.
    for _ in range(max_new_tokens - 1):
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True
        )

        past_key_values = outputs.past_key_values

        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_ids.append(next_token.item())

full_ids = torch.cat(
    [
        input_ids[0].cpu(),
        torch.tensor(generated_ids, dtype=torch.long)
    ],
    dim=0
)

print(tokenizer.decode(full_ids, skip_special_tokens=True))
