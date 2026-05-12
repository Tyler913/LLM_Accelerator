import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "./Qwen3-0.6B-Base"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print("device =", device)

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
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None
    )

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))