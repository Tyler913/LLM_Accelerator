from pathlib import Path
from safetensors import safe_open
import math

MODEL_DIR = Path("./Qwen3-0.6B-Base")

safetensor_files = sorted(MODEL_DIR.glob("*.safetensors"))

if not safetensor_files:
    raise FileNotFoundError("No .safetensors files found.")

total_params = 0

print("Safetensors files:")
for f in safetensor_files:
    print(" ", f)

print("\nTensor list:")

for file in safetensor_files:
    with safe_open(file, framework="pt", device="cpu") as f:
        for name in f.keys():
            tensor = f.get_tensor(name)
            shape = list(tensor.shape)
            numel = tensor.numel()
            dtype = tensor.dtype

            total_params += numel

            print(f"{name:80s} shape={shape} dtype={dtype} params={numel}")

print("\nTotal params:", total_params)
print("Total params in B:", total_params / 1e9)

q8_bytes = total_params * 1
q4_bytes = math.ceil(total_params / 2)

group_size = 64
scale_bytes_fp16 = math.ceil(total_params / group_size) * 2

print("\nRough memory estimate:")
print(f"FP16/BF16 weights: {total_params * 2 / 1024 / 1024:.2f} MB")
print(f"INT8 weights:      {q8_bytes / 1024 / 1024:.2f} MB")
print(f"INT4 weights:      {q4_bytes / 1024 / 1024:.2f} MB")
print(f"Q4 scales fp16:    {scale_bytes_fp16 / 1024 / 1024:.2f} MB, group_size={group_size}")
print(f"Q4 total rough:    {(q4_bytes + scale_bytes_fp16) / 1024 / 1024:.2f} MB")