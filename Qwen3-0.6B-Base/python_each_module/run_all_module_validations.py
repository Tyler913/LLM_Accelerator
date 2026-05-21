import subprocess
import sys
from pathlib import Path


SCRIPTS = [
    "01_validate_embedding.py",
    "02_validate_rmsnorm.py",
    "03_validate_qkv_gemv.py",
    "04_validate_qk_norm_rope.py",
    "05_validate_kv_cache.py",
    "06_validate_attention.py",
    "07_validate_mlp.py",
    "08_validate_decoder_layer.py",
    "09_validate_final_norm_lm_head_argmax.py",
    "10_validate_full_run_one_token.py",
]


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    for script_name in SCRIPTS:
        script_path = script_dir / script_name
        print()
        print("=" * 80, flush=True)
        print(f"Running {script_name}", flush=True)
        print("=" * 80, flush=True)
        subprocess.run([sys.executable, str(script_path)], check=True)

    print()
    print("PASS: all module validations completed.")


if __name__ == "__main__":
    main()
