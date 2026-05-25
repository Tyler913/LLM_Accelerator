import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
VECTOR_DIR = REPO_ROOT / "artifacts" / "test_vectors" / "qwen3_0p6b_fp32_v0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> dict[str, Any]:
    manifest_path = VECTOR_DIR / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run 11_export_fpga_test_vectors.py first."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def verify_checksums(manifest: dict[str, Any]) -> None:
    for vector in manifest["vectors"]:
        path = VECTOR_DIR / vector["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing vector file: {path}")
        actual = sha256_file(path)
        expected = vector["sha256"]
        if actual != expected:
            raise RuntimeError(
                f"Checksum mismatch for {path.name}: expected {expected}, got {actual}"
            )


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: np.ndarray) -> np.ndarray:
    x_fp32 = x.astype(np.float32)
    weight_fp32 = weight.astype(np.float32)
    mean_square = np.mean(x_fp32 * x_fp32, dtype=np.float32)
    inv_rms = np.float32(1.0) / np.sqrt(mean_square + eps.astype(np.float32))
    return (x_fp32 * inv_rms * weight_fp32).astype(np.float32)


def gemv(weight: np.ndarray, x: np.ndarray) -> np.ndarray:
    return (weight.astype(np.float32) @ x.astype(np.float32)).astype(np.float32)


def compare(name: str, actual: np.ndarray, expected: np.ndarray, atol: float, rtol: float) -> bool:
    diff = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    ok = bool(np.allclose(actual, expected, atol=atol, rtol=rtol))
    print("-" * 80)
    print(name)
    print(f"  actual shape: {list(actual.shape)}")
    print(f"  expected shape: {list(expected.shape)}")
    print(f"  max abs error: {float(np.max(diff)):.10e}")
    print(f"  mean abs error: {float(np.mean(diff)):.10e}")
    print(f"  allclose: {ok}")
    return ok


def main() -> None:
    manifest = load_manifest()
    verify_checksums(manifest)

    rmsnorm_data = np.load(VECTOR_DIR / "rmsnorm_layer0_last_token.npz")
    qkv_data = np.load(VECTOR_DIR / "qkv_layer0_last_token.npz")

    rmsnorm_actual = rms_norm(
        rmsnorm_data["input_hidden"],
        rmsnorm_data["norm_weight"],
        rmsnorm_data["eps"],
    )
    q_actual = gemv(qkv_data["q_weight"], qkv_data["input_norm"])
    k_actual = gemv(qkv_data["k_weight"], qkv_data["input_norm"])
    v_actual = gemv(qkv_data["v_weight"], qkv_data["input_norm"])

    print("FPGA FP32 test vector verification")
    print("=" * 80)
    print(f"Vector directory: {VECTOR_DIR}")
    print(f"Prompt: {manifest['prompt']!r}")
    print(f"Token ids: {manifest['token_ids']}")
    print(f"Selected position: {manifest['selected_position']}")
    print(f"Selected token id: {manifest['selected_token_id']}")

    checks = [
        compare(
            "layer0.input_layernorm",
            rmsnorm_actual,
            rmsnorm_data["expected_output"],
            atol=1e-5,
            rtol=1e-5,
        ),
        compare("layer0.q_proj", q_actual, qkv_data["expected_q"], atol=2e-4, rtol=2e-4),
        compare("layer0.k_proj", k_actual, qkv_data["expected_k"], atol=2e-4, rtol=2e-4),
        compare("layer0.v_proj", v_actual, qkv_data["expected_v"], atol=2e-4, rtol=2e-4),
    ]

    if not all(checks):
        raise RuntimeError("FPGA test vector verification failed.")

    print()
    print("PASS: FPGA FP32 RMSNorm and Q/K/V GEMV test vectors verified.")


if __name__ == "__main__":
    main()
