import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
VECTOR_DIR = REPO_ROOT / "artifacts" / "test_vectors" / "qwen3_0p6b_q4_v0"


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
            f"Missing {manifest_path}. Run 13_export_q4_gemv_vectors.py first."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def verify_checksums(manifest: dict[str, Any]) -> None:
    for entry in manifest["files"]:
        path = VECTOR_DIR / entry["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing vector file: {path}")
        actual = sha256_file(path)
        expected = entry["sha256"]
        if actual != expected:
            raise RuntimeError(
                f"Checksum mismatch for {path.name}: expected {expected}, got {actual}"
            )


def unpack_signed_int4(packed: np.ndarray, values_per_row: int) -> np.ndarray:
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    nibbles = np.empty((packed.shape[0], values_per_row), dtype=np.uint8)
    nibbles[:, 0::2] = low
    nibbles[:, 1::2] = high
    q = nibbles.astype(np.int16)
    q[q >= 8] -= 16
    return q.astype(np.int8)


def q4_gemv_from_artifact(
    x_q4_12: np.ndarray,
    packed_weight: np.ndarray,
    scale_q2_14: np.ndarray,
    group_size: int,
    act_frac: int,
    scale_frac: int,
) -> np.ndarray:
    hidden_size = int(x_q4_12.shape[0])
    q4_weight = unpack_signed_int4(packed_weight, hidden_size)
    out_features, in_features = q4_weight.shape
    groups_per_row = in_features // group_size
    x_grouped = x_q4_12.astype(np.int64).reshape(groups_per_row, group_size)
    w_grouped = q4_weight.astype(np.int64).reshape(out_features, groups_per_row, group_size)
    partial_sums = np.sum(w_grouped * x_grouped[None, :, :], axis=2, dtype=np.int64)
    scaled_sums = partial_sums * scale_q2_14.astype(np.int64)
    output_q = np.sum(scaled_sums, axis=1, dtype=np.int64)
    return (output_q.astype(np.float64) / float(1 << (act_frac + scale_frac))).astype(np.float32)


def compare_exact(name: str, actual: np.ndarray, expected: np.ndarray) -> bool:
    ok = bool(np.array_equal(actual, expected))
    diff = np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))
    print("-" * 80)
    print(name)
    print(f"  exact artifact match: {ok}")
    print(f"  max recompute diff: {float(diff):.10e}")
    return ok


def compare_fp32(name: str, actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    diff = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    metrics = {
        "max_abs_error": float(np.max(diff)),
        "mean_abs_error": float(np.mean(diff)),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
    }
    print("-" * 80)
    print(name)
    print(f"  max abs error: {metrics['max_abs_error']:.10e}")
    print(f"  mean abs error: {metrics['mean_abs_error']:.10e}")
    print(f"  rmse: {metrics['rmse']:.10e}")
    return metrics


def verify_dot64_smoke(manifest: dict[str, Any]) -> bool:
    data = np.load(VECTOR_DIR / "q_proj_row0_group0_dot64.npz")
    act_frac = int(manifest["activation_format"]["fraction_bits"])
    scale_frac = int(manifest["weight_format"]["scale_fraction_bits"])
    activation = data["activation_q4_12"].astype(np.int64)
    weight = data["weight_q4_unpacked"].astype(np.int64)
    scale = int(data["scale_q2_14"].item())
    partial = int(np.sum(activation * weight, dtype=np.int64))
    scaled = partial * scale
    expected_float = np.float32(scaled / float(1 << (act_frac + scale_frac)))

    checks = [
        partial == int(data["partial_sum_int64"].item()),
        scaled == int(data["scaled_sum_q26_int64"].item()),
        expected_float == data["expected_float32"].astype(np.float32),
    ]
    print("-" * 80)
    print("q_proj row 0 group 0 dot64 smoke")
    print(f"  partial_sum_int64: {partial}")
    print(f"  scaled_sum_q26_int64: {scaled}")
    print(f"  expected_float32: {float(expected_float):.10e}")
    print(f"  all fields match: {all(checks)}")
    return all(checks)


def main() -> None:
    manifest = load_manifest()
    verify_checksums(manifest)

    data = np.load(VECTOR_DIR / "qkv_layer0_last_token_q4.npz")
    group_size = int(manifest["weight_format"]["group_size"])
    act_frac = int(manifest["activation_format"]["fraction_bits"])
    scale_frac = int(manifest["weight_format"]["scale_fraction_bits"])
    x_q4_12 = data["input_norm_q4_12"]

    q_actual = q4_gemv_from_artifact(
        x_q4_12,
        data["q_weight_q4_packed"],
        data["q_scale_q2_14"],
        group_size,
        act_frac,
        scale_frac,
    )
    k_actual = q4_gemv_from_artifact(
        x_q4_12,
        data["k_weight_q4_packed"],
        data["k_scale_q2_14"],
        group_size,
        act_frac,
        scale_frac,
    )
    v_actual = q4_gemv_from_artifact(
        x_q4_12,
        data["v_weight_q4_packed"],
        data["v_scale_q2_14"],
        group_size,
        act_frac,
        scale_frac,
    )

    print("Q4 GEMV test vector verification")
    print("=" * 80)
    print(f"Vector directory: {VECTOR_DIR}")
    print(f"Format: {manifest['weight_format']['name']}")
    print(f"Activation: {manifest['activation_format']['name']}")

    exact_checks = [
        compare_exact("q_proj stored Q4 output", q_actual, data["actual_q_q4"]),
        compare_exact("k_proj stored Q4 output", k_actual, data["actual_k_q4"]),
        compare_exact("v_proj stored Q4 output", v_actual, data["actual_v_q4"]),
        verify_dot64_smoke(manifest),
    ]

    fp32_metrics = {
        "q_proj": compare_fp32("q_proj vs FP32 expected", q_actual, data["expected_q_fp32"]),
        "k_proj": compare_fp32("k_proj vs FP32 expected", k_actual, data["expected_k_fp32"]),
        "v_proj": compare_fp32("v_proj vs FP32 expected", v_actual, data["expected_v_fp32"]),
    }

    metric_checks = []
    for name, metrics in fp32_metrics.items():
        expected = manifest["metrics_against_fp32_expected"][name]
        metric_checks.append(abs(metrics["max_abs_error"] - expected["max_abs_error"]) < 1e-7)
        metric_checks.append(abs(metrics["mean_abs_error"] - expected["mean_abs_error"]) < 1e-7)
        metric_checks.append(abs(metrics["rmse"] - expected["rmse"]) < 1e-7)

    if not all(exact_checks) or not all(metric_checks):
        raise RuntimeError("Q4 GEMV vector verification failed.")

    print()
    print("PASS: Q4 packed weights, fixed-point scales, and Q/K/V GEMV outputs verified.")


if __name__ == "__main__":
    main()
