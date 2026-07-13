from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


VEC1024 = 1024
VEC2048 = 2048
VEC3072 = 3072
CACHE_BASE = 0x0000_0004_1410_0000
CACHE_BYTES = 2 * 8 * 256 * 128 * 4
TAIL_BASE = 0x0000_0004_0501_0000
TAIL_BYTES = 0x4000

EXPECTED_READ_REQS = 46289
EXPECTED_READ_WORDS = 2140762
EXPECTED_WRITE_REQS = 6156
EXPECTED_WRITE_WORDS = 26624
EXPECTED_WRITE_KINDS = {
    1: 1,
    2: 1,
    3: VEC2048,
    4: VEC1024,
    5: VEC1024,
    6: 2 * VEC1024,
    7: 1,
    8: 1,
    9: 1,
    10: 1,
    11: 1,
    12: 1,
    13: 1,
    14: 1,
    15: 1,
    16: 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit tied-Q4 embedding through complete Layer 0 timing."
    )
    parser.add_argument("trace", type=Path)
    parser.add_argument("--full-chain-manifest", type=Path, required=True)
    parser.add_argument("--xsim-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def in_range(address: int, base: int, byte_count: int) -> bool:
    return base <= address < base + byte_count


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def descriptor_base(manifest: dict[str, Any], slot: int) -> int:
    return int(manifest["descriptors"][slot]["base_addr"])


def read_hex(path: Path) -> list[int]:
    return [int(line.strip(), 16) for line in path.read_text(encoding="ascii").splitlines()]


def main() -> None:
    args = parse_args()
    root = args.full_chain_manifest.resolve().parent
    full_chain = load_json(args.full_chain_manifest)
    qmap_dir = root / "layer0" / "qmap"
    sim_dir = root / "layer0" / "sim_vectors"

    qkv = load_json(qmap_dir / "layer0_qkv_from_embedding_rmsnorm_full_manifest.json")
    input_norm = load_json(qmap_dir / "layer0_chained_input_rmsnorm_runtime_manifest.json")
    frontend = load_json(qmap_dir / "layer0_chained_attention_frontend_runtime_manifest.json")
    score = load_json(qmap_dir / "layer0_chained_attention_score_value_runtime_manifest.json")
    oproj = load_json(qmap_dir / "layer0_chained_o_proj_runtime_manifest.json")
    post = load_json(qmap_dir / "layer0_chained_post_attention_residual_norm_runtime_manifest.json")
    gate = load_json(qmap_dir / "layer0_chained_mlp_gate_up_runtime_manifest.json")
    silu = load_json(qmap_dir / "layer0_chained_mlp_silu_mul_runtime_manifest.json")
    down = load_json(qmap_dir / "layer0_chained_mlp_down_runtime_manifest.json")
    residual = load_json(qmap_dir / "layer0_chained_mlp_residual_add_runtime_manifest.json")

    addresses = full_chain["addresses"]
    token_id = int(full_chain["token_id"])
    embed_weight_row = int(addresses["embedding_weight_base"]) + token_id * 512
    embed_scale_row = int(addresses["embedding_scale_base"]) + token_id * 32
    hidden = int(addresses["input_hidden_base"])
    norm = int(input_norm["memory_layout"]["input_norm_addr"])
    q_base = descriptor_base(qkv, 8)
    k_base = descriptor_base(qkv, 9)
    v_base = descriptor_base(qkv, 10)
    q_rope = int(frontend["memory_layout"]["q_rope_output_addr"])
    attn = int(score["memory_layout"]["attn_out_addr"])
    oproj_out = int(oproj["memory_layout"]["output_addr"])
    post_hidden = int(post["memory_layout"]["post_attention_hidden_addr"])
    post_norm = int(post["memory_layout"]["post_norm_addr"])
    gate_out = int(gate["memory_layout"]["gate_output_addr"])
    up_out = int(gate["memory_layout"]["up_output_addr"])
    silu_hidden = int(silu["memory_layout"]["hidden_addr"])
    down_out = int(down["memory_layout"]["output_addr"])
    layer_out = int(residual["memory_layout"]["layer_out_addr"])

    require(hidden == layer_out, "expected Layer 0 output to reuse the input hidden buffer")
    cache_expected_addr = read_hex(
        sim_dir / "layer0_chained_kv_cache_append_real_expected_addr.hex"
    )
    require(len(cache_expected_addr) == 2 * VEC1024, "cache expected-address count mismatch")

    ranges = {
        "hidden": (hidden, VEC1024 * 4, 8, 2 * VEC1024),
        "norm": (norm, VEC1024 * 4, 4, VEC1024),
        "q": (q_base, VEC2048 * 4, 8, VEC2048),
        "k": (k_base, VEC1024 * 4, 4, VEC1024),
        "v": (v_base, VEC1024 * 4, 4, VEC1024),
        "q_rope": (q_rope, VEC2048 * 4, 8, VEC2048),
        "cache": (CACHE_BASE, CACHE_BYTES, 2 * 16 * 5 * 128, 2 * 16 * 5 * 128),
        "attn": (attn, VEC2048 * 4, 8, VEC2048),
        "oproj": (oproj_out, VEC1024 * 4, 4, VEC1024),
        "post_hidden": (post_hidden, VEC1024 * 4, 4, VEC1024),
        "post_norm": (post_norm, VEC1024 * 4, 4, VEC1024),
        "gate": (gate_out, VEC3072 * 4, 12, VEC3072),
        "up": (up_out, VEC3072 * 4, 12, VEC3072),
        "silu": (silu_hidden, VEC3072 * 4, 12, VEC3072),
        "down": (down_out, VEC1024 * 4, 4, VEC1024),
        "tail_qmap": (TAIL_BASE, TAIL_BYTES, 9, 272),
    }
    range_stats = {
        name: {"requests": 0, "words": 0, "first": None, "last": None}
        for name in ranges
    }

    event_counts: Counter[str] = Counter()
    write_req_kinds: Counter[int] = Counter()
    write_last_kinds: Counter[int] = Counter()
    write_done_kinds: Counter[int] = Counter()
    write_done_cycle: dict[int, int] = {}
    first_cycles: dict[str, int] = {}
    read_words = 0
    write_words = 0
    previous_cycle = -1
    cache_write_index = 0
    embed_weight_reqs = 0
    embed_scale_reqs = 0

    with args.trace.open("r", encoding="ascii", newline="") as file:
        reader = csv.DictReader(file)
        require(
            reader.fieldnames
            == ["cycle", "event", "address", "index_or_kind", "data_or_length", "last"],
            "timing trace header mismatch",
        )
        for row in reader:
            cycle = int(row["cycle"])
            event = row["event"]
            address = int(row["address"], 16)
            index_or_kind = int(row["index_or_kind"])
            data_or_length = int(row["data_or_length"], 16)
            require(cycle >= previous_cycle, "trace cycles are not monotonic")
            previous_cycle = cycle
            event_counts[event] += 1
            first_cycles.setdefault(event, cycle)

            if event == "read_req":
                words = index_or_kind
                require(words > 0 and data_or_length == words * 4, "read request shape mismatch")
                read_words += words
                if in_range(address, embed_weight_row, 512):
                    embed_weight_reqs += 1
                    require(address == embed_weight_row and words == 128, "embedding weight read mismatch")
                if in_range(address, embed_scale_row, 32):
                    embed_scale_reqs += 1
                    require(address == embed_scale_row and words == 8, "embedding scale read mismatch")
                for name, (base, byte_count, _, _) in ranges.items():
                    if in_range(address, base, byte_count):
                        stats = range_stats[name]
                        stats["requests"] += 1
                        stats["words"] += words
                        stats["first"] = cycle if stats["first"] is None else stats["first"]
                        stats["last"] = cycle
            elif event == "write_req":
                kind = index_or_kind
                write_req_kinds[kind] += 1
                require(data_or_length > 0 and data_or_length % 4 == 0, "write length mismatch")
                write_words += data_or_length // 4
                expected_shapes = {
                    1: (hidden, VEC1024 * 4),
                    2: (norm, VEC1024 * 4),
                    7: (q_rope, VEC2048 * 4),
                    8: (attn, VEC2048 * 4),
                    9: (oproj_out, VEC1024 * 4),
                    10: (post_hidden, VEC1024 * 4),
                    11: (post_norm, VEC1024 * 4),
                    12: (gate_out, VEC3072 * 4),
                    13: (up_out, VEC3072 * 4),
                    14: (silu_hidden, VEC3072 * 4),
                    15: (down_out, VEC1024 * 4),
                    16: (layer_out, VEC1024 * 4),
                }
                if kind in expected_shapes:
                    require((address, data_or_length) == expected_shapes[kind], f"write kind {kind} shape mismatch")
                elif kind == 3:
                    require(in_range(address, q_base, VEC2048 * 4) and data_or_length == 4, "Q write mismatch")
                elif kind == 4:
                    require(in_range(address, k_base, VEC1024 * 4) and data_or_length == 4, "K write mismatch")
                elif kind == 5:
                    require(in_range(address, v_base, VEC1024 * 4) and data_or_length == 4, "V write mismatch")
                elif kind == 6:
                    require(cache_write_index < len(cache_expected_addr), "too many cache writes")
                    require(address == cache_expected_addr[cache_write_index] and data_or_length == 4,
                            "cache write address/order mismatch")
                    cache_write_index += 1
                else:
                    raise RuntimeError(f"unknown write kind {kind}")
            elif event == "write_last":
                write_last_kinds[index_or_kind] += 1
            elif event == "write_done":
                write_done_kinds[index_or_kind] += 1
                write_done_cycle[index_or_kind] = cycle

    require(event_counts["read_req"] == EXPECTED_READ_REQS, "read request count mismatch")
    require(event_counts["read_last"] == EXPECTED_READ_REQS, "read completion count mismatch")
    require(read_words == EXPECTED_READ_WORDS, "read word count mismatch")
    require(event_counts["write_req"] == EXPECTED_WRITE_REQS, "write request count mismatch")
    require(event_counts["write_last"] == EXPECTED_WRITE_REQS, "write-last count mismatch")
    require(event_counts["write_done"] == EXPECTED_WRITE_REQS, "write completion count mismatch")
    require(write_words == EXPECTED_WRITE_WORDS, "write word count mismatch")
    require(dict(write_req_kinds) == EXPECTED_WRITE_KINDS, "write request kind counts mismatch")
    require(dict(write_last_kinds) == EXPECTED_WRITE_KINDS, "write-last kind counts mismatch")
    require(dict(write_done_kinds) == EXPECTED_WRITE_KINDS, "write-done kind counts mismatch")
    require(cache_write_index == len(cache_expected_addr), "cache write count mismatch")
    require(embed_weight_reqs == 1 and embed_scale_reqs == 1, "embedding row reads mismatch")
    require(event_counts["scheduler_done"] == 1, "scheduler done count mismatch")
    require(event_counts["tail_start"] == 1, "tail start count mismatch")
    require(event_counts["top_done"] == 1, "top done count mismatch")

    for name, (_, _, expected_reqs, expected_words) in ranges.items():
        stats = range_stats[name]
        require(stats["requests"] == expected_reqs, f"{name} read request count mismatch")
        require(stats["words"] == expected_words, f"{name} read word count mismatch")

    require(write_done_cycle[1] < int(range_stats["hidden"]["first"]),
            "input RMSNorm read hidden before embedding response")
    require(write_done_cycle[2] < int(range_stats["norm"]["first"]),
            "QKV read norm before RMSNorm response")
    first_frontend = min(int(range_stats[name]["first"]) for name in ("q", "k", "v"))
    require(write_done_cycle[5] < first_frontend, "frontend read Q/K/V before QKV completion")
    require(write_done_cycle[6] < int(range_stats["cache"]["first"]),
            "score/value read cache before append completion")
    require(write_done_cycle[7] < int(range_stats["q_rope"]["first"]),
            "score read Q RoPE before frontend completion")
    require(write_done_cycle[8] < int(range_stats["attn"]["first"]),
            "o_proj read attention before write completion")
    require(write_done_cycle[9] < int(range_stats["oproj"]["first"]),
            "post-attention read o_proj before completion")
    require(write_done_cycle[11] < int(range_stats["post_norm"]["first"]),
            "gate/up read post norm before completion")
    first_silu_read = min(int(range_stats["gate"]["first"]), int(range_stats["up"]["first"]))
    require(max(write_done_cycle[12], write_done_cycle[13]) < first_silu_read,
            "SiLU read gate/up before both completions")
    require(write_done_cycle[14] < int(range_stats["silu"]["first"]),
            "down read MLP hidden before completion")
    require(write_done_cycle[10] < int(range_stats["post_hidden"]["first"]) and
            write_done_cycle[15] < int(range_stats["down"]["first"]),
            "final residual read an input before completion")
    require(write_done_cycle[16] < first_cycles["scheduler_done"] < first_cycles["tail_start"] < first_cycles["top_done"],
            "layer/scheduler/tail/top ordering mismatch")

    xsim_text = args.xsim_log.read_text(encoding="utf-8", errors="replace")
    require("PASS: tied-Q4 embedding fed the complete Layer 0 scheduler exactly." in xsim_text,
            "XSim PASS marker missing")
    require("FAIL:" not in xsim_text, "XSim log contains a FAIL marker")
    scheduler_match = re.search(r"scheduler rd/wr = (\d+)/(\d+), (\d+)/(\d+)", xsim_text)
    top_match = re.search(r"top rd/wr\s+= (\d+)/(\d+), (\d+)/(\d+)", xsim_text)
    require(scheduler_match is not None and tuple(map(int, scheduler_match.groups())) == (46278, 2140354, 6155, 25600),
            "XSim scheduler summary mismatch")
    require(top_match is not None and tuple(map(int, top_match.groups())) == (46289, 2140762, 6156, 26624),
            "XSim top summary mismatch")

    report = {
        "status": "PASS",
        "trace": str(args.trace.resolve()),
        "event_counts": dict(event_counts),
        "read_words": read_words,
        "write_words": write_words,
        "write_kind_counts": dict(write_req_kinds),
        "read_ranges": range_stats,
        "write_done_cycles": write_done_cycle,
        "ordering": {
            "scheduler_done": first_cycles["scheduler_done"],
            "tail_start": first_cycles["tail_start"],
            "top_done": first_cycles["top_done"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print("PASS: embedding -> complete Layer 0 timing, counts, and addresses are exact.")


if __name__ == "__main__":
    main()
