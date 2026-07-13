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
EXPECTED_READ_REQS = 8236
EXPECTED_READ_WORDS = 561176
EXPECTED_WRITE_REQS = 4098
EXPECTED_WRITE_WORDS = 6144
EXPECTED_WRITE_KINDS = {1: 1, 2: 1, 3: 2048, 4: 1024, 5: 1024}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit embedding -> Layer 0 RMSNorm -> QKV timing trace."
    )
    parser.add_argument("trace", type=Path)
    parser.add_argument("--chain-manifest", type=Path, required=True)
    parser.add_argument("--qkv-manifest", type=Path, required=True)
    parser.add_argument("--xsim-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def descriptor_base(manifest: dict[str, Any], slot: int) -> int:
    return int(manifest["descriptors"][slot]["base_addr"])


def in_range(address: int, base: int, byte_count: int) -> bool:
    return base <= address < base + byte_count


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    args = parse_args()
    chain = load_json(args.chain_manifest)
    qkv = load_json(args.qkv_manifest)
    addresses = {key: int(value) for key, value in chain["addresses"].items()}
    token_id = int(chain["token_id"])

    embedding_weight_row = addresses["embedding_weight_base"] + token_id * 512
    embedding_scale_row = addresses["embedding_scale_base"] + token_id * 32
    input_hidden = addresses["input_hidden_base"]
    input_norm_output = addresses["input_norm_output_base"]
    q_base = descriptor_base(qkv, 8)
    k_base = descriptor_base(qkv, 9)
    v_base = descriptor_base(qkv, 10)

    event_counts: Counter[str] = Counter()
    write_req_kinds: Counter[int] = Counter()
    write_last_kinds: Counter[int] = Counter()
    write_done_kinds: Counter[int] = Counter()
    read_words = 0
    write_words = 0
    previous_cycle = -1
    first_cycles: dict[str, int] = {}
    last_cycles: dict[str, int] = {}
    hidden_read_reqs = 0
    hidden_read_words = 0
    norm_read_reqs = 0
    norm_read_words = 0
    embedding_weight_reqs = 0
    embedding_scale_reqs = 0

    with args.trace.open("r", encoding="ascii", newline="") as file:
        reader = csv.DictReader(file)
        expected_fields = [
            "cycle",
            "event",
            "address",
            "index_or_kind",
            "data_or_length",
            "last",
        ]
        require(reader.fieldnames == expected_fields, "timing trace header mismatch")
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
            last_cycles[event] = cycle

            if event == "read_req":
                words = index_or_kind
                read_words += words
                if in_range(address, embedding_weight_row, 512):
                    embedding_weight_reqs += 1
                    require(address == embedding_weight_row and words == 128,
                            "embedding weight read shape mismatch")
                if in_range(address, embedding_scale_row, 32):
                    embedding_scale_reqs += 1
                    require(address == embedding_scale_row and words == 8,
                            "embedding scale read shape mismatch")
                if in_range(address, input_hidden, VEC1024 * 4):
                    hidden_read_reqs += 1
                    hidden_read_words += words
                    first_cycles.setdefault("input_hidden_read", cycle)
                if in_range(address, input_norm_output, VEC1024 * 4):
                    norm_read_reqs += 1
                    norm_read_words += words
                    first_cycles.setdefault("input_norm_read", cycle)
            elif event == "write_req":
                kind = index_or_kind
                write_req_kinds[kind] += 1
                require(data_or_length > 0 and data_or_length % 4 == 0,
                        "write request length is zero or unaligned")
                write_words += data_or_length // 4
                if kind == 1:
                    require(address == input_hidden and data_or_length == VEC1024 * 4,
                            "embedding write request mismatch")
                elif kind == 2:
                    require(address == input_norm_output and data_or_length == VEC1024 * 4,
                            "input RMSNorm write request mismatch")
                elif kind == 3:
                    require(in_range(address, q_base, VEC2048 * 4) and data_or_length == 4,
                            "Q write request mismatch")
                elif kind == 4:
                    require(in_range(address, k_base, VEC1024 * 4) and data_or_length == 4,
                            "K write request mismatch")
                elif kind == 5:
                    require(in_range(address, v_base, VEC1024 * 4) and data_or_length == 4,
                            "V write request mismatch")
                else:
                    raise RuntimeError(f"unknown write kind {kind}")
            elif event == "write_last":
                write_last_kinds[index_or_kind] += 1
            elif event == "write_done":
                write_done_kinds[index_or_kind] += 1
                if index_or_kind == 1:
                    last_cycles["embedding_done"] = cycle
                elif index_or_kind == 2:
                    last_cycles["input_norm_done"] = cycle
                elif index_or_kind in (3, 4, 5):
                    last_cycles["qkv_done"] = cycle

    require(event_counts["read_req"] == EXPECTED_READ_REQS,
            "read request count mismatch")
    require(event_counts["read_last"] == EXPECTED_READ_REQS,
            "read completion count mismatch")
    require(read_words == EXPECTED_READ_WORDS, "read word count mismatch")
    require(event_counts["write_req"] == EXPECTED_WRITE_REQS,
            "write request count mismatch")
    require(event_counts["write_last"] == EXPECTED_WRITE_REQS,
            "write-last count mismatch")
    require(event_counts["write_done"] == EXPECTED_WRITE_REQS,
            "write completion count mismatch")
    require(write_words == EXPECTED_WRITE_WORDS, "write word count mismatch")
    require(dict(write_req_kinds) == EXPECTED_WRITE_KINDS,
            "write request kind counts mismatch")
    require(dict(write_last_kinds) == EXPECTED_WRITE_KINDS,
            "write-last kind counts mismatch")
    require(dict(write_done_kinds) == EXPECTED_WRITE_KINDS,
            "write completion kind counts mismatch")
    require(event_counts["scheduler_done"] == 1, "scheduler done count mismatch")
    require(event_counts["top_done"] == 1, "top done count mismatch")
    require(embedding_weight_reqs == 1 and embedding_scale_reqs == 1,
            "embedding row read counts mismatch")
    require(hidden_read_reqs == 4 and hidden_read_words == VEC1024,
            "input hidden read counts mismatch")
    require(norm_read_reqs == 4 and norm_read_words == VEC1024,
            "input norm read counts mismatch")

    require(last_cycles["embedding_done"] < first_cycles["input_hidden_read"],
            "input RMSNorm read hidden before embedding write response")
    require(last_cycles["input_norm_done"] < first_cycles["input_norm_read"],
            "QKV read activation before RMSNorm write response")
    require(last_cycles["qkv_done"] < first_cycles["scheduler_done"],
            "scheduler completed before final QKV write response")
    require(first_cycles["scheduler_done"] < first_cycles["top_done"],
            "top completed before scheduler")

    xsim_text = args.xsim_log.read_text(encoding="utf-8", errors="replace")
    require("PASS: tied-Q4 embedding fed Layer 0 input RMSNorm and full QKV exactly."
            in xsim_text, "XSim PASS marker missing")
    require("FAIL:" not in xsim_text, "XSim log contains a FAIL marker")
    cycle_matches = {
        key: re.search(pattern, xsim_text)
        for key, pattern in {
            "embedding": r"cycles embedding_done/input_read = (\d+)/(\d+)",
            "norm": r"cycles norm_done/qkv_read\s+= (\d+)/(\d+)",
            "completion": r"cycles qkv_done/scheduler/top\s+= (\d+)/(\d+)/(\d+)",
        }.items()
    }
    require(all(match is not None for match in cycle_matches.values()),
            "XSim cycle summary is incomplete")

    report = {
        "status": "PASS",
        "trace": str(args.trace.resolve()),
        "event_counts": dict(event_counts),
        "read_words": read_words,
        "write_words": write_words,
        "write_kind_counts": dict(write_req_kinds),
        "ordering": {
            "embedding_done": last_cycles["embedding_done"],
            "first_input_hidden_read": first_cycles["input_hidden_read"],
            "input_norm_done": last_cycles["input_norm_done"],
            "first_input_norm_read": first_cycles["input_norm_read"],
            "final_qkv_write_done": last_cycles["qkv_done"],
            "scheduler_done": first_cycles["scheduler_done"],
            "top_done": first_cycles["top_done"],
        },
        "addresses": {
            "embedding_weight_row": embedding_weight_row,
            "embedding_scale_row": embedding_scale_row,
            "input_hidden": input_hidden,
            "input_norm_output": input_norm_output,
            "q_base": q_base,
            "k_base": k_base,
            "v_base": v_base,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print("PASS: embedding -> Layer 0 RMSNorm -> QKV timing and counts are exact.")


if __name__ == "__main__":
    main()
