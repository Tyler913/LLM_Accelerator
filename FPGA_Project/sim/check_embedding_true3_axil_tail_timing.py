from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_READ_REQS = 224330
EXPECTED_READ_WORDS = 27088110
EXPECTED_WRITE_REQS = 18468
EXPECTED_WRITE_WORDS = 78851
EXPECTED_SCHEDULER = (138834, 6421062, 18465, 76800)
EXPECTED_TOP = (
    EXPECTED_READ_REQS,
    EXPECTED_READ_WORDS,
    EXPECTED_WRITE_REQS,
    EXPECTED_WRITE_WORDS,
)

WRITE_Q = 1
WRITE_K = 2
WRITE_V = 3
WRITE_CACHE = 4
WRITE_Q_ROPE = 5
WRITE_ATTN = 6
WRITE_O_PROJ = 7
WRITE_POST_HIDDEN = 8
WRITE_POST_NORM = 9
WRITE_GATE = 10
WRITE_UP = 11
WRITE_SILU = 12
WRITE_DOWN = 13
WRITE_LAYER = 14
WRITE_INPUT_NORM = 15
WRITE_EMBEDDING = 16
WRITE_TAIL_NORM = 101
WRITE_TAIL_OUTPUT = 102

EXPECTED_WRITE_KINDS = {
    WRITE_Q: 3 * 2048,
    WRITE_K: 3 * 1024,
    WRITE_V: 3 * 1024,
    WRITE_CACHE: 3 * 2 * 1024,
    WRITE_Q_ROPE: 3,
    WRITE_ATTN: 3,
    WRITE_O_PROJ: 3,
    WRITE_POST_HIDDEN: 3,
    WRITE_POST_NORM: 3,
    WRITE_GATE: 3,
    WRITE_UP: 3,
    WRITE_SILU: 3,
    WRITE_DOWN: 3,
    WRITE_LAYER: 3,
    WRITE_INPUT_NORM: 3,
    WRITE_EMBEDDING: 1,
    WRITE_TAIL_NORM: 1,
    WRITE_TAIL_OUTPUT: 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit AXI-Lite tied-Q4 embedding through three complete layers "
            "and the full-vocabulary final-token tail."
        )
    )
    parser.add_argument("trace", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--xsim-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def in_range(address: int, base: int, byte_count: int) -> bool:
    return base <= address < base + byte_count


def first_after(cycles: list[int], threshold: int) -> int:
    for cycle in cycles:
        if cycle > threshold:
            return cycle
    return -1


def parse_summary(text: str, pattern: str, name: str) -> tuple[int, int, int, int]:
    match = re.search(pattern, text)
    require(match is not None, f"missing {name} summary in XSim log")
    assert match is not None
    return tuple(map(int, match.groups()))  # type: ignore[return-value]


def main() -> None:
    args = parse_args()
    trace_path = args.trace.resolve()
    manifest_path = args.manifest.resolve()
    root = manifest_path.parent
    manifest = load_json(manifest_path)
    layer0_manifest = load_json(root / "embedding_layer0" / "full_chain_manifest.json")
    tail_manifest = load_json(
        root / "final_tail" / "qmap" / "final_token_tail_embedding_true3_manifest.json"
    )

    token_id = int(manifest["token_id"])
    addresses = manifest["addresses"]
    layer_outputs = [
        int(addresses["layer0_output"]),
        int(addresses["layer1_output"]),
        int(addresses["layer2_output"]),
    ]
    require(
        int(addresses["final_hidden_runtime_override"]) == layer_outputs[2],
        "final hidden override does not select Layer 2 output",
    )
    embedding_addresses = layer0_manifest["addresses"]
    weight_base = int(embedding_addresses["embedding_weight_base"])
    scale_base = int(embedding_addresses["embedding_scale_base"])
    qkv_staging_base = int(embedding_addresses["qkv_staging_base"])
    weight_row = weight_base + token_id * 512
    scale_row = scale_base + token_id * 32
    legacy_qkv_base = 0x0000_0004_0008_0000

    final_expected = manifest["final"]
    expected_token = int(final_expected["best_token"])
    expected_score = int(final_expected["best_score_q26"])
    require(
        expected_token == int(tail_manifest["expected"]["best_token"])
        and expected_score == int(tail_manifest["expected"]["best_score_q26"]),
        "true3 and tail manifests disagree on token or score",
    )

    events: Counter[str] = Counter()
    write_kinds: Counter[int] = Counter()
    write_done_kinds: Counter[int] = Counter()
    read_words = 0
    write_words = 0
    previous_cycle = -1
    active_read: tuple[int, int, int] | None = None
    active_write: tuple[int, int, int, int] | None = None
    pending_write: tuple[int, int, int, int] | None = None
    first_event_cycle: dict[str, int] = {}
    write_done_by_kind: dict[int, list[tuple[int, int]]] = {}
    write_req_by_kind: dict[int, list[tuple[int, int, int]]] = {}
    read_cycles_by_range: dict[str, list[int]] = {
        "layer0_hidden": [],
        "layer1_hidden": [],
        "layer2_hidden": [],
        "qkv_staging": [],
        "legacy_qkv": [],
    }
    read_requests: list[tuple[int, int, int]] = []

    with trace_path.open("r", encoding="ascii", newline="") as file:
        reader = csv.DictReader(file)
        require(
            reader.fieldnames
            == ["cycle", "event", "address", "index_or_kind", "data_or_length", "last"],
            "event trace header mismatch",
        )
        for row in reader:
            cycle = int(row["cycle"])
            event = row["event"]
            address = int(row["address"], 16)
            index_or_kind = int(row["index_or_kind"])
            data_or_length = int(row["data_or_length"], 16)
            require(cycle >= previous_cycle, "trace cycles are not monotonic")
            previous_cycle = cycle
            events[event] += 1
            first_event_cycle.setdefault(event, cycle)

            if event == "read_req":
                words = index_or_kind
                require(active_read is None, "overlapping read requests in memory model")
                require(words > 0 and data_or_length == words * 4, "read request length mismatch")
                active_read = (address, words, cycle)
                read_requests.append((cycle, address, words))
                read_words += words
                if in_range(address, layer_outputs[0], 4096):
                    read_cycles_by_range["layer0_hidden"].append(cycle)
                if in_range(address, layer_outputs[1], 4096):
                    read_cycles_by_range["layer1_hidden"].append(cycle)
                if in_range(address, layer_outputs[2], 4096):
                    read_cycles_by_range["layer2_hidden"].append(cycle)
                if in_range(address, qkv_staging_base, 0x22B000):
                    read_cycles_by_range["qkv_staging"].append(cycle)
                if address == legacy_qkv_base:
                    read_cycles_by_range["legacy_qkv"].append(cycle)
            elif event == "read_last":
                require(active_read is not None, "read completion without request")
                base, words, request_cycle = active_read
                require(cycle > request_cycle, "read completion did not follow request")
                require(index_or_kind == words, "read completion word count mismatch")
                require(address == base + (words - 1) * 4, "read completion address mismatch")
                active_read = None
            elif event == "write_req":
                kind = index_or_kind
                require(active_write is None and pending_write is None, "overlapping write requests")
                require(data_or_length > 0 and data_or_length % 4 == 0, "write request length mismatch")
                active_write = (address, kind, data_or_length, cycle)
                write_kinds[kind] += 1
                write_words += data_or_length // 4
                write_req_by_kind.setdefault(kind, []).append((cycle, address, data_or_length))
            elif event == "write_last":
                require(active_write is not None, "write-last without request")
                base, kind, length, request_cycle = active_write
                require(index_or_kind == kind, "write-last kind mismatch")
                require(cycle >= request_cycle, "write-last preceded request")
                require(address == base + length - 4, "write-last address mismatch")
                pending_write = active_write
                active_write = None
            elif event == "write_done":
                require(pending_write is not None, "write response without completed write data")
                base, kind, length, request_cycle = pending_write
                require(address == base and index_or_kind == kind, "write response target mismatch")
                require(cycle > request_cycle, "write response did not follow request")
                write_done_kinds[kind] += 1
                write_done_by_kind.setdefault(kind, []).append((cycle, address))
                pending_write = None

    require(active_read is None, "trace ended with an active read")
    require(active_write is None and pending_write is None, "trace ended with an active write")
    require(events["read_req"] == EXPECTED_READ_REQS, "aggregate read request count mismatch")
    require(events["read_last"] == EXPECTED_READ_REQS, "aggregate read completion count mismatch")
    require(read_words == EXPECTED_READ_WORDS, "aggregate read word count mismatch")
    require(events["write_req"] == EXPECTED_WRITE_REQS, "aggregate write request count mismatch")
    require(events["write_last"] == EXPECTED_WRITE_REQS, "aggregate write-last count mismatch")
    require(events["write_done"] == EXPECTED_WRITE_REQS, "aggregate write response count mismatch")
    require(write_words == EXPECTED_WRITE_WORDS, "aggregate write word count mismatch")
    require(dict(write_kinds) == EXPECTED_WRITE_KINDS, "write request kind counts mismatch")
    require(dict(write_done_kinds) == EXPECTED_WRITE_KINDS, "write response kind counts mismatch")
    require(events["scheduler_done"] == 1, "scheduler-done pulse count mismatch")
    require(events["tail_start"] == 1, "tail-start pulse count mismatch")
    require(events["top_done"] == 1, "top-done pulse count mismatch")

    require(len(read_requests) >= 2, "missing embedding reads")
    require(read_requests[0][1:] == (weight_row, 128), "first read is not the selected embedding weight row")
    require(read_requests[1][1:] == (scale_row, 8), "second read is not the selected embedding scale row")
    require(read_cycles_by_range["qkv_staging"], "Layer 0 staged QKV packet was never read")
    require(not read_cycles_by_range["legacy_qkv"], "legacy Layer 0 QKV packet was read")

    embedding_requests = write_req_by_kind[WRITE_EMBEDDING]
    embedding_done = write_done_by_kind[WRITE_EMBEDDING]
    require(embedding_requests == [(embedding_requests[0][0], layer_outputs[0], 4096)],
            "embedding write request shape mismatch")
    require(len(embedding_done) == 1 and embedding_done[0][1] == layer_outputs[0],
            "embedding write response mismatch")

    layer_requests = write_req_by_kind[WRITE_LAYER]
    layer_done = write_done_by_kind[WRITE_LAYER]
    require(
        [(address, length) for _, address, length in layer_requests]
        == [(address, 4096) for address in layer_outputs],
        "layer output write sequence mismatch",
    )
    require(
        [address for _, address in layer_done] == layer_outputs,
        "layer output response sequence mismatch",
    )

    embed_done_cycle = embedding_done[0][0]
    layer_done_cycles = [cycle for cycle, _ in layer_done]
    layer0_reads = read_cycles_by_range["layer0_hidden"]
    layer1_reads = read_cycles_by_range["layer1_hidden"]
    layer2_reads = read_cycles_by_range["layer2_hidden"]
    first_layer0_read = first_after(layer0_reads, embed_done_cycle)
    first_layer1_read = first_after(layer0_reads, layer_done_cycles[0])
    first_layer2_read = first_after(layer1_reads, layer_done_cycles[1])
    scheduler_done_cycle = first_event_cycle["scheduler_done"]
    tail_start_cycle = first_event_cycle["tail_start"]
    top_done_cycle = first_event_cycle["top_done"]
    first_tail_hidden_read = first_after(layer2_reads, tail_start_cycle)

    require(first_layer0_read > embed_done_cycle, "Layer 0 read hidden before embedding response")
    require(first_layer1_read > layer_done_cycles[0], "Layer 1 read hidden before Layer 0 response")
    require(first_layer2_read > layer_done_cycles[1], "Layer 2 read hidden before Layer 1 response")
    require(
        layer_done_cycles[2] < scheduler_done_cycle < tail_start_cycle < first_tail_hidden_read < top_done_cycle,
        "Layer 2, scheduler, tail, and top ordering mismatch",
    )

    xsim_text = args.xsim_log.read_text(encoding="utf-8", errors="replace")
    require("FAIL:" not in xsim_text, "XSim log contains a FAIL marker")
    require(
        "PASS: AXI-Lite tied-Q4 embedding ran through three complete layers and the "
        "full-vocabulary final-token tail exactly." in xsim_text,
        "XSim PASS marker missing",
    )
    scheduler_summary = parse_summary(
        xsim_text,
        r"scheduler rd/wr\s*=\s*(\d+)/(\d+) reads,\s*(\d+)/(\d+) writes",
        "scheduler",
    )
    top_summary = parse_summary(
        xsim_text,
        r"top rd/wr counters\s*=\s*(\d+)/(\d+) reads,\s*(\d+)/(\d+) writes",
        "top",
    )
    require(scheduler_summary == EXPECTED_SCHEDULER, "scheduler summary mismatch")
    require(top_summary == EXPECTED_TOP, "top summary mismatch")
    token_match = re.search(r"observed token/score\s*=\s*(\d+)\s*/\s*(-?\d+)", xsim_text)
    require(token_match is not None, "observed token/score summary missing")
    assert token_match is not None
    require(
        tuple(map(int, token_match.groups())) == (expected_token, expected_score),
        "observed final token or score mismatch",
    )

    report = {
        "status": "PASS",
        "trace": str(trace_path),
        "manifest": str(manifest_path),
        "token_id": token_id,
        "final_token": expected_token,
        "final_score_q26": expected_score,
        "event_counts": dict(events),
        "read_words": read_words,
        "write_words": write_words,
        "write_kind_counts": dict(write_kinds),
        "addresses": {
            "embedding_weight_row": f"0x{weight_row:016x}",
            "embedding_scale_row": f"0x{scale_row:016x}",
            "qkv_staging": f"0x{qkv_staging_base:016x}",
            "layer_outputs": [f"0x{address:016x}" for address in layer_outputs],
        },
        "cycles": {
            "embedding_response": embed_done_cycle,
            "layer0_first_read": first_layer0_read,
            "layer0_response": layer_done_cycles[0],
            "layer1_first_read": first_layer1_read,
            "layer1_response": layer_done_cycles[1],
            "layer2_first_read": first_layer2_read,
            "layer2_response": layer_done_cycles[2],
            "scheduler_done": scheduler_done_cycle,
            "tail_start": tail_start_cycle,
            "tail_first_hidden_read": first_tail_hidden_read,
            "top_done": top_done_cycle,
        },
        "scheduler_counters": scheduler_summary,
        "top_counters": top_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print("PASS: AXI-Lite embedding true3 final-tail timing and counts are exact.")


if __name__ == "__main__":
    main()
