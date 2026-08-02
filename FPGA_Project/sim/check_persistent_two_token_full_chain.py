from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PL_DDR_BASE = 0x0000_0004_0000_0000
PL_DDR_BYTES = 0x2000_0000

SCHEDULER_READ_REQS_AT_CACHE_LENGTH_5 = 46_278
SCHEDULER_READ_WORDS_AT_CACHE_LENGTH_5 = 2_140_354
SCHEDULER_WRITE_REQS = 6_155
SCHEDULER_WRITE_WORDS = 25_600
# The resource-reduced LM head reads each of 151936 rows as one packed-Q4
# weight request followed by one scale request.  The other 32 requests are the
# two embedding-row reads plus QMAP/header and final-norm-side traffic.
TOP_EXTRA_READ_REQS = 303_904
TOP_EXTRA_READ_WORDS = 20_667_048
TOP_EXTRA_WRITE_REQS = 3
TOP_EXTRA_WRITE_WORDS = 2_051

KV_HEADS = 8
Q_HEADS = 16
HEAD_DIM = 128
MAX_CONTEXT = 256
KV_KIND_BYTES = KV_HEADS * MAX_CONTEXT * HEAD_DIM * 4
KV_LAYER_BYTES = 2 * KV_KIND_BYTES
CACHE_READS_PER_KIND_POSITION = Q_HEADS * HEAD_DIM
CACHE_READS_PER_POSITION = 2 * CACHE_READS_PER_KIND_POSITION

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit one no-reset RuntimeContext process containing two feedback-closed "
            "full-chain decode starts."
        )
    )
    parser.add_argument("trace", type=Path)
    parser.add_argument("--full-chain-manifest", type=Path, required=True)
    parser.add_argument("--persistent-manifest", type=Path, required=True)
    parser.add_argument("--xsim-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def integer(value: Any) -> int:
    return int(value, 0) if isinstance(value, str) else int(value)


def in_range(address: int, base: int, byte_count: int) -> bool:
    return base <= address < base + byte_count


def read_hex(path: Path) -> list[int]:
    require(path.is_file(), f"missing vector file: {path}")
    values: list[int] = []
    with path.open("r", encoding="ascii") as source:
        for line in source:
            text = line.strip()
            if text and not text.startswith("//"):
                values.append(int(text, 16))
    return values


def qmap_descriptor_base(path: Path, slot: int) -> int:
    words = read_hex(path)
    low_index = 64 + slot * 32 + 8
    require(low_index + 1 < len(words), f"descriptor slot {slot} escapes {path}")
    return words[low_index] | (words[low_index + 1] << 32)


def expected_kind_counts(layer_count: int) -> dict[int, int]:
    return {
        WRITE_Q: layer_count * 2048,
        WRITE_K: layer_count * 1024,
        WRITE_V: layer_count * 1024,
        WRITE_CACHE: layer_count * 2048,
        WRITE_Q_ROPE: layer_count,
        WRITE_ATTN: layer_count,
        WRITE_O_PROJ: layer_count,
        WRITE_POST_HIDDEN: layer_count,
        WRITE_POST_NORM: layer_count,
        WRITE_GATE: layer_count,
        WRITE_UP: layer_count,
        WRITE_SILU: layer_count,
        WRITE_DOWN: layer_count,
        WRITE_LAYER: layer_count,
        WRITE_INPUT_NORM: layer_count,
        WRITE_EMBEDDING: 1,
        WRITE_TAIL_NORM: 1,
        WRITE_TAIL_OUTPUT: 1,
    }


def expected_write_requests(
    full_manifest: dict[str, Any],
    persistent_root: Path,
    step: int,
    tail_norm_base: int,
    tail_output_base: int,
) -> dict[int, list[tuple[int, int]]]:
    expected: dict[int, list[tuple[int, int]]] = defaultdict(list)
    layers = list(full_manifest["layers"])
    for layer_index, layer in enumerate(layers):
        bases = layer["write_bases"]
        for kind, key, count in (
            (WRITE_Q, "q", 2048),
            (WRITE_K, "k", 1024),
            (WRITE_V, "v", 1024),
        ):
            base = integer(bases[key])
            expected[kind].extend((base + index * 4, 4) for index in range(count))

        cache_path = (
            persistent_root
            / f"layer_{layer_index:02d}"
            / f"position_{step:03d}"
            / "kv_write_addr.hex"
        )
        cache_addresses = read_hex(cache_path)
        require(
            len(cache_addresses) == 2048,
            f"Layer {layer_index} step {step} cache-address length mismatch",
        )
        expected[WRITE_CACHE].extend((address, 4) for address in cache_addresses)

        for kind, key, length in (
            (WRITE_INPUT_NORM, "input_norm", 4096),
            (WRITE_Q_ROPE, "q_rope", 8192),
            (WRITE_ATTN, "attn_out", 8192),
            (WRITE_O_PROJ, "o_proj", 4096),
            (WRITE_POST_HIDDEN, "post_hidden", 4096),
            (WRITE_POST_NORM, "post_norm", 4096),
            (WRITE_GATE, "gate", 12288),
            (WRITE_UP, "up", 12288),
            (WRITE_SILU, "silu", 12288),
            (WRITE_DOWN, "down", 4096),
        ):
            expected[kind].append((integer(bases[key]), length))

        runtime_output = integer(
            layer.get("runtime_output_hidden_base", bases["layer"])
        )
        expected[WRITE_LAYER].append((runtime_output, 4096))

    hidden_a = integer(full_manifest["runtime_context"]["hidden_a_base"])
    expected[WRITE_EMBEDDING].append((hidden_a, 4096))
    expected[WRITE_TAIL_NORM].append((tail_norm_base, 4096))
    expected[WRITE_TAIL_OUTPUT].append((tail_output_base, 12))
    return expected


def decode_cache_address(
    address: int, cache_base: int, layer_count: int
) -> tuple[int, int, int, int, int] | None:
    if not in_range(address, cache_base, layer_count * KV_LAYER_BYTES):
        return None
    offset = address - cache_base
    layer = offset // KV_LAYER_BYTES
    local = offset % KV_LAYER_BYTES
    kind = local // KV_KIND_BYTES
    within_kind = local % KV_KIND_BYTES
    head_stride = MAX_CONTEXT * HEAD_DIM * 4
    head = within_kind // head_stride
    within_head = within_kind % head_stride
    position = within_head // (HEAD_DIM * 4)
    dim = (within_head % (HEAD_DIM * 4)) // 4
    return layer, kind, head, position, dim


def main() -> None:
    args = parse_args()
    trace_path = args.trace.resolve()
    full_path = args.full_chain_manifest.resolve()
    persistent_path = args.persistent_manifest.resolve()
    full = load_json(full_path)
    persistent = load_json(persistent_path)
    persistent_root = persistent_path.parent

    layers = list(full["layers"])
    layer_count = integer(full["layer_count"])
    require(layer_count == len(layers), "full-chain layer count/list mismatch")
    require(3 <= layer_count <= 28, "persistent full-chain audit expects 3..28 layers")
    require(full["address_audit"]["status"] == "PASS", "full-chain address audit failed")
    runtime = full.get("runtime_context")
    require(
        isinstance(runtime, dict) and runtime.get("enabled") is True,
        "full-chain manifest is not RuntimeContext-enabled",
    )
    require(
        integer(persistent["layer_count"]) == layer_count
        and [integer(value) for value in persistent["layer_ids"]]
        == list(range(layer_count)),
        "persistent/full-chain layer contract mismatch",
    )
    require(
        persistent["self_check"]["status"] == "PASS"
        and persistent["self_check"]["cache_retention_every_layer"] is True
        and persistent["self_check"]["full_vocab_argmax_checked"] is True,
        "persistent golden self-check/retention/full-vocabulary scan failed",
    )
    if layer_count == 28:
        require(
            persistent.get("model_complete") is True
            and persistent.get("valid_as_full_model_decode") is True
            and persistent.get("tail_semantics") == "full_model_decode",
            "full28 persistent golden is not classified as a complete model decode",
        )
    elif "model_complete" in persistent:
        require(
            persistent["model_complete"] is False
            and persistent.get("valid_as_full_model_decode") is False
            and persistent.get("tail_semantics") == "truncated_prefix_diagnostic",
            "partial persistent golden is not classified as a truncated-prefix diagnostic",
        )

    tokens = [integer(value) for value in persistent["token_ids"]]
    positions = [integer(value) for value in persistent["positions"]]
    results = list(persistent["final_tail"]["position_results"])
    expected_outputs = [integer(result["argmax_token"]) for result in results]
    expected_scores = [integer(result["argmax_score_q26"]) for result in results]
    require(len(tokens) == len(results) == 2, "persistent golden must contain two tokens")
    require(positions == [0, 1], "persistent golden positions must be exactly 0/1")
    require(tokens[1] == expected_outputs[0], "persistent golden is not feedback closed")

    cache_base = integer(persistent["cache"]["base_addr"])
    require(integer(persistent["cache"]["max_context"]) == MAX_CONTEXT, "cache context mismatch")
    require(
        integer(persistent["cache"]["writes_per_layer_position"]) == 2048,
        "cache write-count contract mismatch",
    )
    require(
        all(integer(layer["cache_slice"]["base"]) == cache_base + index * KV_LAYER_BYTES
            for index, layer in enumerate(layers)),
        "full-chain cache slices disagree with persistent cache layout",
    )

    embedding = full["embedding"]
    weight_base = integer(embedding["weight_base"])
    scale_base = integer(embedding["scale_base"])
    tail_image = (full_path.parent / full["final_tail"]["files"]["qmap_image"]).resolve()
    tail_norm_base = qmap_descriptor_base(tail_image, 1)
    tail_output_base = qmap_descriptor_base(tail_image, 4)
    qmap_base_labels: dict[int, str] = {}
    for layer_index, layer in enumerate(layers):
        for key in (
            "qkv",
            "input_rmsnorm",
            "attn_frontend",
            "attn_score_value",
            "o_proj",
            "post_attn_norm",
            "mlp_gate_up",
            "mlp_silu_mul",
            "mlp_down",
            "mlp_residual_add",
        ):
            qmap_base_labels[integer(layer["qmap_bases"][key])] = (
                f"layer{layer_index}:{key}"
            )
    qmap_base_labels[integer(full["final_tail"]["qmap_base"])] = "final_tail"

    expected_counts = expected_kind_counts(layer_count)
    expected_requests = [
        expected_write_requests(full, persistent_root, step, tail_norm_base, tail_output_base)
        for step in range(2)
    ]

    event_counts = [Counter(), Counter()]
    write_kind_counts = [Counter(), Counter()]
    write_done_kind_counts = [Counter(), Counter()]
    write_requests: list[dict[int, list[tuple[int, int, int]]]] = [
        defaultdict(list),
        defaultdict(list),
    ]
    write_done_cycles: list[dict[int, list[tuple[int, int]]]] = [
        defaultdict(list),
        defaultdict(list),
    ]
    cache_reads = [Counter(), Counter()]
    seen_qmap_bases: list[set[int]] = [set(), set()]
    cache_read_cycles: list[dict[tuple[int, int, int], list[int]]] = [
        defaultdict(list),
        defaultdict(list),
    ]
    first_reads: list[list[tuple[int, int, int]]] = [[], []]
    first_event_cycle: list[dict[str, int]] = [{}, {}]
    read_words = [0, 0]
    write_words = [0, 0]
    step_start_cycles = [-1, -1]
    step_done_cycles = [-1, -1]

    active_step: int | None = None
    active_read: tuple[int, int, int] | None = None
    active_write: tuple[int, int, int, int] | None = None
    pending_write: tuple[int, int, int, int, int] | None = None
    previous_cycle = -1
    next_step = 0

    with trace_path.open("r", encoding="ascii", newline="") as source:
        reader = csv.DictReader(source)
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

            if event == "persistent_step_start":
                require(active_step is None, "overlapping persistent steps")
                require(next_step < 2 and index_or_kind == next_step, "step-start order mismatch")
                require(
                    data_or_length == tokens[next_step],
                    f"step {next_step} start token mismatch",
                )
                active_step = next_step
                step_start_cycles[next_step] = cycle
                next_step += 1
                continue

            if event == "persistent_step_done":
                require(active_step is not None, "step-done without active step")
                step = active_step
                require(index_or_kind == step, "step-done id mismatch")
                require(data_or_length == expected_outputs[step], "step-done token mismatch")
                require(
                    active_read is None and active_write is None and pending_write is None,
                    "step ended with an active memory transaction",
                )
                step_done_cycles[step] = cycle
                active_step = None
                continue

            require(active_step is not None, f"event {event} occurred outside a decode step")
            step = active_step
            event_counts[step][event] += 1
            first_event_cycle[step].setdefault(event, cycle)

            if event == "read_req":
                words = index_or_kind
                require(active_read is None, "overlapping read requests")
                require(words > 0 and data_or_length == words * 4, "read length mismatch")
                require(
                    in_range(address, PL_DDR_BASE, PL_DDR_BYTES)
                    and in_range(address + data_or_length - 1, PL_DDR_BASE, PL_DDR_BYTES),
                    f"read leaves PL DDR: 0x{address:016x}",
                )
                active_read = (address, words, cycle)
                read_words[step] += words
                if address in qmap_base_labels:
                    seen_qmap_bases[step].add(address)
                if len(first_reads[step]) < 2:
                    first_reads[step].append((cycle, address, words))
                decoded = decode_cache_address(address, cache_base, layer_count)
                if decoded is not None:
                    layer, kind, _head, position, _dim = decoded
                    require(words == 1, "cache read is not one word")
                    require(position <= step, "decode read a future cache position")
                    key = (layer, kind, position)
                    cache_reads[step][key] += 1
                    cache_read_cycles[step][key].append(cycle)
            elif event == "read_last":
                require(active_read is not None, "read-last without request")
                base, words, request_cycle = active_read
                require(cycle > request_cycle, "read-last preceded request")
                require(index_or_kind == words, "read-last word count mismatch")
                require(address == base + (words - 1) * 4, "read-last address mismatch")
                active_read = None
            elif event == "write_req":
                kind = index_or_kind
                require(
                    active_write is None and pending_write is None,
                    "overlapping write requests",
                )
                require(data_or_length > 0 and data_or_length % 4 == 0, "write length mismatch")
                require(
                    in_range(address, PL_DDR_BASE, PL_DDR_BYTES)
                    and in_range(address + data_or_length - 1, PL_DDR_BASE, PL_DDR_BYTES),
                    f"write leaves PL DDR: 0x{address:016x}",
                )
                active_write = (address, kind, data_or_length, cycle)
                write_kind_counts[step][kind] += 1
                write_words[step] += data_or_length // 4
                write_requests[step][kind].append((cycle, address, data_or_length))
            elif event == "write_last":
                require(active_write is not None, "write-last without request")
                base, kind, length, request_cycle = active_write
                require(index_or_kind == kind, "write-last kind mismatch")
                require(cycle >= request_cycle, "write-last preceded request")
                require(address == base + length - 4, "write-last address mismatch")
                pending_write = (base, kind, length, request_cycle, cycle)
                active_write = None
            elif event == "write_done":
                require(pending_write is not None, "write-done without completed data")
                base, kind, _length, request_cycle, last_cycle = pending_write
                require(address == base and index_or_kind == kind, "write-done target mismatch")
                require(cycle > request_cycle and cycle > last_cycle, "write-done ordering mismatch")
                write_done_kind_counts[step][kind] += 1
                write_done_cycles[step][kind].append((cycle, address))
                pending_write = None

    require(next_step == 2 and active_step is None, "trace does not contain two complete steps")
    require(step_start_cycles[0] < step_done_cycles[0] < step_start_cycles[1] < step_done_cycles[1],
            "two-step marker ordering mismatch")

    step_reports: list[dict[str, Any]] = []
    for step in range(2):
        cache_length = step + 1
        layer_read_reqs = SCHEDULER_READ_REQS_AT_CACHE_LENGTH_5 - (
            (5 - cache_length) * CACHE_READS_PER_POSITION
        )
        layer_read_words = SCHEDULER_READ_WORDS_AT_CACHE_LENGTH_5 - (
            (5 - cache_length) * CACHE_READS_PER_POSITION
        )
        expected_top_reads = layer_count * layer_read_reqs + TOP_EXTRA_READ_REQS
        expected_top_read_words = layer_count * layer_read_words + TOP_EXTRA_READ_WORDS
        expected_top_writes = layer_count * SCHEDULER_WRITE_REQS + TOP_EXTRA_WRITE_REQS
        expected_top_write_words = layer_count * SCHEDULER_WRITE_WORDS + TOP_EXTRA_WRITE_WORDS

        require(event_counts[step]["read_req"] == expected_top_reads, f"step {step} read count mismatch")
        require(event_counts[step]["read_last"] == expected_top_reads, f"step {step} read-last count mismatch")
        require(read_words[step] == expected_top_read_words, f"step {step} read words mismatch")
        require(event_counts[step]["write_req"] == expected_top_writes, f"step {step} write count mismatch")
        require(event_counts[step]["write_last"] == expected_top_writes, f"step {step} write-last count mismatch")
        require(event_counts[step]["write_done"] == expected_top_writes, f"step {step} write-done count mismatch")
        require(write_words[step] == expected_top_write_words, f"step {step} write words mismatch")
        require(dict(write_kind_counts[step]) == expected_counts, f"step {step} write-kind count mismatch")
        require(dict(write_done_kind_counts[step]) == expected_counts, f"step {step} write-done-kind count mismatch")
        for event in ("scheduler_done", "tail_start", "top_done"):
            require(event_counts[step][event] == 1, f"step {step} {event} pulse mismatch")

        expected_first_reads = [
            (weight_base + tokens[step] * 512, 128),
            (scale_base + tokens[step] * 32, 8),
        ]
        require(
            [(address, words) for _, address, words in first_reads[step]]
            == expected_first_reads,
            f"step {step} embedding row reads mismatch",
        )
        missing_qmaps = sorted(set(qmap_base_labels) - seen_qmap_bases[step])
        require(
            not missing_qmaps,
            f"step {step} QMAP packets never read: "
            + ", ".join(qmap_base_labels[base] for base in missing_qmaps),
        )
        for kind, requests in expected_requests[step].items():
            observed = [
                (address, length)
                for _, address, length in write_requests[step][kind]
            ]
            require(observed == requests, f"step {step} write address sequence mismatch for kind {kind}")

        for layer in range(layer_count):
            for kind in range(2):
                for position in range(cache_length):
                    key = (layer, kind, position)
                    require(
                        cache_reads[step][key] == CACHE_READS_PER_KIND_POSITION,
                        f"step {step} cache reads mismatch at layer/kind/position {key}",
                    )
        require(
            sum(cache_reads[step].values())
            == layer_count * cache_length * CACHE_READS_PER_POSITION,
            f"step {step} aggregate cache read count mismatch",
        )

        last_layer_done = write_done_cycles[step][WRITE_LAYER][-1][0]
        scheduler_done = first_event_cycle[step]["scheduler_done"]
        tail_start = first_event_cycle[step]["tail_start"]
        top_done = first_event_cycle[step]["top_done"]
        require(
            step_start_cycles[step]
            < last_layer_done
            < scheduler_done
            < tail_start
            < top_done
            < step_done_cycles[step],
            f"step {step} scheduler/tail/top ordering mismatch",
        )
        step_reports.append(
            {
                "step": step,
                "input_token": tokens[step],
                "output_token": expected_outputs[step],
                "output_score_q26": expected_scores[step],
                "cache_length": cache_length,
                "read_requests": expected_top_reads,
                "read_words": expected_top_read_words,
                "write_requests": expected_top_writes,
                "write_words": expected_top_write_words,
                "qmap_packets_seen": len(seen_qmap_bases[step]),
                "cycles": {
                    "start": step_start_cycles[step],
                    "last_layer_write_done": last_layer_done,
                    "scheduler_done": scheduler_done,
                    "tail_start": tail_start,
                    "top_done": top_done,
                    "done": step_done_cycles[step],
                },
            }
        )

    step0_cache_done = max(cycle for cycle, _ in write_done_cycles[0][WRITE_CACHE])
    step1_prior_read = min(
        cycle
        for (layer, kind, position), cycles in cache_read_cycles[1].items()
        if position == 0
        for cycle in cycles
    )
    require(
        step0_cache_done < step1_prior_read,
        "step 1 read retained position 0 before step 0 cache writes completed",
    )

    xsim_text = args.xsim_log.read_text(encoding="utf-8", errors="replace")
    require("FAIL:" not in xsim_text, "XSim log contains a FAIL marker")
    matches = re.findall(
        r"PERSISTENT_STEP_RESULT step=(\d+) input_token=(\d+) position=(\d+) "
        r"output_token=(\d+) output_score=(-?\d+)",
        xsim_text,
    )
    require(len(matches) == 2, "XSim log does not contain exactly two step results")
    for step, match in enumerate(matches):
        observed = tuple(map(int, match))
        expected = (step, tokens[step], step, expected_outputs[step], expected_scores[step])
        require(observed == expected, f"XSim step {step} token/score summary mismatch")
    require(
        re.search(r"PERSISTENT_NO_RESET reset_release_count=1\b", xsim_text) is not None,
        "single-reset-release proof is missing",
    )
    require(
        f"PASS: AXI-Lite RuntimeContext persistent two-token decode ran through "
        f"{layer_count} complete layers without reset." in xsim_text,
        "persistent XSim PASS marker is missing",
    )

    report = {
        "status": "PASS",
        "trace": str(trace_path),
        "full_chain_manifest": str(full_path),
        "persistent_manifest": str(persistent_path),
        "layer_count": layer_count,
        "feedback_closed_tokens": tokens,
        "expected_outputs": expected_outputs,
        "cache": {
            "base": f"0x{cache_base:016x}",
            "step0_last_write_done": step0_cache_done,
            "step1_first_retained_read": step1_prior_read,
            "retained_position0_read_by_every_layer": True,
        },
        "steps": step_reports,
        "reset_release_count": 1,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print("PASS: persistent two-token full-chain timing and retention are exact")


if __name__ == "__main__":
    main()
