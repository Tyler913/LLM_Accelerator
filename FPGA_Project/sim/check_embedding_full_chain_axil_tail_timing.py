from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEDULER_READ_REQS_PER_LAYER = 46_278
SCHEDULER_READ_WORDS_PER_LAYER = 2_140_354
SCHEDULER_WRITE_REQS_PER_LAYER = 6_155
SCHEDULER_WRITE_WORDS_PER_LAYER = 25_600
TOP_EXTRA_READ_REQS = 85_496
TOP_EXTRA_READ_WORDS = 20_667_048
TOP_EXTRA_WRITE_REQS = 3
TOP_EXTRA_WRITE_WORDS = 2_051

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

PL_DDR_BASE = 0x0000_0004_0000_0000
PL_DDR_BYTES = 0x2000_0000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit manifest-driven AXI-Lite tied-Q4 embedding through an arbitrary "
            "decoder depth and the full-vocabulary final-token tail."
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


def read_hex_addresses(path: Path) -> list[int]:
    values: list[int] = []
    with path.open("r", encoding="ascii") as source:
        for line in source:
            text = line.strip()
            if text and not text.startswith("//"):
                values.append(int(text, 16))
    return values


def relative_file(root: Path, layer: dict[str, Any], key: str) -> Path:
    path = (root / str(layer["files"][key])).resolve()
    require(path.is_file(), f"missing layer vector {key}: {path}")
    return path


def expected_write_kind_counts(layer_count: int) -> dict[int, int]:
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


def resolve_hidden_contract(
    manifest: dict[str, Any], layers: list[dict[str, Any]]
) -> tuple[bool, int, list[int], list[int], int]:
    """Resolve the addresses actually used by the embedding/layer/tail chain.

    Legacy manifests describe the physical chain directly in ``output_base`` and
    each layer's ``input_hidden_base``/``output_hidden_base`` fields.  Runtime
    manifests keep those descriptor-golden addresses but execute from two shared
    hidden buffers.  New runtime manifests publish the actual addresses explicitly;
    manifests produced before that schema correction are accepted only when their
    stale fields exactly match the corresponding descriptor-golden address.
    """

    embedding = manifest["embedding"]
    final_tail = manifest["final_tail"]
    runtime_context = manifest.get("runtime_context")
    runtime_enabled = bool(
        isinstance(runtime_context, dict) and runtime_context.get("enabled", False)
    )
    if not runtime_enabled:
        embedding_output = int(embedding["output_base"])
        layer_inputs = [int(layer["input_hidden_base"]) for layer in layers]
        layer_outputs = [int(layer["output_hidden_base"]) for layer in layers]
        require(layer_inputs[0] == embedding_output, "Layer 0 input does not use embedding output")
        for layer_index in range(1, len(layers)):
            require(
                layer_inputs[layer_index] == layer_outputs[layer_index - 1],
                f"Layer {layer_index} input does not chain from the previous layer",
            )
        final_hidden = int(final_tail["runtime_hidden_override"])
        require(
            final_hidden == layer_outputs[-1],
            "final hidden override does not select the last layer output",
        )
        return False, embedding_output, layer_inputs, layer_outputs, final_hidden

    assert isinstance(runtime_context, dict)
    require(runtime_context.get("hidden_a_base") is not None, "runtime hidden A base is missing")
    require(runtime_context.get("hidden_b_base") is not None, "runtime hidden B base is missing")
    hidden_a = int(runtime_context["hidden_a_base"])
    hidden_b = int(runtime_context["hidden_b_base"])
    require(hidden_a != 0 and hidden_b != 0, "runtime hidden bases must be non-zero")
    require(hidden_a != hidden_b, "runtime hidden A and B bases must differ")
    require((hidden_a | hidden_b) & 0x3 == 0, "runtime hidden bases must be word aligned")

    expected_inputs = [hidden_a if (index & 1) == 0 else hidden_b for index in range(len(layers))]
    expected_outputs = [hidden_b if (index & 1) == 0 else hidden_a for index in range(len(layers))]

    published_embedding_output = int(embedding["output_base"])
    descriptor_embedding_output = int(
        embedding.get("descriptor_golden_output_base", layers[0]["input_hidden_base"])
    )
    if "descriptor_golden_output_base" in embedding:
        require(
            published_embedding_output == hidden_a,
            "runtime embedding output_base does not select hidden A",
        )
        require(
            descriptor_embedding_output == int(layers[0]["input_hidden_base"]),
            "embedding descriptor-golden output disagrees with Layer 0 descriptor input",
        )
    elif published_embedding_output != hidden_a:
        require(
            published_embedding_output == descriptor_embedding_output,
            "old runtime embedding output is neither hidden A nor the descriptor-golden address",
        )

    layer_inputs: list[int] = []
    layer_outputs: list[int] = []
    for layer_index, layer in enumerate(layers):
        runtime_input = int(layer.get("runtime_input_hidden_base", expected_inputs[layer_index]))
        runtime_output = int(layer.get("runtime_output_hidden_base", expected_outputs[layer_index]))
        require(
            runtime_input == expected_inputs[layer_index],
            f"Layer {layer_index} runtime input does not follow A/B ping-pong parity",
        )
        require(
            runtime_output == expected_outputs[layer_index],
            f"Layer {layer_index} runtime output does not follow A/B ping-pong parity",
        )
        layer_inputs.append(runtime_input)
        layer_outputs.append(runtime_output)

    final_hidden = layer_outputs[-1]
    require(
        runtime_context.get("last_layer_hidden_base") is not None,
        "runtime last-layer hidden base is missing",
    )
    require(
        int(runtime_context["last_layer_hidden_base"]) == final_hidden,
        "runtime last-layer hidden base does not match A/B parity",
    )

    published_tail_hidden = int(final_tail["runtime_hidden_override"])
    descriptor_tail_hidden = int(
        final_tail.get("descriptor_golden_hidden_base", layers[-1]["output_hidden_base"])
    )
    if "descriptor_golden_hidden_base" in final_tail:
        require(
            published_tail_hidden == final_hidden,
            "runtime final hidden override does not select the last A/B output",
        )
        require(
            descriptor_tail_hidden == int(layers[-1]["output_hidden_base"]),
            "final-tail descriptor-golden hidden disagrees with the last layer descriptor output",
        )
    elif published_tail_hidden != final_hidden:
        require(
            published_tail_hidden == descriptor_tail_hidden,
            "old runtime final hidden override is neither the last A/B output nor the descriptor address",
        )

    return True, hidden_a, layer_inputs, layer_outputs, final_hidden


def expected_write_requests(
    root: Path, layers: list[dict[str, Any]], layer_outputs: list[int]
) -> dict[int, list[tuple[int, int]]]:
    expected: dict[int, list[tuple[int, int]]] = defaultdict(list)
    row_specs = (
        (WRITE_Q, "q", 2048),
        (WRITE_K, "k", 1024),
        (WRITE_V, "v", 1024),
    )
    burst_specs = (
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
    )
    for layer in layers:
        bases = layer["write_bases"]
        for kind, key, count in row_specs:
            base = int(bases[key])
            expected[kind].extend((base + 4 * index, 4) for index in range(count))
        cache_addresses = read_hex_addresses(relative_file(root, layer, "expected_cache_addr"))
        require(len(cache_addresses) == 2048, "cache expected-address vector length mismatch")
        expected[WRITE_CACHE].extend((address, 4) for address in cache_addresses)
        for kind, key, byte_count in burst_specs:
            expected[kind].append((int(bases[key]), byte_count))
    expected[WRITE_LAYER].extend((address, 4096) for address in layer_outputs)
    return expected


def main() -> None:
    args = parse_args()
    trace_path = args.trace.resolve()
    manifest_path = args.manifest.resolve()
    root = manifest_path.parent
    manifest = load_json(manifest_path)
    layers = list(manifest["layers"])
    layer_count = int(manifest["layer_count"])
    require(layer_count == len(layers), "manifest layer count disagrees with layer list")
    require(3 <= layer_count <= 28, "full-chain audit expects 3..28 layers")
    require(manifest["address_audit"]["status"] == "PASS", "manifest address audit failed")

    token_id = int(manifest["token_id"])
    embedding = manifest["embedding"]
    weight_base = int(embedding["weight_base"])
    scale_base = int(embedding["scale_base"])
    runtime_enabled, embedding_output, layer_inputs, layer_outputs, final_hidden = (
        resolve_hidden_contract(manifest, layers)
    )
    weight_row = weight_base + token_id * 512
    scale_row = scale_base + token_id * 32

    final_tail = manifest["final_tail"]
    require(final_hidden == layer_outputs[-1], "resolved final hidden does not match Layer N output")
    expected_token = int(final_tail["best_token"])
    expected_score = int(final_tail["best_score_q26"])

    expected_scheduler = (
        layer_count * SCHEDULER_READ_REQS_PER_LAYER,
        layer_count * SCHEDULER_READ_WORDS_PER_LAYER,
        layer_count * SCHEDULER_WRITE_REQS_PER_LAYER,
        layer_count * SCHEDULER_WRITE_WORDS_PER_LAYER,
    )
    expected_top = (
        expected_scheduler[0] + TOP_EXTRA_READ_REQS,
        expected_scheduler[1] + TOP_EXTRA_READ_WORDS,
        expected_scheduler[2] + TOP_EXTRA_WRITE_REQS,
        expected_scheduler[3] + TOP_EXTRA_WRITE_WORDS,
    )
    expected_kinds = expected_write_kind_counts(layer_count)
    expected_requests = expected_write_requests(root, layers, layer_outputs)

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
            qmap_base_labels[int(layer["qmap_bases"][key])] = f"layer{layer_index}:{key}"
    qmap_base_labels[int(final_tail["qmap_base"])] = "final_tail"
    seen_qmap_bases: set[int] = set()

    events: Counter[str] = Counter()
    write_kinds: Counter[int] = Counter()
    write_done_kinds: Counter[int] = Counter()
    read_words = 0
    write_words = 0
    previous_cycle = -1
    active_read: tuple[int, int, int] | None = None
    active_write: tuple[int, int, int, int] | None = None
    pending_write: tuple[int, int, int, int, int] | None = None
    first_event_cycle: dict[str, int] = {}
    write_done_by_kind: dict[int, list[tuple[int, int]]] = defaultdict(list)
    write_req_by_kind: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    hidden_read_cycles: dict[int, list[int]] = defaultdict(list)
    first_two_reads: list[tuple[int, int, int]] = []

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
                require(
                    in_range(address, PL_DDR_BASE, PL_DDR_BYTES)
                    and in_range(address + data_or_length - 1, PL_DDR_BASE, PL_DDR_BYTES),
                    f"read request leaves PL DDR aperture: 0x{address:016x}",
                )
                active_read = (address, words, cycle)
                if len(first_two_reads) < 2:
                    first_two_reads.append((cycle, address, words))
                read_words += words
                if address in qmap_base_labels:
                    seen_qmap_bases.add(address)
                for hidden_base in set(layer_inputs + [layer_outputs[-1]]):
                    if in_range(address, hidden_base, 4096):
                        hidden_read_cycles[hidden_base].append(cycle)
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
                require(
                    in_range(address, PL_DDR_BASE, PL_DDR_BYTES)
                    and in_range(address + data_or_length - 1, PL_DDR_BASE, PL_DDR_BYTES),
                    f"write request leaves PL DDR aperture: 0x{address:016x}",
                )
                active_write = (address, kind, data_or_length, cycle)
                write_kinds[kind] += 1
                write_words += data_or_length // 4
                write_req_by_kind[kind].append((cycle, address, data_or_length))
            elif event == "write_last":
                require(active_write is not None, "write-last without request")
                base, kind, length, request_cycle = active_write
                require(index_or_kind == kind, "write-last kind mismatch")
                require(cycle >= request_cycle, "write-last preceded request")
                require(address == base + length - 4, "write-last address mismatch")
                pending_write = (base, kind, length, request_cycle, cycle)
                active_write = None
            elif event == "write_done":
                require(pending_write is not None, "write response without completed write data")
                base, kind, _length, request_cycle, last_cycle = pending_write
                require(address == base and index_or_kind == kind, "write response target mismatch")
                require(cycle > request_cycle and cycle > last_cycle, "write response ordering mismatch")
                write_done_kinds[kind] += 1
                write_done_by_kind[kind].append((cycle, address))
                pending_write = None

    require(active_read is None, "trace ended with an active read")
    require(active_write is None and pending_write is None, "trace ended with an active write")
    require(events["read_req"] == expected_top[0], "aggregate read request count mismatch")
    require(events["read_last"] == expected_top[0], "aggregate read completion count mismatch")
    require(read_words == expected_top[1], "aggregate read word count mismatch")
    require(events["write_req"] == expected_top[2], "aggregate write request count mismatch")
    require(events["write_last"] == expected_top[2], "aggregate write-last count mismatch")
    require(events["write_done"] == expected_top[2], "aggregate write response count mismatch")
    require(write_words == expected_top[3], "aggregate write word count mismatch")
    require(dict(write_kinds) == expected_kinds, "write request kind counts mismatch")
    require(dict(write_done_kinds) == expected_kinds, "write response kind counts mismatch")
    require(events["scheduler_done"] == 1, "scheduler-done pulse count mismatch")
    require(events["tail_start"] == 1, "tail-start pulse count mismatch")
    require(events["top_done"] == 1, "top-done pulse count mismatch")

    require(len(first_two_reads) == 2, "missing embedding reads")
    require(first_two_reads[0][1:] == (weight_row, 128), "first read is not embedding weight row")
    require(first_two_reads[1][1:] == (scale_row, 8), "second read is not embedding scale row")
    missing_qmaps = sorted(set(qmap_base_labels) - seen_qmap_bases)
    require(
        not missing_qmaps,
        "QMAP packets never read: " + ", ".join(qmap_base_labels[base] for base in missing_qmaps),
    )

    for kind, expected in expected_requests.items():
        observed = [(address, length) for _, address, length in write_req_by_kind[kind]]
        require(observed == expected, f"write request address sequence mismatch for kind {kind}")
    embedding_requests = write_req_by_kind[WRITE_EMBEDDING]
    require(
        [(address, length) for _, address, length in embedding_requests]
        == [(embedding_output, 4096)],
        "embedding write request shape mismatch",
    )
    require(
        [(address, length) for _, address, length in write_req_by_kind[WRITE_TAIL_NORM]]
        and write_req_by_kind[WRITE_TAIL_NORM][0][2] == 4096,
        "tail norm write shape mismatch",
    )
    require(
        len(write_req_by_kind[WRITE_TAIL_OUTPUT]) == 1
        and write_req_by_kind[WRITE_TAIL_OUTPUT][0][2] == 12,
        "tail output write shape mismatch",
    )

    embedding_done = write_done_by_kind[WRITE_EMBEDDING]
    layer_done = write_done_by_kind[WRITE_LAYER]
    require(len(embedding_done) == 1, "embedding write response count mismatch")
    require(
        [address for _, address in layer_done] == layer_outputs,
        "layer output response sequence mismatch",
    )
    embedding_done_cycle = embedding_done[0][0]
    layer_done_cycles = [cycle for cycle, _ in layer_done]
    first_layer_reads: list[int] = []
    first_layer_reads.append(first_after(hidden_read_cycles[layer_inputs[0]], embedding_done_cycle))
    require(
        first_layer_reads[0] > embedding_done_cycle,
        "Layer 0 read hidden before embedding response",
    )
    for layer_index in range(1, layer_count):
        first_read = first_after(
            hidden_read_cycles[layer_inputs[layer_index]], layer_done_cycles[layer_index - 1]
        )
        first_layer_reads.append(first_read)
        require(
            first_read > layer_done_cycles[layer_index - 1],
            f"Layer {layer_index} read hidden before Layer {layer_index - 1} response",
        )

    scheduler_done_cycle = first_event_cycle["scheduler_done"]
    tail_start_cycle = first_event_cycle["tail_start"]
    top_done_cycle = first_event_cycle["top_done"]
    first_tail_hidden_read = first_after(hidden_read_cycles[layer_outputs[-1]], tail_start_cycle)
    require(
        layer_done_cycles[-1]
        < scheduler_done_cycle
        < tail_start_cycle
        < first_tail_hidden_read
        < top_done_cycle,
        "last layer, scheduler, tail, and top ordering mismatch",
    )

    xsim_text = args.xsim_log.read_text(encoding="utf-8", errors="replace")
    require("FAIL:" not in xsim_text, "XSim log contains a FAIL marker")
    require(
        f"PASS: AXI-Lite tied-Q4 embedding ran through {layer_count} complete layers "
        "and the full-vocabulary final-token tail exactly." in xsim_text,
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
    require(scheduler_summary == expected_scheduler, "scheduler summary mismatch")
    require(top_summary == expected_top, "top summary mismatch")
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
        "layer_count": layer_count,
        "runtime_context_enabled": runtime_enabled,
        "token_id": token_id,
        "final_token": expected_token,
        "final_score_q26": expected_score,
        "event_counts": dict(events),
        "read_words": read_words,
        "write_words": write_words,
        "write_kind_counts": dict(write_kinds),
        "qmap_packets_seen": [qmap_base_labels[base] for base in sorted(seen_qmap_bases)],
        "addresses": {
            "embedding_weight_row": f"0x{weight_row:016x}",
            "embedding_scale_row": f"0x{scale_row:016x}",
            "embedding_output": f"0x{embedding_output:016x}",
            "layer_inputs": [f"0x{address:016x}" for address in layer_inputs],
            "layer_outputs": [f"0x{address:016x}" for address in layer_outputs],
            "final_hidden": f"0x{final_hidden:016x}",
        },
        "cycles": {
            "embedding_response": embedding_done_cycle,
            "layer_first_hidden_reads": first_layer_reads,
            "layer_responses": layer_done_cycles,
            "scheduler_done": scheduler_done_cycle,
            "tail_start": tail_start_cycle,
            "tail_first_hidden_read": first_tail_hidden_read,
            "top_done": top_done_cycle,
        },
        "scheduler_counters": expected_scheduler,
        "top_counters": expected_top,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print(
        f"PASS: AXI-Lite embedding full-chain timing and counts are exact "
        f"for {layer_count} layers."
    )


if __name__ == "__main__":
    main()
