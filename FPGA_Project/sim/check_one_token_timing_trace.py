#!/usr/bin/env python3
"""Audit one-token top-to-tail ordering from the event CSV trace."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


SCENARIOS = {
    "l1_l2": {
        "layers": [
            (1, 0x0000000415092540),
            (2, 0x0000000425092540),
        ],
        "done_mask": 0x6,
        "scheduler_counters": (92556, 4280708, 12310, 51200),
    },
    "true3": {
        "layers": [
            (0, 0x0000000405092540),
            (1, 0x0000000415092540),
            (2, 0x0000000425092540),
        ],
        "done_mask": 0x7,
        "scheduler_counters": (138820, 6418838, 18464, 75776),
    },
}

VECTOR_BYTES = 1024 * 4
TAIL_HIDDEN_BASE = 0x0000000425092540
TAIL_GAMMA_BASE = 0x0000000405011540
TAIL_NORM_BASE = 0x0000000405012540
TAIL_OUTPUT_BASE = 0x0000000405013540
TAIL_WEIGHT_BASE = 0x0000000400100000
TAIL_SCALE_BASE = 0x0000000404B30000
TAIL_QMAP_BASE = 0x0000000405010000


def parse_int(value: str) -> int:
    return int(value, 0)


def in_range(address: int, base: int, size: int) -> bool:
    return base <= address < base + size


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def first_or_none(values: list[int]) -> int | None:
    return values[0] if values else None


def last_or_none(values: list[int]) -> int | None:
    return values[-1] if values else None


def audit_trace(trace_path: Path, scenario_name: str) -> dict[str, Any]:
    scenario = SCENARIOS[scenario_name]
    layers: list[tuple[int, int]] = scenario["layers"]
    failures: list[str] = []
    row_count = 0
    previous_cycle = -1
    final_row: dict[str, str] | None = None

    write_request_cycles: dict[int, list[int]] = {
        base: [] for _, base in layers
    }
    write_request_cycles[TAIL_NORM_BASE] = []
    write_request_cycles[TAIL_OUTPUT_BASE] = []
    write_done_cycles: dict[int, list[int]] = {
        base: [] for base in write_request_cycles
    }
    consumer_read_cycles: dict[int, list[int]] = {
        base: [] for _, base in layers
    }
    tail_read_cycles = {
        "hidden": [],
        "gamma": [],
        "norm": [],
        "weight": [],
        "scale": [],
    }
    tail_response_cycles = {
        "hidden": [],
        "gamma": [],
        "norm": [],
        "weight": [],
        "scale": [],
    }
    layer_done_mask_cycles: dict[int, int] = {}
    scheduler_complete_cycle: int | None = None
    top_done_cycle: int | None = None
    pending_write: tuple[int, int] | None = None
    pending_read: tuple[int, str | None] | None = None

    def read_category(address: int) -> str | None:
        if in_range(address, TAIL_HIDDEN_BASE, VECTOR_BYTES):
            return "hidden"
        if in_range(address, TAIL_GAMMA_BASE, VECTOR_BYTES):
            return "gamma"
        if in_range(address, TAIL_NORM_BASE, VECTOR_BYTES):
            return "norm"
        if TAIL_WEIGHT_BASE <= address < TAIL_SCALE_BASE:
            return "weight"
        if TAIL_SCALE_BASE <= address < TAIL_QMAP_BASE:
            return "scale"
        return None

    with trace_path.open("r", encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "cycle",
            "done",
            "error",
            "layers_started",
            "layers_completed",
            "loop_state",
            "layer_done",
            "layer_error",
            "stage_error",
            "body_stage_error",
            "rd_req_fire",
            "rd_req_addr",
            "rd_rsp_last_fire",
            "wr_req_fire",
            "wr_req_addr",
            "wr_done",
            "rd_bursts",
            "rd_words",
            "wr_reqs",
            "wr_words",
        }
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"trace is missing columns: {sorted(missing)}")

        for row in reader:
            row_count += 1
            cycle = parse_int(row["cycle"])
            require(cycle >= previous_cycle, f"cycle moved backwards at row {row_count}", failures)
            previous_cycle = cycle
            final_row = row

            if (
                parse_int(row["error"]) != 0
                or parse_int(row["layer_error"]) != 0
                or parse_int(row["stage_error"]) != 0
                or parse_int(row["body_stage_error"]) != 0
            ):
                failures.append(f"error signal observed at cycle {cycle}")

            done_mask = parse_int(row["layer_done"])
            for bit in range(28):
                mask = 1 << bit
                if done_mask & mask and mask not in layer_done_mask_cycles:
                    layer_done_mask_cycles[mask] = cycle

            if (
                scheduler_complete_cycle is None
                and parse_int(row["layers_completed"]) == len(layers)
            ):
                scheduler_complete_cycle = cycle

            if top_done_cycle is None and parse_int(row["done"]) != 0:
                top_done_cycle = cycle

            if parse_int(row["wr_req_fire"]) != 0:
                address = parse_int(row["wr_req_addr"])
                require(
                    pending_write is None,
                    f"new write request at cycle {cycle} before previous write completed",
                    failures,
                )
                pending_write = (address, cycle)
                if address in write_request_cycles:
                    write_request_cycles[address].append(cycle)

            if parse_int(row["wr_done"]) != 0:
                require(pending_write is not None, f"write done without request at cycle {cycle}", failures)
                if pending_write is not None:
                    address, _ = pending_write
                    if address in write_done_cycles:
                        write_done_cycles[address].append(cycle)
                    pending_write = None

            if parse_int(row["rd_req_fire"]) != 0:
                address = parse_int(row["rd_req_addr"])
                category = read_category(address) if parse_int(row["loop_state"]) == 5 else None
                require(
                    pending_read is None,
                    f"new read request at cycle {cycle} before previous read completed",
                    failures,
                )
                pending_read = (address, category)
                if category is not None:
                    tail_read_cycles[category].append(cycle)
                for _, base in layers:
                    if in_range(address, base, VECTOR_BYTES):
                        consumer_read_cycles[base].append(cycle)

            if parse_int(row["rd_rsp_last_fire"]) != 0:
                require(pending_read is not None, f"read last without request at cycle {cycle}", failures)
                if pending_read is not None:
                    _, category = pending_read
                    if category is not None:
                        tail_response_cycles[category].append(cycle)
                    pending_read = None

    require(row_count > 0, "trace has no data rows", failures)
    require(final_row is not None, "trace has no final row", failures)
    require(pending_write is None, "trace ended with an outstanding write", failures)
    require(pending_read is None, "trace ended with an outstanding read", failures)

    expected_mask = scenario["done_mask"]
    expected_counters = scenario["scheduler_counters"]
    if final_row is not None:
        require(parse_int(final_row["done"]) == 1, "final trace row does not assert top done", failures)
        require(parse_int(final_row["error"]) == 0, "final trace row asserts error", failures)
        require(
            parse_int(final_row["layers_started"]) == len(layers),
            "final layers_started mismatch",
            failures,
        )
        require(
            parse_int(final_row["layers_completed"]) == len(layers),
            "final layers_completed mismatch",
            failures,
        )
        require(parse_int(final_row["layer_done"]) == expected_mask, "final layer done mask mismatch", failures)
        actual_counters = (
            parse_int(final_row["rd_bursts"]),
            parse_int(final_row["rd_words"]),
            parse_int(final_row["wr_reqs"]),
            parse_int(final_row["wr_words"]),
        )
        require(actual_counters == expected_counters, "final scheduler counters mismatch", failures)

    for layer_index, base in layers:
        reqs = write_request_cycles[base]
        dones = write_done_cycles[base]
        reads = consumer_read_cycles[base]
        require(len(reqs) == 1, f"layer {layer_index} output write request count is {len(reqs)}, expected 1", failures)
        require(len(dones) == 1, f"layer {layer_index} output write done count is {len(dones)}, expected 1", failures)
        require(len(reads) > 0, f"layer {layer_index} output was never consumed", failures)
        if reqs and dones:
            require(reqs[0] < dones[0], f"layer {layer_index} output write did not complete after request", failures)
        if dones and reads:
            reads_after_done = [cycle for cycle in reads if cycle > dones[0]]
            require(bool(reads_after_done), f"layer {layer_index} output has no read after write completion", failures)
            require(
                all(cycle > dones[0] for cycle in reads),
                f"layer {layer_index} output was read before its write completed",
                failures,
            )

    expected_tail_reads = {
        "hidden": 4,
        "gamma": 4,
        "norm": 4,
        "weight": 75968,
        "scale": 9496,
    }
    for category, expected_count in expected_tail_reads.items():
        require(
            len(tail_read_cycles[category]) == expected_count,
            f"tail {category} read request count is {len(tail_read_cycles[category])}, expected {expected_count}",
            failures,
        )
        require(
            len(tail_response_cycles[category]) == expected_count,
            f"tail {category} read response count is {len(tail_response_cycles[category])}, expected {expected_count}",
            failures,
        )

    require(len(write_request_cycles[TAIL_NORM_BASE]) == 1, "final norm write request count mismatch", failures)
    require(len(write_done_cycles[TAIL_NORM_BASE]) == 1, "final norm write done count mismatch", failures)
    require(len(write_request_cycles[TAIL_OUTPUT_BASE]) == 1, "token/score write request count mismatch", failures)
    require(len(write_done_cycles[TAIL_OUTPUT_BASE]) == 1, "token/score write done count mismatch", failures)

    last_layer_base = layers[-1][1]
    last_layer_done = first_or_none(write_done_cycles[last_layer_base])
    hidden_first = first_or_none(tail_read_cycles["hidden"])
    hidden_last_rsp = last_or_none(tail_response_cycles["hidden"])
    gamma_first = first_or_none(tail_read_cycles["gamma"])
    gamma_last_rsp = last_or_none(tail_response_cycles["gamma"])
    norm_write_req = first_or_none(write_request_cycles[TAIL_NORM_BASE])
    norm_write_done = first_or_none(write_done_cycles[TAIL_NORM_BASE])
    norm_read_first = first_or_none(tail_read_cycles["norm"])
    norm_read_last_rsp = last_or_none(tail_response_cycles["norm"])
    weight_first = first_or_none(tail_read_cycles["weight"])
    tile_last_rsp_candidates = tail_response_cycles["weight"] + tail_response_cycles["scale"]
    tile_last_rsp = max(tile_last_rsp_candidates) if tile_last_rsp_candidates else None
    output_write_req = first_or_none(write_request_cycles[TAIL_OUTPUT_BASE])
    output_write_done = first_or_none(write_done_cycles[TAIL_OUTPUT_BASE])

    ordered_events = [
        ("last layer output write done", last_layer_done),
        ("scheduler complete", scheduler_complete_cycle),
        ("tail hidden first read", hidden_first),
        ("tail hidden last response", hidden_last_rsp),
        ("tail gamma first read", gamma_first),
        ("tail gamma last response", gamma_last_rsp),
        ("final norm write request", norm_write_req),
        ("final norm write done", norm_write_done),
        ("final norm first read", norm_read_first),
        ("final norm last response", norm_read_last_rsp),
        ("LM-head first weight read", weight_first),
        ("LM-head last weight/scale response", tile_last_rsp),
        ("token/score write request", output_write_req),
        ("token/score write done", output_write_done),
        ("top done", top_done_cycle),
    ]
    for name, cycle in ordered_events:
        require(cycle is not None, f"missing event: {name}", failures)
    present_cycles = [cycle for _, cycle in ordered_events if cycle is not None]
    require(
        all(left < right for left, right in zip(present_cycles, present_cycles[1:])),
        "tail event order is not strictly increasing",
        failures,
    )

    return {
        "scenario": scenario_name,
        "trace": str(trace_path),
        "rows": row_count,
        "last_cycle": previous_cycle,
        "events": {name: cycle for name, cycle in ordered_events},
        "tail_read_requests": {key: len(value) for key, value in tail_read_cycles.items()},
        "tail_read_responses": {key: len(value) for key, value in tail_response_cycles.items()},
        "scheduler_counters": {
            "read_requests": expected_counters[0],
            "read_words": expected_counters[1],
            "write_requests": expected_counters[2],
            "write_words": expected_counters[3],
        },
        "failures": failures,
    }


def audit_xsim_log(log_path: Path, report: dict[str, Any]) -> None:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    failures: list[str] = report["failures"]
    require("PASS:" in text, "XSim log has no PASS line", failures)
    require("FAIL:" not in text, "XSim log contains a FAIL line", failures)
    match = re.search(
        r"expected token/score\s*=\s*(-?\d+)\s*/\s*(-?\d+).*?"
        r"observed token/score\s*=\s*(-?\d+)\s*/\s*(-?\d+)",
        text,
        flags=re.DOTALL,
    )
    require(match is not None, "XSim log token/score lines are missing", failures)
    if match is not None:
        expected = (int(match.group(1)), int(match.group(2)))
        observed = (int(match.group(3)), int(match.group(4)))
        report["token_score"] = {"expected": expected, "observed": observed}
        require(expected == observed, "XSim token/score mismatch", failures)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    parser.add_argument("--xsim-log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = audit_trace(args.trace, args.scenario)
    if args.xsim_log is not None:
        audit_xsim_log(args.xsim_log, report)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")

    print(json.dumps(report, indent=2))
    if report["failures"]:
        print(f"FAIL: timing trace audit found {len(report['failures'])} issue(s).")
        return 1
    print("PASS: one-token timing trace ordering and counts are exact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
