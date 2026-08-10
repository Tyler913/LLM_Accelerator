from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_OUTPUT_TOKENS = [28458, 64]
EXPECTED_OUTPUT_SCORES_Q26 = [1227344433, 1015661901]
PL_DDR_BASE = 0x0000000400000000
PL_DDR_END_EXCLUSIVE = 0x0000000420000000
DEFAULT_PL_TARGET_ASSIGNMENT = 'set device_filter {name =~ "PL"}'
PACKAGE_FORMAT_VERSION = 2
PACKAGE_NAME = "qwen3_0p6b_full28_fpga_board_test"
EXPECTED_TOKENIZER_ASSET_BYTES = 3_629_566
EXPECTED_TOKENIZER_ASSET_SHA256 = (
    "c20242603ef4144e3f3f2ec4ba97c0e9c315aadd41f1bd2c5740e2a7ffa03a7d"
)

REPORT_FILES = (
    "board_build_artifacts.txt",
    "check_timing_verbose.rpt",
    "impl_1_status.txt",
    "post_route_clock_utilization.rpt",
    "post_route_drc.rpt",
    "post_route_methodology.rpt",
    "post_route_power.rpt",
    "post_route_status.rpt",
    "post_route_timing_summary.rpt",
    "post_route_utilization.rpt",
    "post_route_utilization_hierarchical.rpt",
    "stdout.log",
    "synth_1_status.txt",
)

REGRESSION_FILES = (
    ("summary.txt", "summary.txt"),
    ("xsim/persistent_timing_audit.json", "persistent_timing_audit.json"),
    ("xsim/xsim.log", "xsim.log"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"{label} is empty: {path}")
    return path


def path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return path_is_within(left, right) or path_is_within(right, left)


def validate_output_location(
    output: Path,
    repo_root: Path,
    protected_paths: list[Path],
) -> None:
    output = output.resolve()
    repo_root = repo_root.resolve()
    if output == repo_root or not path_is_within(output, repo_root):
        raise RuntimeError(
            f"Output must be a child directory inside the repository: {output}"
        )

    for protected in protected_paths:
        protected = protected.resolve()
        if paths_overlap(output, protected):
            raise RuntimeError(
                "Output must not overlap a release input or durable source: "
                f"output={output}, protected={protected}"
            )


def require_replaceable_release(path: Path) -> None:
    if not path.is_dir():
        raise RuntimeError(f"Refusing to replace a non-directory output: {path}")
    manifest_path = path / "package_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            "Refusing --force replacement because the target is not a "
            f"recognized board release: {path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Refusing --force replacement of an unreadable release: {path}"
        ) from exc
    if (
        manifest.get("name") != PACKAGE_NAME
        or manifest.get("format_version") not in (1, PACKAGE_FORMAT_VERSION)
    ):
        raise RuntimeError(
            "Refusing --force replacement of an unrecognized release: "
            f"{path}"
        )


def promote_staged_release(staging: Path, output: Path, *, replace: bool) -> None:
    if not output.exists():
        staging.rename(output)
        return
    if not replace:
        raise FileExistsError(f"Output already exists: {output}")

    backup = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.backup-",
            dir=output.parent,
        )
    )
    backup.rmdir()
    output.rename(backup)
    try:
        staging.rename(output)
    except BaseException:
        backup.rename(output)
        raise
    try:
        shutil.rmtree(backup)
    except OSError as exc:
        print(f"WARNING: replaced release backup could not be removed: {exc}")


def require_exact_sequence(
    actual: list[str], expected: list[str], label: str
) -> None:
    if actual == expected:
        return
    mismatch = next(
        (
            index
            for index in range(max(len(actual), len(expected)))
            if index >= len(actual)
            or index >= len(expected)
            or actual[index] != expected[index]
        ),
        0,
    )
    actual_line = actual[mismatch] if mismatch < len(actual) else "<missing>"
    expected_line = expected[mismatch] if mismatch < len(expected) else "<none>"
    raise RuntimeError(
        f"{label} mismatch at operation {mismatch}: "
        f"expected {expected_line!r}, got {actual_line!r}"
    )


def validate_board_launcher(path: Path) -> dict[str, Any]:
    path = require_file(path, "board launcher")
    text = path.read_text(encoding="utf-8")
    default_assignments = [
        line.rstrip()
        for line in text.splitlines()
        if line.startswith("set device_filter ")
    ]
    require_exact_sequence(
        default_assignments,
        [DEFAULT_PL_TARGET_ASSIGNMENT],
        "Board launcher default PL target selector",
    )
    selection = 'select_unique_target $device_filter "PL device"'
    selection_count = sum(
        line.rstrip() == selection for line in text.splitlines()
    )
    if selection_count != 1:
        raise RuntimeError(
            "Board launcher must select exactly one PL device through "
            "device_filter"
        )
    required_generate_contract = (
        "set generate_elf [file join $package_root sw a_qgen.elf]",
        'if {$mode ni {"model" "control" "generate"}} {',
        'require_file $generate_elf "interactive generation ELF"',
        "dow $generate_elf",
    )
    for marker in required_generate_contract:
        if marker not in text:
            raise RuntimeError(
                f"Board launcher lacks interactive generation contract: {marker}"
            )
    return {
        "sha256": sha256_file(path),
        "default_device_filter": 'name =~ "PL"',
        "unique_pl_selection": True,
        "generate_mode": True,
    }


def validate_runtime_loader(
    loader_path: Path, segments: list[dict[str, Any]]
) -> dict[str, Any]:
    loader_text = loader_path.read_text(encoding="utf-8")
    actual_operations = [
        line.rstrip()
        for line in loader_text.splitlines()
        if line.startswith("set segment_file ") or line.startswith("dow ")
    ]
    expected_operations: list[str] = []
    for segment in segments:
        expected_operations.extend(
            (
                'set segment_file [file join $script_dir '
                f'"{segment["file"]}"]',
                "dow -data -bypass-cache-sync $segment_file "
                f'0x{int(segment["address"]):016X}',
            )
        )
    require_exact_sequence(
        actual_operations,
        expected_operations,
        "Runtime XSDB loader",
    )
    return {
        "sha256": sha256_file(loader_path),
        "segment_download_count": len(segments),
        "download_command": "dow -data -bypass-cache-sync",
        "file_address_order_matches_manifest": True,
    }


def validate_xsa_bitstream(xsa: Path, bitstream: Path) -> dict[str, Any]:
    bitstream_sha256 = sha256_file(bitstream)
    with zipfile.ZipFile(xsa) as archive:
        bit_entries = [
            name for name in archive.namelist() if name.lower().endswith(".bit")
        ]
        if len(bit_entries) != 1:
            raise RuntimeError(
                f"Expected one embedded bitstream in {xsa}, found {bit_entries}"
            )
        digest = hashlib.sha256()
        with archive.open(bit_entries[0]) as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != bitstream_sha256:
            raise RuntimeError("XSA embedded bitstream does not match release bit")
    return {
        "bitstream_sha256": bitstream_sha256,
        "xsa_sha256": sha256_file(xsa),
        "embedded_bitstream": bit_entries[0],
        "embedded_bitstream_matches": True,
    }


def copy_file(source: Path, destination: Path, *, hardlink: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    if hardlink:
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def validate_runtime_image(runtime_dir: Path) -> dict[str, Any]:
    manifest_path = require_file(
        runtime_dir / "pl_ddr_binary_segments.json",
        "runtime segment manifest",
    )
    loader_path = require_file(
        runtime_dir / "load_pl_ddr_runtime.tcl",
        "runtime XSDB loader",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    segments = manifest.get("segments", [])
    if manifest.get("segment_count") != len(segments) or len(segments) != 61:
        raise RuntimeError(
            "The board release requires exactly 61 runtime segments"
        )
    for source_key in ("source_load_plan", "source_manifest"):
        source_name = manifest.get(source_key)
        expected_sha256 = manifest.get(f"{source_key}_sha256")
        if not source_name or not expected_sha256:
            raise RuntimeError(
                f"Runtime segment manifest lacks {source_key} provenance"
            )
        source_path = require_file(
            runtime_dir / source_name,
            f"runtime {source_key}",
        )
        if sha256_file(source_path) != expected_sha256:
            raise RuntimeError(
                f"Runtime {source_key} provenance SHA256 mismatch"
            )
    full_chain_manifest = json.loads(
        (runtime_dir / manifest["source_manifest"]).read_text(encoding="utf-8")
    )

    previous_end = PL_DDR_BASE
    total_bytes = 0
    for index, segment in enumerate(segments):
        if segment.get("index") != index:
            raise RuntimeError(f"Runtime segment index mismatch at {index}")
        segment_path = require_file(
            runtime_dir / segment["file"],
            f"runtime segment {index}",
        )
        if segment_path.stat().st_size != segment["nbytes"]:
            raise RuntimeError(f"Runtime segment size mismatch: {segment_path}")
        if sha256_file(segment_path) != segment["sha256"]:
            raise RuntimeError(f"Runtime segment SHA256 mismatch: {segment_path}")

        address = int(segment["address"])
        end = address + int(segment["nbytes"])
        if address < previous_end:
            raise RuntimeError(f"Runtime segments overlap at index {index}")
        if address < PL_DDR_BASE or end > PL_DDR_END_EXCLUSIVE:
            raise RuntimeError(
                f"Runtime segment {index} is outside the 512 MiB PL-DDR aperture"
            )
        previous_end = end
        total_bytes += int(segment["nbytes"])

    if total_bytes != manifest.get("total_segment_bytes"):
        raise RuntimeError("Runtime manifest total_segment_bytes mismatch")
    loader_audit = validate_runtime_loader(loader_path, segments)

    mutable_sizes = {
        "input_norm": 1024 * 4,
        "q": 2048 * 4,
        "k": 1024 * 4,
        "v": 1024 * 4,
        "q_rope": 2048 * 4,
        "attn_out": 2048 * 4,
        "o_proj": 1024 * 4,
        "post_hidden": 1024 * 4,
        "post_norm": 1024 * 4,
        "gate": 3072 * 4,
        "up": 3072 * 4,
        "silu": 3072 * 4,
        "down": 1024 * 4,
        "layer": 1024 * 4,
    }
    mutable_regions: list[tuple[str, int, int]] = []
    for layer in full_chain_manifest["layers"]:
        layer_id = int(layer["layer_id"])
        write_bases = layer["write_bases"]
        for name, nbytes in mutable_sizes.items():
            mutable_regions.append(
                (
                    f"layer{layer_id}.{name}",
                    int(write_bases[name]),
                    nbytes,
                )
            )

    address_formula = full_chain_manifest["address_formula"]
    runtime_context = full_chain_manifest["runtime_context"]
    mutable_regions.extend(
        (
            (
                "kv_cache",
                int(address_formula["kv_cache_base"]),
                int(full_chain_manifest["layer_count"])
                * int(address_formula["kv_cache_bytes_per_layer"]),
            ),
            (
                "runtime_hidden_a",
                int(runtime_context["hidden_a_base"]),
                1024 * 4,
            ),
            (
                "runtime_hidden_b",
                int(runtime_context["hidden_b_base"]),
                1024 * 4,
            ),
        )
    )
    final_tail_base = int(full_chain_manifest["final_tail"]["qmap_base"])
    mutable_regions.extend(
        (
            ("final_tail.final_norm", final_tail_base + 0x2540, 1024 * 4),
            ("final_tail.output", final_tail_base + 0x3540, 3 * 4),
        )
    )

    def require_zero_region(name: str, address: int, nbytes: int) -> None:
        cursor = address
        remaining = nbytes
        while remaining:
            containing = next(
                (
                    segment
                    for segment in segments
                    if int(segment["address"]) <= cursor
                    < int(segment["address"]) + int(segment["nbytes"])
                ),
                None,
            )
            if containing is None:
                raise RuntimeError(
                    f"Mutable runtime region is not packaged: {name} "
                    f"at 0x{cursor:016X}"
                )
            segment_start = int(containing["address"])
            available = (
                segment_start + int(containing["nbytes"]) - cursor
            )
            take = min(remaining, available)
            segment_path = runtime_dir / containing["file"]
            with segment_path.open("rb") as handle:
                handle.seek(cursor - segment_start)
                unread = take
                while unread:
                    chunk = handle.read(min(unread, 1024 * 1024))
                    if len(chunk) == 0:
                        raise RuntimeError(
                            f"Unexpected EOF while checking mutable region {name}"
                        )
                    if any(chunk):
                        raise RuntimeError(
                            f"Mutable runtime region is not zero-initialized: "
                            f"{name} at 0x{cursor:016X}"
                        )
                    unread -= len(chunk)
            cursor += take
            remaining -= take

    for region_name, region_address, region_nbytes in mutable_regions:
        require_zero_region(region_name, region_address, region_nbytes)

    return {
        "segment_count": len(segments),
        "total_segment_bytes": total_bytes,
        "first_address": f"0x{int(segments[0]['address']):016X}",
        "last_end_exclusive": (
            f"0x{int(segments[-1]['address']) + int(segments[-1]['nbytes']):016X}"
        ),
        "manifest_sha256": sha256_file(manifest_path),
        "loader": loader_audit,
        "zero_initialized_mutable_region_count": len(mutable_regions),
    }


def validate_model_config_against_runtime(
    runtime_dir: Path,
    model_config_header: Path,
) -> dict[str, Any]:
    segment_manifest_path = require_file(
        runtime_dir / "pl_ddr_binary_segments.json",
        "runtime segment manifest",
    )
    segment_manifest = json.loads(
        segment_manifest_path.read_text(encoding="utf-8")
    )
    full_manifest_path = require_file(
        runtime_dir / segment_manifest["source_manifest"],
        "runtime full-chain manifest",
    )
    full_manifest = json.loads(full_manifest_path.read_text(encoding="utf-8"))
    header_path = require_file(model_config_header, "model config header")
    header = header_path.read_text(encoding="utf-8")

    full_manifest_sha256 = sha256_file(full_manifest_path)
    provenance_match = re.search(
        r"Source manifest SHA256:\s*([0-9a-fA-F]{64})",
        header,
    )
    if (
        provenance_match is None
        or provenance_match.group(1).lower() != full_manifest_sha256
    ):
        raise RuntimeError(
            "Model config header provenance does not match the runtime manifest"
        )

    layer_count_match = re.search(
        r"#define\s+QOT_MODEL_LAYER_COUNT\s+(\d+)u\b",
        header,
    )
    layer_count = int(full_manifest["layer_count"])
    if (
        layer_count_match is None
        or int(layer_count_match.group(1)) != layer_count
    ):
        raise RuntimeError("Model config layer count does not match runtime")

    def require_macro(name: str, expected: int) -> None:
        match = re.search(
            rf"#define\s+{re.escape(name)}\s+"
            r"UINT64_C\((0x[0-9a-fA-F]+)\)",
            header,
        )
        if match is None:
            raise RuntimeError(f"Model config header lacks {name}")
        actual = int(match.group(1), 16)
        if actual != expected:
            raise RuntimeError(
                f"Model config {name} mismatch: "
                f"0x{actual:016X} != 0x{expected:016X}"
            )

    embedding = full_manifest["embedding"]
    runtime_context = full_manifest["runtime_context"]
    address_formula = full_manifest["address_formula"]
    macro_values = {
        "QOT_MODEL_EMBED_WEIGHT_BASE": int(embedding["weight_base"]),
        "QOT_MODEL_EMBED_SCALE_BASE": int(embedding["scale_base"]),
        "QOT_MODEL_HIDDEN_A_BASE": int(runtime_context["hidden_a_base"]),
        "QOT_MODEL_HIDDEN_B_BASE": int(runtime_context["hidden_b_base"]),
        "QOT_MODEL_KV_CACHE_BASE": int(address_formula["kv_cache_base"]),
        "QOT_MODEL_FINAL_TAIL_QMAP_BASE": int(
            full_manifest["final_tail"]["qmap_base"]
        ),
        "QOT_MODEL_ROPE_COS_BASE": int(runtime_context["rope_cos_base"]),
        "QOT_MODEL_ROPE_SIN_BASE": int(runtime_context["rope_sin_base"]),
    }
    for name, expected in macro_values.items():
        require_macro(name, expected)

    array_match = re.search(
        r"qot_model_layer_qmap_bases"
        r"\[QOT_MODEL_LAYER_COUNT\]\s*=\s*\{(.*?)\n\};",
        header,
        flags=re.DOTALL,
    )
    if array_match is None:
        raise RuntimeError("Could not parse the model QMAP base table")
    actual_table = [
        int(value, 16)
        for value in re.findall(
            r"UINT64_C\((0x[0-9a-fA-F]+)\)",
            array_match.group(1),
        )
    ]
    table_keys = (
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
    )
    expected_table = [
        int(layer["qmap_bases"][key])
        for layer in full_manifest["layers"]
        for key in table_keys
    ]
    if actual_table != expected_table:
        mismatch_index = next(
            (
                index
                for index, (actual, expected) in enumerate(
                    zip(actual_table, expected_table)
                )
                if actual != expected
            ),
            min(len(actual_table), len(expected_table)),
        )
        raise RuntimeError(
            "Model QMAP base table does not match runtime manifest at "
            f"flattened index {mismatch_index}"
        )

    return {
        "full_manifest_sha256": full_manifest_sha256,
        "model_config_header_sha256": sha256_file(header_path),
        "layer_count": layer_count,
        "qmap_base_count": len(expected_table),
        "scalar_base_count": len(macro_values),
        "matches_runtime_manifest": True,
    }


def validate_implementation(reports_dir: Path) -> dict[str, Any]:
    timing = require_file(
        reports_dir / "post_route_timing_summary.rpt",
        "post-route timing report",
    ).read_text(encoding="utf-8", errors="replace")
    check_timing = require_file(
        reports_dir / "check_timing_verbose.rpt",
        "verbose check_timing report",
    ).read_text(encoding="utf-8", errors="replace")
    route = require_file(
        reports_dir / "post_route_status.rpt",
        "post-route routing report",
    ).read_text(encoding="utf-8", errors="replace")
    drc = require_file(
        reports_dir / "post_route_drc.rpt",
        "post-route DRC report",
    ).read_text(encoding="utf-8", errors="replace")
    methodology = require_file(
        reports_dir / "post_route_methodology.rpt",
        "post-route methodology report",
    ).read_text(encoding="utf-8", errors="replace")
    status = require_file(
        reports_dir / "impl_1_status.txt",
        "implementation status",
    ).read_text(encoding="utf-8", errors="replace")
    utilization = require_file(
        reports_dir / "post_route_utilization.rpt",
        "post-route utilization report",
    ).read_text(encoding="utf-8", errors="replace")

    if "All user specified timing constraints are met." not in timing:
        raise RuntimeError("Post-route timing is not closed")
    if "checking no_clock (0)" not in timing:
        raise RuntimeError("Post-route timing has register/latch pins without clocks")
    if "checking unconstrained_internal_endpoints (0)" not in timing:
        raise RuntimeError("Post-route timing has unconstrained internal endpoints")
    if "checking no_clock (0)" not in check_timing:
        raise RuntimeError("Verbose check_timing has pins without clocks")
    if "checking unconstrained_internal_endpoints (0)" not in check_timing:
        raise RuntimeError(
            "Verbose check_timing has unconstrained internal endpoints"
        )
    if (
        "checking no_output_delay (1)" not in check_timing
        or re.search(
            r"^C0_DDR4_0_reset_n\s*$",
            check_timing,
            flags=re.MULTILINE,
        )
        is None
    ):
        raise RuntimeError(
            "Verbose check_timing does not contain only the known DDR4 "
            "asynchronous-reset output-delay exception"
        )
    if "| Error " in drc or "| Critical Warning " in drc:
        raise RuntimeError("Post-route DRC contains an error or critical warning")
    if "| Error " in methodology or "| Critical Warning " in methodology:
        raise RuntimeError(
            "Post-route methodology contains an error or critical warning"
        )
    if "status=write_bitstream Complete!" not in status:
        raise RuntimeError("Implementation did not complete write_bitstream")

    timing_lines = timing.splitlines()
    timing_values: list[str] | None = None
    for line_index, line in enumerate(timing_lines):
        if "WNS(ns)" not in line:
            continue
        for candidate in timing_lines[line_index + 1 : line_index + 8]:
            values = candidate.split()
            if len(values) == 12 and re.fullmatch(r"-?\d+\.\d+", values[0]):
                timing_values = values
                break
        if timing_values is not None:
            break
    if timing_values is None:
        raise RuntimeError("Could not parse post-route timing summary values")

    def route_count(label: str) -> int:
        match = re.search(
            rf"{re.escape(label)}\.*\s*:\s*(\d+)\s*:",
            route,
        )
        if match is None:
            raise RuntimeError(f"Could not parse route count: {label}")
        return int(match.group(1))

    routable_nets = route_count("# of routable nets")
    fully_routed_nets = route_count("# of fully routed nets")
    routing_errors = route_count("# of nets with routing errors")
    if fully_routed_nets != routable_nets or routing_errors != 0:
        raise RuntimeError("Post-route report is not fully routed and error-free")

    def utilization_row(label: str) -> dict[str, float | int]:
        match = re.search(
            rf"^\|\s*{re.escape(label)}\s*\|\s*([\d.]+)\s*\|"
            rf"\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*([\d.]+)\s*\|"
            rf"\s*([\d.]+)\s*\|",
            utilization,
            re.MULTILINE,
        )
        if match is None:
            raise RuntimeError(f"Could not parse utilization row: {label}")
        used_text, available_text, percent_text = match.groups()
        used: float | int = (
            float(used_text) if "." in used_text else int(used_text)
        )
        available: float | int = (
            float(available_text)
            if "." in available_text
            else int(available_text)
        )
        return {
            "used": used,
            "available": available,
            "percent": float(percent_text),
        }

    def warning_summary(report: str) -> dict[str, int]:
        return {
            rule: int(count)
            for rule, count in re.findall(
                r"^\|\s*(\S+)\s*\|\s*Warning\s*\|.*\|\s*(\d+)\s*\|\s*$",
                report,
                flags=re.MULTILINE,
            )
        }

    drc_warnings = warning_summary(drc)
    methodology_warnings = warning_summary(methodology)
    allowed_drc_warning_rules = {
        "DPIP-2",
        "DPOP-3",
        "DPOP-4",
        "DPOR-2",
        "PDCN-1569",
        "REQP-1773",
        "RTSTAT-10",
    }
    allowed_methodology_warning_rules = {
        "DPIR-2",
        "LUTAR-1",
        "ULMTCS-1",
    }
    unexpected_drc = sorted(set(drc_warnings) - allowed_drc_warning_rules)
    unexpected_methodology = sorted(
        set(methodology_warnings) - allowed_methodology_warning_rules
    )
    if unexpected_drc or unexpected_methodology:
        raise RuntimeError(
            "Implementation contains unreviewed warning rules: "
            f"DRC={unexpected_drc}, methodology={unexpected_methodology}"
        )

    return {
        "timing_constraints_met": True,
        "register_latch_pins_without_clock": 0,
        "unconstrained_internal_endpoints": 0,
        "known_async_output_delay_exception": "C0_DDR4_0_reset_n",
        "wns_ns": float(timing_values[0]),
        "whs_ns": float(timing_values[4]),
        "wpws_ns": float(timing_values[8]),
        "routable_nets": routable_nets,
        "fully_routed_nets": fully_routed_nets,
        "routing_errors": routing_errors,
        "drc_errors": 0,
        "drc_critical_warnings": 0,
        "drc_warning_rules": drc_warnings,
        "methodology_errors": 0,
        "methodology_critical_warnings": 0,
        "methodology_warning_rules": methodology_warnings,
        "utilization": {
            "clb_luts": utilization_row("CLB LUTs"),
            "clb_registers": utilization_row("CLB Registers"),
            "block_ram_tiles": utilization_row("Block RAM Tile"),
            "dsps": utilization_row("DSPs"),
        },
    }


def validate_regression(regression_dir: Path) -> dict[str, Any]:
    summary_path = require_file(
        regression_dir / "summary.txt",
        "persistent regression summary",
    )
    audit_path = require_file(
        regression_dir / "xsim" / "persistent_timing_audit.json",
        "persistent timing audit",
    )
    summary = summary_path.read_text(encoding="utf-8", errors="replace")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    steps = audit.get("steps", [])

    if audit.get("status") != "PASS":
        raise RuntimeError("Persistent timing audit did not PASS")
    if audit.get("layer_count") != 28:
        raise RuntimeError("Persistent regression is not the full 28-layer run")
    if len(steps) != 2:
        raise RuntimeError("Persistent regression is not a two-token run")
    tokens = [int(step["output_token"]) for step in steps]
    scores = [int(step["output_score_q26"]) for step in steps]
    if tokens != EXPECTED_OUTPUT_TOKENS:
        raise RuntimeError(f"Unexpected persistent outputs: {tokens}")
    if scores != EXPECTED_OUTPUT_SCORES_Q26:
        raise RuntimeError(f"Unexpected persistent scores: {scores}")
    if "PASS" not in summary:
        raise RuntimeError("Persistent regression summary has no PASS marker")

    return {
        "status": "PASS",
        "layer_count": 28,
        "token_count": 2,
        "input_tokens": [374, 28458],
        "output_tokens": tokens,
        "output_scores_q26": scores,
        "audit_sha256": sha256_file(audit_path),
        "summary_sha256": sha256_file(summary_path),
    }


def validate_vitis_workspace(
    workspace: Path,
    build_log_path: Path,
    release_xsa: Path,
    *,
    fsbl: Path,
    control_elf: Path,
    model_elf: Path,
    generate_elf: Path,
) -> dict[str, Any]:
    durable_source_dir = Path(__file__).resolve().parent
    control_launch_path = require_file(
        workspace / "a_qctl" / "_ide" / "launch.json",
        "control Vitis launch",
    )
    model_launch_path = require_file(
        workspace / "a_qmdl" / "_ide" / "launch.json",
        "model Vitis launch",
    )
    generate_launch_path = require_file(
        workspace / "a_qgen" / "_ide" / "launch.json",
        "generation Vitis launch",
    )
    model_config_path = require_file(
        workspace / "a_qmdl" / "src" / "UserConfig.cmake",
        "model Vitis compile configuration",
    )
    generate_config_path = require_file(
        workspace / "a_qgen" / "src" / "UserConfig.cmake",
        "generation Vitis compile configuration",
    )
    xparameters_path = require_file(
        workspace
        / "p_qot"
        / "export"
        / "p_qot"
        / "sw"
        / "standalone_psu_cortexa53_0"
        / "include"
        / "xparameters.h",
        "Vitis xparameters.h",
    )
    build_log_path = require_file(build_log_path, "Vitis build log")
    platform_xsa = require_file(
        workspace
        / "p_qot"
        / "export"
        / "p_qot"
        / "hw"
        / release_xsa.name,
        "Vitis platform XSA",
    )
    release_xsa_sha256 = sha256_file(release_xsa)
    if sha256_file(platform_xsa) != release_xsa_sha256:
        raise RuntimeError("Vitis platform was not built from the release XSA")

    def inspect_launch(path: Path) -> tuple[bool, list[bool]]:
        launch = json.loads(path.read_text(encoding="utf-8"))
        configurations = launch.get("configurations", [])
        if not configurations:
            raise RuntimeError(f"No configuration in {path}")
        run_psu_init_values = []
        stop_at_entry_values = []
        for configuration in configurations:
            setup = configuration["targetSetup"]
            run_psu_init_values.append(
                bool(setup["zuInitialization"]["usingPsuInit"]["runPsuInit"])
            )
            stop_at_entry_values.extend(
                bool(download["stopAtEntry"])
                for download in setup.get("downloadElf", [])
            )
        return any(run_psu_init_values), stop_at_entry_values

    control_run_psu_init, control_stop_at_entry = inspect_launch(
        control_launch_path
    )
    model_run_psu_init, model_stop_at_entry = inspect_launch(model_launch_path)
    generate_run_psu_init, generate_stop_at_entry = inspect_launch(
        generate_launch_path
    )
    if control_run_psu_init or model_run_psu_init or generate_run_psu_init:
        raise RuntimeError("Vitis hardware launch must keep runPsuInit=false")
    if any(control_stop_at_entry):
        raise RuntimeError("Control smoke launch must run immediately")
    if not model_stop_at_entry or not all(model_stop_at_entry):
        raise RuntimeError("Model smoke launch must stop at entry before loading DDR")
    if not generate_stop_at_entry or not all(generate_stop_at_entry):
        raise RuntimeError(
            "Generation launch must stop at entry before loading PL DDR"
        )

    model_config = model_config_path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    if "QOT_MODEL_BOARD_SMOKE=1" not in model_config:
        raise RuntimeError("Model Vitis app lacks QOT_MODEL_BOARD_SMOKE=1")

    generate_config = generate_config_path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    source_blocks = re.findall(
        r"^set\(USER_COMPILE_SOURCES[ \t]*\r?\n(.*?)^\)",
        generate_config,
        flags=re.MULTILINE | re.DOTALL,
    )
    if len(source_blocks) != 1:
        raise RuntimeError(
            "Generation Vitis app must contain exactly one "
            "USER_COMPILE_SOURCES block"
        )
    expected_generate_compile_sources = [
        "main.c",
        "qot_session.c",
        "qot_protocol.c",
        "qot_uart.c",
        "qtk_tokenizer_runtime.c",
        "qtk_text_tokenizer.c",
        "tokenizer_asset.S",
    ]
    generate_compile_lines = [
        line.strip() for line in source_blocks[0].splitlines() if line.strip()
    ]
    expected_generate_compile_lines = [
        f'"{name}"' for name in expected_generate_compile_sources
    ]
    if generate_compile_lines != expected_generate_compile_lines:
        raise RuntimeError(
            "Generation Vitis compile sources mismatch: "
            f"expected {expected_generate_compile_sources}, "
            f"got {generate_compile_lines}"
        )
    generate_compile_sources = expected_generate_compile_sources

    xparameters = xparameters_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).lower()
    required_addresses = {
        "control_base": "xpar_qmap_one_token_axi_bd_0_baseaddr 0xa0040000",
        "control_high": "xpar_qmap_one_token_axi_bd_0_highaddr 0xa004ffff",
        "ddr_status": "xpar_axi_gpio_0_baseaddr 0xa0010000",
        "pl_ddr_base": "xpar_ddr4_0_baseaddress 0x400000000",
        "pl_ddr_high": "xpar_ddr4_0_highaddress 0x41fffffff",
    }
    for label, expected in required_addresses.items():
        if expected not in xparameters:
            raise RuntimeError(
                f"Vitis xparameters.h lacks expected {label}: {expected}"
            )

    build_log = build_log_path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    for marker in (
        "platform_build=0",
        "control_app_build=0",
        "model_app_build=0",
        "generate_app_build=0",
        "-DQOT_MODEL_BOARD_SMOKE=1",
    ):
        if marker not in build_log:
            raise RuntimeError(f"Vitis build log lacks success marker: {marker}")

    source_hashes = {}
    for name in (
        "main.c",
        "qmap_one_token_runtime.h",
        "qmap_one_token_regs.h",
        "qmap_model_config_generated.h",
    ):
        durable = require_file(
            durable_source_dir / name,
            f"durable runtime source {name}",
        )
        durable_sha256 = sha256_file(durable)
        source_hashes[name] = durable_sha256
        for application in ("a_qctl", "a_qmdl"):
            copied = require_file(
                workspace / application / "src" / name,
                f"{application} source {name}",
            )
            if sha256_file(copied) != durable_sha256:
                raise RuntimeError(
                    f"{application} was not built from the current {name}"
                )

    generate_source_hashes = {}
    common_generate_sources = (
        "qmap_one_token_runtime.h",
        "qmap_one_token_regs.h",
        "qmap_model_config_generated.h",
    )
    for name in common_generate_sources:
        durable = require_file(
            durable_source_dir / name,
            f"durable generation runtime source {name}",
        )
        copied = require_file(
            workspace / "a_qgen" / "src" / name,
            f"a_qgen source {name}",
        )
        durable_sha256 = sha256_file(durable)
        if sha256_file(copied) != durable_sha256:
            raise RuntimeError(f"a_qgen was not built from the current {name}")
        generate_source_hashes[name] = durable_sha256

    demo_source_dir = durable_source_dir.parent / "qmap_prompt_demo"
    demo_source_map = {
        "main_generate.c": "main.c",
        "qot_session.c": "qot_session.c",
        "qot_session.h": "qot_session.h",
        "qot_protocol.c": "qot_protocol.c",
        "qot_protocol.h": "qot_protocol.h",
        "qot_uart.c": "qot_uart.c",
        "qot_uart.h": "qot_uart.h",
    }
    for durable_name, copied_name in demo_source_map.items():
        durable = require_file(
            demo_source_dir / durable_name,
            f"durable prompt-demo source {durable_name}",
        )
        copied = require_file(
            workspace / "a_qgen" / "src" / copied_name,
            f"a_qgen source {copied_name}",
        )
        durable_sha256 = sha256_file(durable)
        if sha256_file(copied) != durable_sha256:
            raise RuntimeError(
                f"a_qgen was not built from the current {durable_name}"
            )
        generate_source_hashes[durable_name] = durable_sha256

    tokenizer_runtime_dir = demo_source_dir / "tokenizer_runtime"
    for name in (
        "qtk_tokenizer_runtime.c",
        "qtk_tokenizer_runtime.h",
        "qtk_text_tokenizer.c",
        "qtk_text_tokenizer.h",
    ):
        durable = require_file(
            tokenizer_runtime_dir / name,
            f"durable tokenizer runtime source {name}",
        )
        copied = require_file(
            workspace / "a_qgen" / "src" / name,
            f"a_qgen tokenizer source {name}",
        )
        durable_sha256 = sha256_file(durable)
        if sha256_file(copied) != durable_sha256:
            raise RuntimeError(f"a_qgen was not built from the current {name}")
        generate_source_hashes[name] = durable_sha256

    tokenizer_asset_path = require_file(
        workspace / "a_qgen" / "src" / "qwen3_tokenizer.qtk",
        "a_qgen tokenizer asset",
    )
    tokenizer_asset_sha256 = sha256_file(tokenizer_asset_path)
    if (
        tokenizer_asset_path.stat().st_size != EXPECTED_TOKENIZER_ASSET_BYTES
        or tokenizer_asset_sha256 != EXPECTED_TOKENIZER_ASSET_SHA256
    ):
        raise RuntimeError(
            "a_qgen tokenizer asset does not match the released Qwen3 asset"
        )
    generate_source_hashes["qwen3_tokenizer.qtk"] = tokenizer_asset_sha256

    tokenizer_assembly_path = require_file(
        workspace / "a_qgen" / "src" / "tokenizer_asset.S",
        "a_qgen tokenizer assembly wrapper",
    )
    tokenizer_assembly = tokenizer_assembly_path.read_text(encoding="utf-8")
    expected_incbin = f'.incbin "{tokenizer_asset_path.as_posix()}"'
    required_assembly_markers = (
        '.section .rodata.qot_tokenizer_asset,"a",%progbits',
        ".global qot_tokenizer_asset_start",
        expected_incbin,
        ".global qot_tokenizer_asset_end",
    )
    for marker in required_assembly_markers:
        if tokenizer_assembly.count(marker) != 1:
            raise RuntimeError(
                "a_qgen tokenizer assembly contract mismatch: "
                f"expected one {marker!r}"
            )
    tokenizer_assembly_sha256 = sha256_file(tokenizer_assembly_path)
    generate_source_hashes["tokenizer_asset.S"] = tokenizer_assembly_sha256

    provided_artifacts = {
        "fsbl": fsbl,
        "control": control_elf,
        "model": model_elf,
        "generate": generate_elf,
    }
    workspace_artifact_paths = {
        "fsbl": (
            workspace
            / "p_qot"
            / "export"
            / "p_qot"
            / "sw"
            / "boot"
            / "fsbl.elf"
        ),
        "control": workspace / "a_qctl" / "build" / "a_qctl.elf",
        "model": workspace / "a_qmdl" / "build" / "a_qmdl.elf",
        "generate": workspace / "a_qgen" / "build" / "a_qgen.elf",
    }
    software_build_artifacts = {}
    for name, provided_path in provided_artifacts.items():
        workspace_path = require_file(
            workspace_artifact_paths[name],
            f"Vitis {name} build artifact",
        )
        workspace_sha256 = sha256_file(workspace_path)
        provided_sha256 = sha256_file(provided_path)
        if provided_sha256 != workspace_sha256:
            raise RuntimeError(
                f"Provided {name} ELF does not match the audited Vitis build: "
                f"provided={provided_path}, workspace={workspace_path}"
            )
        software_build_artifacts[name] = {
            "workspace_relative_path": workspace_path.relative_to(workspace).as_posix(),
            "nbytes": workspace_path.stat().st_size,
            "sha256": workspace_sha256,
        }

    return {
        "platform": "p_qot",
        "control_app": "a_qctl",
        "model_app": "a_qmdl",
        "generate_app": "a_qgen",
        "control_run_psu_init": False,
        "control_stop_at_entry": False,
        "model_run_psu_init": False,
        "model_stop_at_entry": True,
        "generate_run_psu_init": False,
        "generate_stop_at_entry": True,
        "model_compile_definition": "QOT_MODEL_BOARD_SMOKE=1",
        "control_base": "0xA0040000",
        "ddr_status_base": "0xA0010000",
        "pl_ddr_aperture": "0x400000000..0x41FFFFFFFF",
        "xparameters_sha256": sha256_file(xparameters_path),
        "build_log_sha256": sha256_file(build_log_path),
        "release_xsa_sha256": release_xsa_sha256,
        "runtime_source_sha256": source_hashes,
        "generate_source_sha256": generate_source_hashes,
        "generate_user_config_sha256": sha256_file(generate_config_path),
        "generate_compile_sources": generate_compile_sources,
        "tokenizer_asset": {
            "workspace_relative_path": tokenizer_asset_path.relative_to(
                workspace
            ).as_posix(),
            "nbytes": tokenizer_asset_path.stat().st_size,
            "sha256": tokenizer_asset_sha256,
            "assembly_workspace_relative_path": tokenizer_assembly_path.relative_to(
                workspace
            ).as_posix(),
            "assembly_sha256": tokenizer_assembly_sha256,
        },
        "software_build_artifacts": software_build_artifacts,
    }


def build_readme(runtime: dict[str, Any]) -> str:
    runtime_mib = runtime["total_segment_bytes"] / (1024 * 1024)
    return f"""# Qwen3-0.6B full28 FPGA board-test package

State: **ready to start hardware validation; no full28 hardware PASS is claimed yet.**

This package contains the routed bitstream/XSA, FSBL, AXI-Lite control smoke,
full28 persistent two-token model smoke, the complete Q4 PL-DDR runtime image,
and the local implementation/simulation evidence used to release it.

Before connecting the board, verify the package inventory:

```powershell
conda run -n llm_fpga python .\\verify_board_release.py
```

## Board preparation

1. Set the board boot mode to JTAG, connect JTAG and the UART, and power the
   board with the PL DDR4 hardware installed.
2. Open the UART at **115200 baud, 8 data bits, no parity, 1 stop bit**.
3. Start the AMD hardware server on `tcp:127.0.0.1:3121`. Vitis can start it,
   or run `hw_server.bat -s tcp::3121` from the Vivado/Vitis installation.

## Test order

From PowerShell in this package directory:

```powershell
.\\run_board_smoke.ps1 -Mode control
```

The UART must end with `PASS qot_run_no_memory_validation_smoke`.

Then run:

```powershell
.\\run_board_smoke.ps1 -Mode model
```

Model mode initializes the PS with FSBL, waits for PL DDR4 calibration, loads
{runtime["segment_count"]} verified binary segments ({runtime_mib:.2f} MiB),
checks all 281 QMAP packet headers, and starts the full28 persistent two-token
app.
The JTAG data load is large and can take several minutes.
Every segment preload deliberately uses `dow -data -bypass-cache-sync`; plain
`dow -data` can fail while XSDB attempts to synchronize the reset A53 caches.

The required UART result is:

```text
PASS token position=0 output=28458 score=1227344433
PASS token position=1 output=64 score=1015661901
PASS Qwen3-0.6B full28 persistent two-token board smoke
```

The launcher defaults `QOT_DEVICE_FILTER` to `name =~ "PL"`. If multiple PL
targets are connected, override it with an XSDB filter that selects exactly the
intended FPGA. To use a remote hardware server, set `QOT_HW_SERVER_URL`.

After the fixed model smoke passes, start the interactive token-ID application:

```powershell
.\run_board_smoke.ps1 -Mode generate
```

At the UART prompt, enter `HELP`. The first machine-facing request format is
`TOKENS <max_new> <count> <token_id_0> ...`. Each request starts a fresh,
bounded session and reports `TOKEN` records followed by a `DONE` reason.

`board_readiness_audit.json` records the release gates. `package_manifest.json`
contains the size and SHA256 of every packaged file.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble and audit the Qwen3 full28 FPGA board release"
    )
    parser.add_argument("--bit", required=True, type=Path)
    parser.add_argument("--xsa", required=True, type=Path)
    parser.add_argument("--fsbl", required=True, type=Path)
    parser.add_argument("--control-elf", required=True, type=Path)
    parser.add_argument("--model-elf", required=True, type=Path)
    parser.add_argument("--generate-elf", required=True, type=Path)
    parser.add_argument("--runtime-dir", required=True, type=Path)
    parser.add_argument("--reports-dir", required=True, type=Path)
    parser.add_argument("--regression-dir", required=True, type=Path)
    parser.add_argument("--vitis-workspace", required=True, type=Path)
    parser.add_argument("--vitis-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--copy-runtime",
        action="store_true",
        help="Copy runtime binaries instead of using same-volume hardlinks",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]
    output = args.output.resolve()

    bit = require_file(args.bit, "bitstream")
    xsa = require_file(args.xsa, "hardware platform")
    fsbl = require_file(args.fsbl, "FSBL")
    control_elf = require_file(args.control_elf, "control smoke ELF")
    model_elf = require_file(args.model_elf, "model smoke ELF")
    generate_elf = require_file(args.generate_elf, "interactive generation ELF")
    runtime_dir = args.runtime_dir.resolve()
    reports_dir = args.reports_dir.resolve()
    regression_dir = args.regression_dir.resolve()
    vitis_workspace = args.vitis_workspace.resolve()
    vitis_log = args.vitis_log.resolve()

    validate_output_location(
        output,
        repo_root,
        [
            script_dir,
            repo_root / ".git",
            repo_root / "Source",
            repo_root / "FPGA_Project",
            repo_root / "init",
            repo_root / "Qwen3-0.6B-Base",
            bit,
            xsa,
            fsbl,
            control_elf,
            model_elf,
            generate_elf,
            runtime_dir,
            reports_dir,
            regression_dir,
            vitis_workspace,
            vitis_log,
        ],
    )
    if output.exists():
        if not args.force:
            raise FileExistsError(
                f"Output exists; pass --force to replace it: {output}"
            )
        require_replaceable_release(output)

    runtime_audit = validate_runtime_image(runtime_dir)
    launcher_path = script_dir / "launch_qwen3_board.tcl"
    launcher_audit = validate_board_launcher(launcher_path)
    model_config_audit = validate_model_config_against_runtime(
        runtime_dir,
        script_dir / "qmap_model_config_generated.h",
    )
    implementation_audit = validate_implementation(reports_dir)
    regression_audit = validate_regression(regression_dir)
    vitis_audit = validate_vitis_workspace(
        vitis_workspace,
        vitis_log,
        xsa,
        fsbl=fsbl,
        control_elf=control_elf,
        model_elf=model_elf,
        generate_elf=generate_elf,
    )
    hardware_export_audit = validate_xsa_bitstream(xsa, bit)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging-",
            dir=output.parent,
        )
    )
    promoted = False
    try:
        copy_file(
            bit,
            staging / "hw" / "llm_system_qwen3_one_token_boardready.bit",
        )
        copy_file(
            xsa,
            staging / "hw" / "llm_system_qwen3_one_token_boardready.xsa",
        )
        copy_file(fsbl, staging / "sw" / "fsbl.elf")
        copy_file(control_elf, staging / "sw" / "a_qctl.elf")
        copy_file(model_elf, staging / "sw" / "a_qmdl.elf")
        copy_file(generate_elf, staging / "sw" / "a_qgen.elf")
        copy_file(
            launcher_path,
            staging / "launch_qwen3_board.tcl",
        )
        copy_file(
            script_dir / "run_board_smoke.ps1",
            staging / "run_board_smoke.ps1",
        )
        copy_file(
            script_dir / "verify_board_release.py",
            staging / "verify_board_release.py",
        )

        runtime_output = staging / "runtime"
        for source in sorted(runtime_dir.iterdir()):
            if source.is_file():
                copy_file(
                    source,
                    runtime_output / source.name,
                    hardlink=not args.copy_runtime,
                )

        for name in REPORT_FILES:
            copy_file(
                require_file(
                    reports_dir / name,
                    f"implementation report {name}",
                ),
                staging / "reports" / name,
            )
        copy_file(
            vitis_log,
            staging / "reports" / "vitis_build_stdout.log",
        )

        for relative_source, destination_name in REGRESSION_FILES:
            copy_file(
                require_file(
                    regression_dir / relative_source,
                    f"regression evidence {relative_source}",
                ),
                staging / "regression" / destination_name,
            )

        software_audit = {}
        for name, source in {
            "fsbl": fsbl,
            "control": control_elf,
            "model": model_elf,
            "generate": generate_elf,
        }.items():
            software_audit[name] = {
                "nbytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }

        readiness_audit = {
            "format_version": PACKAGE_FORMAT_VERSION,
            "release_state": "BOARD_TEST_READY_NOT_YET_HARDWARE_VALIDATED",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "implementation": implementation_audit,
            "hardware_export": hardware_export_audit,
            "runtime_image": runtime_audit,
            "board_launcher": launcher_audit,
            "model_config": model_config_audit,
            "persistent_full28_regression": regression_audit,
            "vitis": vitis_audit,
            "software": software_audit,
            "runtime_copy_mode": "copy" if args.copy_runtime else "hardlink",
            "next_gate": (
                "Run control, model, then generate mode on physical hardware "
                "and preserve the UART logs."
            ),
        }
        (staging / "board_readiness_audit.json").write_text(
            json.dumps(readiness_audit, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "BOARD_TEST_README.md").write_text(
            build_readme(runtime_audit),
            encoding="utf-8",
        )

        files = []
        manifest_path = staging / "package_manifest.json"
        for path in sorted(staging.rglob("*")):
            if path.is_file() and path != manifest_path:
                files.append(
                    {
                        "path": path.relative_to(staging).as_posix(),
                        "nbytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        package_manifest = {
            "format_version": PACKAGE_FORMAT_VERSION,
            "name": PACKAGE_NAME,
            "release_state": readiness_audit["release_state"],
            "generated_utc": readiness_audit["generated_utc"],
            "file_count": len(files),
            "total_file_bytes": sum(item["nbytes"] for item in files),
            "files": files,
        }
        manifest_path.write_text(
            json.dumps(package_manifest, indent=2) + "\n",
            encoding="utf-8",
        )

        for item in files:
            copied = staging / item["path"]
            if (
                copied.stat().st_size != item["nbytes"]
                or sha256_file(copied) != item["sha256"]
            ):
                raise RuntimeError(
                    f"Staged package verification failed: {item['path']}"
                )

        promote_staged_release(staging, output, replace=args.force)
        promoted = True
    finally:
        if not promoted and staging.exists():
            shutil.rmtree(staging)

    print(f"board_release={output}")
    print(f"file_count={package_manifest['file_count']}")
    print(f"total_file_bytes={package_manifest['total_file_bytes']}")
    print(f"runtime_segments={runtime_audit['segment_count']}")
    print("release_state=BOARD_TEST_READY_NOT_YET_HARDWARE_VALIDATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
