from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOAD_PLAN = (
    REPO_ROOT
    / "Temp"
    / "embedding_full28_final_chain_vectors"
    / "pl_ddr_runtime_load_plan.json"
)

ZERO_CHUNK = bytes(1024 * 1024)
WORDS_PER_CHUNK = (1024 * 1024) // 4


@dataclass(frozen=True)
class PlanEntry:
    operation: str
    name: str
    address: int
    nbytes: int
    encoding: str | None
    path: Path | None
    sha256: str | None

    @property
    def end(self) -> int:
        return self.address + self.nbytes


@dataclass(frozen=True)
class Segment:
    index: int
    address: int
    end: int
    entries: tuple[PlanEntry, ...]

    @property
    def nbytes(self) -> int:
        return self.end - self.address


def parse_size(value: str) -> int:
    text = value.strip().lower()
    multipliers = {
        "k": 1024,
        "kb": 1024,
        "m": 1024 * 1024,
        "mb": 1024 * 1024,
        "g": 1024 * 1024 * 1024,
        "gb": 1024 * 1024 * 1024,
    }
    for suffix, multiplier in sorted(
        multipliers.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if text.endswith(suffix):
            number = text[: -len(suffix)].strip()
            return int(number, 0) * multiplier
    return int(text, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a verified QMAP PL-DDR JSON load plan into compact binary "
            "segments and an XSDB data-load script."
        )
    )
    parser.add_argument("--load-plan", type=Path, default=DEFAULT_LOAD_PLAN)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--merge-gap-bytes", type=parse_size, default=64 * 1024)
    parser.add_argument("--max-segment-bytes", type=parse_size, default=128 * 1024 * 1024)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_entries(plan_path: Path) -> tuple[dict[str, Any], list[PlanEntry]]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("format_version") != 1:
        raise ValueError("only load-plan format_version=1 is supported")
    if plan.get("path_base") != "load_plan_directory":
        raise ValueError("load plan must use path_base=load_plan_directory")

    raw_entries = plan.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("load plan contains no entries")

    entries: list[PlanEntry] = []
    for raw in raw_entries:
        operation = str(raw.get("operation", ""))
        name = str(raw.get("name", ""))
        address = int(raw.get("address", -1))
        nbytes = int(raw.get("nbytes", -1))
        if operation not in {"file", "zero"}:
            raise ValueError(f"{name}: unsupported operation {operation!r}")
        if not name or address < 0 or nbytes <= 0:
            raise ValueError(f"invalid load-plan entry: {raw!r}")

        if operation == "file":
            encoding = str(raw.get("encoding", ""))
            if encoding not in {"words32_hex_le", "binary"}:
                raise ValueError(f"{name}: unsupported encoding {encoding!r}")
            relative_path = Path(str(raw.get("path", "")))
            if relative_path.is_absolute():
                raise ValueError(f"{name}: file path must be relative to the load plan")
            path = (plan_path.parent / relative_path).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            expected_sha256 = str(raw.get("sha256", "")).lower()
            if len(expected_sha256) != 64:
                raise ValueError(f"{name}: missing or invalid SHA256")
        else:
            encoding = None
            path = None
            expected_sha256 = None

        entries.append(
            PlanEntry(
                operation=operation,
                name=name,
                address=address,
                nbytes=nbytes,
                encoding=encoding,
                path=path,
                sha256=expected_sha256,
            )
        )

    entries.sort(key=lambda entry: (entry.address, entry.end, entry.name))
    previous: PlanEntry | None = None
    for entry in entries:
        if previous is not None and entry.address < previous.end:
            raise ValueError(
                f"overlapping load-plan entries: {previous.name} and {entry.name}"
            )
        previous = entry

    if len(entries) != int(plan.get("entry_count", -1)):
        raise ValueError("entry_count does not match entries")
    if sum(entry.nbytes for entry in entries if entry.operation == "file") != int(
        plan.get("total_file_bytes", -1)
    ):
        raise ValueError("total_file_bytes does not match entries")
    if sum(entry.nbytes for entry in entries if entry.operation == "zero") != int(
        plan.get("total_zero_bytes", -1)
    ):
        raise ValueError("total_zero_bytes does not match entries")
    return plan, entries


def copy_verified_source_manifest(
    plan: dict[str, Any], plan_path: Path, output_dir: Path
) -> str:
    manifest_text = str(plan.get("source_manifest", "")).strip()
    expected_sha256 = str(plan.get("source_manifest_sha256", "")).lower()
    if not manifest_text:
        raise ValueError("load plan is missing source_manifest")
    if len(expected_sha256) != 64:
        raise ValueError("load plan is missing a valid source_manifest_sha256")

    relative_path = Path(manifest_text)
    if relative_path.is_absolute():
        raise ValueError("source_manifest must be relative to the load plan")
    source_path = (plan_path.parent / relative_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    actual_sha256 = sha256_file(source_path)
    if actual_sha256.lower() != expected_sha256:
        raise ValueError(
            f"{source_path}: SHA256 mismatch, expected {expected_sha256}, "
            f"got {actual_sha256}"
        )

    destination_path = (output_dir / relative_path).resolve()
    try:
        destination_path.relative_to(output_dir)
    except ValueError as exc:
        raise ValueError("source_manifest escapes the output directory") from exc
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path != destination_path:
        shutil.copy2(source_path, destination_path)
    return relative_path.as_posix()


def group_segments(
    entries: Iterable[PlanEntry], merge_gap_bytes: int, max_segment_bytes: int
) -> list[Segment]:
    if merge_gap_bytes < 0:
        raise ValueError("--merge-gap-bytes must be nonnegative")
    if max_segment_bytes <= 0:
        raise ValueError("--max-segment-bytes must be positive")

    groups: list[list[PlanEntry]] = []
    for entry in entries:
        if not groups:
            groups.append([entry])
            continue
        group = groups[-1]
        group_start = group[0].address
        group_end = group[-1].end
        gap = entry.address - group_end
        proposed_size = entry.end - group_start
        if gap <= merge_gap_bytes and proposed_size <= max_segment_bytes:
            group.append(entry)
        else:
            groups.append([entry])

    return [
        Segment(
            index=index,
            address=group[0].address,
            end=group[-1].end,
            entries=tuple(group),
        )
        for index, group in enumerate(groups)
    ]


def write_zeros(output: BinaryIO, nbytes: int, digest: hashlib._Hash) -> None:
    remaining = nbytes
    while remaining:
        chunk = ZERO_CHUNK if remaining >= len(ZERO_CHUNK) else ZERO_CHUNK[:remaining]
        output.write(chunk)
        digest.update(chunk)
        remaining -= len(chunk)


def write_words32_hex(
    output: BinaryIO, path: Path, expected_nbytes: int, digest: hashlib._Hash
) -> int:
    written = 0
    values: list[int] = []

    def flush_values() -> None:
        nonlocal written
        if not values:
            return
        data = struct.pack(f"<{len(values)}I", *values)
        output.write(data)
        digest.update(data)
        written += len(data)
        values.clear()

    with path.open("rt", encoding="ascii", newline="") as source:
        for line_number, line in enumerate(source, start=1):
            text = line.strip()
            if not text:
                continue
            if len(text) != 8:
                raise ValueError(f"{path}:{line_number}: expected one 8-digit word")
            try:
                values.append(int(text, 16))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid hexadecimal word") from exc
            if len(values) >= WORDS_PER_CHUNK:
                flush_values()
    flush_values()
    if written != expected_nbytes:
        raise ValueError(f"{path}: decoded {written} bytes, expected {expected_nbytes}")
    return written


def write_binary(
    output: BinaryIO, path: Path, expected_nbytes: int, digest: hashlib._Hash
) -> int:
    written = 0
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
            written += len(chunk)
    if written != expected_nbytes:
        raise ValueError(f"{path}: copied {written} bytes, expected {expected_nbytes}")
    return written


def write_segment(segment: Segment, output_path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    cursor = segment.address
    entry_records: list[dict[str, Any]] = []
    with output_path.open("wb") as output:
        for entry in segment.entries:
            gap = entry.address - cursor
            if gap:
                write_zeros(output, gap, digest)
                cursor += gap

            if entry.operation == "zero":
                write_zeros(output, entry.nbytes, digest)
            else:
                assert entry.path is not None
                assert entry.sha256 is not None
                actual_source_sha256 = sha256_file(entry.path)
                if actual_source_sha256.lower() != entry.sha256:
                    raise ValueError(
                        f"{entry.path}: SHA256 mismatch, expected {entry.sha256}, "
                        f"got {actual_source_sha256}"
                    )
                if entry.encoding == "words32_hex_le":
                    write_words32_hex(output, entry.path, entry.nbytes, digest)
                elif entry.encoding == "binary":
                    write_binary(output, entry.path, entry.nbytes, digest)
                else:
                    raise AssertionError(entry.encoding)

            cursor = entry.end
            entry_records.append(
                {
                    "name": entry.name,
                    "operation": entry.operation,
                    "address": entry.address,
                    "nbytes": entry.nbytes,
                }
            )

    if cursor != segment.end or output_path.stat().st_size != segment.nbytes:
        raise AssertionError("segment byte count does not match planned address span")
    return {
        "index": segment.index,
        "file": output_path.name,
        "address": segment.address,
        "address_hex": f"0x{segment.address:016X}",
        "nbytes": segment.nbytes,
        "sha256": digest.hexdigest(),
        "entries": entry_records,
    }


def make_xsdb_loader(segment_records: list[dict[str, Any]]) -> str:
    lines = [
        "# Generated by 57_pack_qmap_runtime_load_plan.py.",
        "# Select the Cortex-A53 #0 target and initialize/program the board first.",
        "set script_dir [file dirname [file normalize [info script]]]",
        'puts "Loading Qwen3 Q4 runtime image into PL DDR4..."',
    ]
    for record in segment_records:
        lines.extend(
            [
                f'set segment_file [file join $script_dir "{record["file"]}"]',
                (
                    f'puts "  {record["file"]}: {record["nbytes"]} bytes '
                    f'at {record["address_hex"]}"'
                ),
                f'dow -data -bypass-cache-sync $segment_file {record["address_hex"]}',
            ]
        )
    lines.extend(
        [
            'puts "Qwen3 PL-DDR runtime image load complete."',
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    load_plan_path = args.load_plan.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plan, entries = load_entries(load_plan_path)
    source_manifest_copy = copy_verified_source_manifest(
        plan, load_plan_path, output_dir
    )
    segments = group_segments(entries, args.merge_gap_bytes, args.max_segment_bytes)

    records: list[dict[str, Any]] = []
    for segment in segments:
        output_path = output_dir / f"qwen3_runtime_{segment.index:02d}.bin"
        print(
            f"packing segment {segment.index}: "
            f"0x{segment.address:016X}..0x{segment.end - 1:016X} "
            f"({segment.nbytes} bytes)"
        )
        records.append(write_segment(segment, output_path))

    load_plan_copy = output_dir / "pl_ddr_runtime_load_plan.json"
    shutil.copy2(load_plan_path, load_plan_copy)
    manifest = {
        "format_version": 1,
        "name": "qwen3_0p6b_pl_ddr_binary_segments",
        "source_load_plan": load_plan_copy.name,
        "source_load_plan_sha256": sha256_file(load_plan_path),
        "source_manifest": source_manifest_copy,
        "source_manifest_sha256": plan.get("source_manifest_sha256"),
        "segment_count": len(records),
        "total_segment_bytes": sum(record["nbytes"] for record in records),
        "segments": records,
    }
    manifest_path = output_dir / "pl_ddr_binary_segments.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    loader_path = output_dir / "load_pl_ddr_runtime.tcl"
    loader_path.write_text(make_xsdb_loader(records), encoding="ascii")

    print(f"manifest={manifest_path}")
    print(f"source_manifest={output_dir / source_manifest_copy}")
    print(f"xsdb_loader={loader_path}")
    print(f"segments={len(records)}")
    print(f"segment_bytes={manifest['total_segment_bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
