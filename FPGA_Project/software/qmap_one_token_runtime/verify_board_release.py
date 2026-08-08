from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_PL_TARGET_ASSIGNMENT = 'set device_filter {name =~ "PL"}'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def verify_launcher_contract(package: Path) -> dict[str, object]:
    launcher_path = package / "launch_qwen3_board.tcl"
    launcher = launcher_path.read_text(encoding="utf-8")
    default_assignments = [
        line.rstrip()
        for line in launcher.splitlines()
        if line.startswith("set device_filter ")
    ]
    require_exact_sequence(
        default_assignments,
        [DEFAULT_PL_TARGET_ASSIGNMENT],
        "Packaged launcher default PL target selector",
    )
    selection = 'select_unique_target $device_filter "PL device"'
    selection_count = sum(
        line.rstrip() == selection for line in launcher.splitlines()
    )
    if selection_count != 1:
        raise RuntimeError(
            "Packaged launcher must select exactly one PL device through "
            "device_filter"
        )
    return {
        "default_device_filter": 'name =~ "PL"',
        "unique_pl_selection": True,
    }


def verify_runtime_loader_contract(package: Path) -> dict[str, object]:
    runtime_dir = package / "runtime"
    segment_manifest = json.loads(
        (runtime_dir / "pl_ddr_binary_segments.json").read_text(encoding="utf-8")
    )
    segments = segment_manifest.get("segments")
    if not isinstance(segments, list) or not segments:
        raise RuntimeError("Runtime segment manifest contains no segments")
    if segment_manifest.get("segment_count") != len(segments):
        raise RuntimeError("Runtime segment_count does not match segments")
    if len(segments) != 61:
        raise RuntimeError(
            f"Board release requires exactly 61 runtime segments, got {len(segments)}"
        )

    expected_operations: list[str] = []
    for index, segment in enumerate(segments):
        if segment.get("index") != index:
            raise RuntimeError(f"Runtime segment index mismatch at {index}")
        segment_path = runtime_dir / str(segment["file"])
        if not segment_path.is_file():
            raise FileNotFoundError(
                f"Packaged runtime segment is missing: {segment_path.name}"
            )
        expected_operations.extend(
            (
                'set segment_file [file join $script_dir '
                f'"{segment["file"]}"]',
                "dow -data -bypass-cache-sync $segment_file "
                f'0x{int(segment["address"]):016X}',
            )
        )

    loader = (runtime_dir / "load_pl_ddr_runtime.tcl").read_text(
        encoding="utf-8"
    )
    actual_operations = [
        line.rstrip()
        for line in loader.splitlines()
        if line.startswith("set segment_file ") or line.startswith("dow ")
    ]
    require_exact_sequence(
        actual_operations,
        expected_operations,
        "Packaged runtime XSDB loader",
    )
    return {
        "segment_download_count": len(segments),
        "download_command": "dow -data -bypass-cache-sync",
        "file_address_order_matches_manifest": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify every file in a packaged Qwen3 FPGA board release"
    )
    parser.add_argument(
        "package",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()
    package = args.package.resolve()
    manifest_path = package / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    expected_paths = set()
    for item in manifest["files"]:
        relative = Path(item["path"])
        expected_paths.add(relative.as_posix())
        path = package / relative
        if not path.is_file():
            raise FileNotFoundError(f"Packaged file is missing: {relative}")
        if path.stat().st_size != item["nbytes"]:
            raise RuntimeError(f"Packaged file size mismatch: {relative}")
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Packaged file SHA256 mismatch: {relative}")

    actual_paths = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file() and path.name != "package_manifest.json"
    }
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        raise RuntimeError(
            f"Package inventory mismatch; missing={missing}, extra={extra}"
        )

    audit = json.loads(
        (package / "board_readiness_audit.json").read_text(encoding="utf-8")
    )
    required_state = "BOARD_TEST_READY_NOT_YET_HARDWARE_VALIDATED"
    if audit.get("release_state") != required_state:
        raise RuntimeError(
            f"Unexpected board release state: {audit.get('release_state')}"
        )

    launcher_contract = verify_launcher_contract(package)
    loader_contract = verify_runtime_loader_contract(package)

    readme_path = package / "BOARD_TEST_README.md"
    readme = readme_path.read_text(encoding="utf-8")
    invalid_controls = [
        (index, ord(character))
        for index, character in enumerate(readme)
        if ord(character) < 32 and character not in "\t\n\r"
    ]
    if invalid_controls:
        raise RuntimeError(
            "BOARD_TEST_README.md contains disallowed control characters: "
            f"{invalid_controls[:8]}"
        )

    required_commands = (
        r"conda run -n llm_fpga python .\verify_board_release.py",
        r".\run_board_smoke.ps1 -Mode control",
        r".\run_board_smoke.ps1 -Mode model",
    )
    for command in required_commands:
        if command not in readme:
            raise RuntimeError(
                f"BOARD_TEST_README.md is missing command: {command}"
            )
    required_contract_markers = (
        '`name =~ "PL"`',
        "`dow -data -bypass-cache-sync`",
    )
    for marker in required_contract_markers:
        if marker not in readme:
            raise RuntimeError(
                f"BOARD_TEST_README.md is missing launcher contract: {marker}"
            )

    print(
        "PASS board release package: "
        f"{manifest['file_count']} files, {manifest['total_file_bytes']} bytes"
    )
    print(f"release_state={required_state}")
    print(
        "launcher_default_device_filter="
        f"{launcher_contract['default_device_filter']}"
    )
    print(
        "runtime_loader="
        f"{loader_contract['segment_download_count']} x "
        f"{loader_contract['download_command']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
