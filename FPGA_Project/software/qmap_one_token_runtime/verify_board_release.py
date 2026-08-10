from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath


DEFAULT_PL_TARGET_ASSIGNMENT = 'set device_filter {name =~ "PL"}'
PACKAGE_NAME = "qwen3_0p6b_full28_fpga_board_test"
SUPPORTED_FORMAT_VERSIONS = (1, 2)
RELEASE_STATE = "BOARD_TEST_READY_NOT_YET_HARDWARE_VALIDATED"
EXPECTED_TOKENIZER_ASSET_BYTES = 3_629_566
EXPECTED_TOKENIZER_ASSET_SHA256 = (
    "c20242603ef4144e3f3f2ec4ba97c0e9c315aadd41f1bd2c5740e2a7ffa03a7d"
)
MANIFEST_KEYS = {
    "format_version",
    "name",
    "release_state",
    "generated_utc",
    "file_count",
    "total_file_bytes",
    "files",
}
FILE_ENTRY_KEYS = {"path", "nbytes", "sha256"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"JSON object contains duplicate key: {key!r}")
        result[key] = value
    return result


def path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def require_integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RuntimeError(f"{label} must be an integer >= {minimum}")
    return value


def parse_manifest_relative_path(raw_path: object, label: str) -> PurePosixPath:
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError(f"{label} must be a non-empty POSIX path string")
    if "\\" in raw_path:
        raise RuntimeError(f"{label} must use forward slashes: {raw_path!r}")
    relative = PurePosixPath(raw_path)
    if (
        relative.is_absolute()
        or raw_path != relative.as_posix()
        or any(part in ("", ".", "..") or ":" in part for part in relative.parts)
    ):
        raise RuntimeError(f"{label} is not a canonical relative path: {raw_path!r}")
    return relative


def verify_manifest_inventory(package: Path) -> tuple[dict[str, object], set[str]]:
    manifest_path = package / "package_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Package manifest is missing: {manifest_path}")
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_json_keys,
    )
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
        actual_keys = sorted(manifest) if isinstance(manifest, dict) else []
        raise RuntimeError(
            "Package manifest schema mismatch: "
            f"expected={sorted(MANIFEST_KEYS)}, actual={actual_keys}"
        )

    format_version = require_integer(
        manifest["format_version"], "manifest format_version", minimum=1
    )
    if format_version not in SUPPORTED_FORMAT_VERSIONS:
        raise RuntimeError(
            f"Unsupported package manifest format_version: {format_version}"
        )
    if manifest["name"] != PACKAGE_NAME:
        raise RuntimeError(f"Unexpected package name: {manifest['name']!r}")
    if manifest["release_state"] != RELEASE_STATE:
        raise RuntimeError(
            f"Unexpected package release_state: {manifest['release_state']!r}"
        )
    if not isinstance(manifest["generated_utc"], str) or not manifest["generated_utc"]:
        raise RuntimeError("manifest generated_utc must be a non-empty string")
    declared_count = require_integer(manifest["file_count"], "manifest file_count")
    declared_total = require_integer(
        manifest["total_file_bytes"], "manifest total_file_bytes"
    )
    entries = manifest["files"]
    if not isinstance(entries, list):
        raise RuntimeError("manifest files must be a list")
    if declared_count != len(entries):
        raise RuntimeError(
            f"manifest file_count mismatch: {declared_count} != {len(entries)}"
        )

    expected_paths: set[str] = set()
    computed_total = 0
    package_resolved = package.resolve()
    for index, item in enumerate(entries):
        if not isinstance(item, dict) or set(item) != FILE_ENTRY_KEYS:
            actual_keys = sorted(item) if isinstance(item, dict) else []
            raise RuntimeError(
                f"manifest files[{index}] schema mismatch: "
                f"expected={sorted(FILE_ENTRY_KEYS)}, actual={actual_keys}"
            )
        relative = parse_manifest_relative_path(
            item["path"], f"manifest files[{index}].path"
        )
        relative_text = relative.as_posix()
        if relative_text == "package_manifest.json":
            raise RuntimeError("package_manifest.json must not inventory itself")
        if relative_text in expected_paths:
            raise RuntimeError(f"Duplicate manifest path: {relative_text}")
        expected_paths.add(relative_text)

        nbytes = require_integer(
            item["nbytes"], f"manifest files[{index}].nbytes"
        )
        expected_sha256 = item["sha256"]
        if not isinstance(expected_sha256, str) or re.fullmatch(
            r"[0-9a-f]{64}", expected_sha256
        ) is None:
            raise RuntimeError(
                f"manifest files[{index}].sha256 must be 64 lowercase hex digits"
            )

        lexical_path = package.joinpath(*relative.parts)
        if lexical_path.is_symlink():
            raise RuntimeError(f"Packaged file must not be a symlink: {relative_text}")
        path = lexical_path.resolve()
        if not path_is_within(path, package_resolved):
            raise RuntimeError(f"Packaged path escapes package root: {relative_text}")
        if not path.is_file():
            raise FileNotFoundError(f"Packaged file is missing: {relative_text}")
        actual_size = path.stat().st_size
        if actual_size != nbytes:
            raise RuntimeError(f"Packaged file size mismatch: {relative_text}")
        if sha256_file(path) != expected_sha256:
            raise RuntimeError(f"Packaged file SHA256 mismatch: {relative_text}")
        computed_total += actual_size

    if declared_total != computed_total:
        raise RuntimeError(
            f"manifest total_file_bytes mismatch: {declared_total} != {computed_total}"
        )

    symlinks = [
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_symlink()
    ]
    if symlinks:
        raise RuntimeError(f"Package contains symlinks: {sorted(symlinks)}")
    actual_paths = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        raise RuntimeError(
            f"Package inventory mismatch; missing={missing}, extra={extra}"
        )
    return manifest, expected_paths


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


def verify_launcher_contract(
    package: Path, format_version: int
) -> dict[str, object]:
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
    generate_mode = format_version >= 2
    if generate_mode:
        required_generate_contract = (
            "set generate_elf [file join $package_root sw a_qgen.elf]",
            'if {$mode ni {"model" "control" "generate"}} {',
            'require_file $generate_elf "interactive generation ELF"',
            "dow $generate_elf",
        )
        for marker in required_generate_contract:
            if marker not in launcher:
                raise RuntimeError(
                    f"Packaged launcher lacks generation contract: {marker}"
                )
        generate_elf = package / "sw" / "a_qgen.elf"
        if not generate_elf.is_file() or generate_elf.stat().st_size == 0:
            raise FileNotFoundError(
                "Packaged interactive generation ELF is missing or empty"
            )
        wrapper = (package / "run_board_smoke.ps1").read_text(encoding="utf-8")
        if 'ValidateSet("model", "control", "generate")' not in wrapper:
            raise RuntimeError(
                "Packaged PowerShell wrapper lacks generate mode"
            )
    return {
        "default_device_filter": 'name =~ "PL"',
        "unique_pl_selection": True,
        "generate_mode": generate_mode,
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
    total_segment_bytes = 0
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise RuntimeError(f"Runtime segment {index} must be an object")
        if segment.get("index") != index:
            raise RuntimeError(f"Runtime segment index mismatch at {index}")
        relative = parse_manifest_relative_path(
            segment.get("file"), f"runtime segment {index} file"
        )
        if len(relative.parts) != 1:
            raise RuntimeError(
                f"Runtime segment {index} must use a flat filename"
            )
        segment_path = runtime_dir / relative.name
        if not segment_path.is_file():
            raise FileNotFoundError(
                f"Packaged runtime segment is missing: {segment_path.name}"
            )
        expected_size = require_integer(
            segment.get("nbytes"), f"runtime segment {index} nbytes"
        )
        expected_sha256 = segment.get("sha256")
        if not isinstance(expected_sha256, str) or re.fullmatch(
            r"[0-9a-f]{64}", expected_sha256
        ) is None:
            raise RuntimeError(f"Runtime segment {index} SHA256 is malformed")
        if segment_path.stat().st_size != expected_size:
            raise RuntimeError(f"Runtime segment size mismatch: {relative.as_posix()}")
        if sha256_file(segment_path) != expected_sha256:
            raise RuntimeError(f"Runtime segment SHA256 mismatch: {relative.as_posix()}")
        total_segment_bytes += expected_size
        expected_operations.extend(
            (
                'set segment_file [file join $script_dir '
                f'"{relative.as_posix()}"]',
                "dow -data -bypass-cache-sync $segment_file "
                f'0x{int(segment["address"]):016X}',
            )
        )
    if segment_manifest.get("total_segment_bytes") != total_segment_bytes:
        raise RuntimeError("Runtime total_segment_bytes mismatch")

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


def verify_v2_software_audit(
    package: Path, audit: dict[str, object]
) -> dict[str, object]:
    if audit.get("format_version") != 2:
        raise RuntimeError("A format-v2 package requires a format-v2 readiness audit")
    software = audit.get("software")
    vitis = audit.get("vitis")
    if not isinstance(software, dict) or not isinstance(vitis, dict):
        raise RuntimeError("Format-v2 readiness audit lacks software/Vitis data")
    vitis_artifacts = vitis.get("software_build_artifacts")
    if not isinstance(vitis_artifacts, dict):
        raise RuntimeError("Format-v2 Vitis audit lacks software build artifacts")
    tokenizer_asset = vitis.get("tokenizer_asset")
    if not isinstance(tokenizer_asset, dict) or set(tokenizer_asset) != {
        "workspace_relative_path",
        "nbytes",
        "sha256",
        "assembly_workspace_relative_path",
        "assembly_sha256",
    }:
        raise RuntimeError("Format-v2 Vitis audit lacks tokenizer asset provenance")
    if (
        tokenizer_asset.get("nbytes") != EXPECTED_TOKENIZER_ASSET_BYTES
        or tokenizer_asset.get("sha256") != EXPECTED_TOKENIZER_ASSET_SHA256
    ):
        raise RuntimeError("Format-v2 Vitis audit has an unexpected tokenizer asset")
    parse_manifest_relative_path(
        tokenizer_asset["workspace_relative_path"],
        "Vitis tokenizer workspace_relative_path",
    )
    parse_manifest_relative_path(
        tokenizer_asset["assembly_workspace_relative_path"],
        "Vitis tokenizer assembly_workspace_relative_path",
    )
    if not isinstance(tokenizer_asset["assembly_sha256"], str) or re.fullmatch(
        r"[0-9a-f]{64}", tokenizer_asset["assembly_sha256"]
    ) is None:
        raise RuntimeError("Format-v2 tokenizer assembly SHA256 is malformed")

    expected_generate_compile_sources = [
        "main.c",
        "qot_session.c",
        "qot_protocol.c",
        "qot_uart.c",
        "qtk_tokenizer_runtime.c",
        "qtk_text_tokenizer.c",
        "tokenizer_asset.S",
    ]
    if vitis.get("generate_compile_sources") != expected_generate_compile_sources:
        raise RuntimeError("Format-v2 Vitis audit has unexpected generation sources")
    generate_source_hashes = vitis.get("generate_source_sha256")
    required_generate_hashes = {
        "qtk_tokenizer_runtime.c",
        "qtk_tokenizer_runtime.h",
        "qtk_text_tokenizer.c",
        "qtk_text_tokenizer.h",
        "qwen3_tokenizer.qtk",
        "tokenizer_asset.S",
    }
    if not isinstance(generate_source_hashes, dict) or not required_generate_hashes.issubset(
        generate_source_hashes
    ):
        raise RuntimeError("Format-v2 Vitis audit lacks tokenizer source hashes")
    for name in required_generate_hashes:
        value = generate_source_hashes[name]
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise RuntimeError(f"Format-v2 source SHA256 is malformed for {name}")
    if generate_source_hashes["qwen3_tokenizer.qtk"] != EXPECTED_TOKENIZER_ASSET_SHA256:
        raise RuntimeError("Format-v2 generated-source audit has wrong tokenizer asset")
    if generate_source_hashes["tokenizer_asset.S"] != tokenizer_asset[
        "assembly_sha256"
    ]:
        raise RuntimeError("Format-v2 tokenizer assembly hashes disagree")

    packaged_paths = {
        "fsbl": package / "sw" / "fsbl.elf",
        "control": package / "sw" / "a_qctl.elf",
        "model": package / "sw" / "a_qmdl.elf",
        "generate": package / "sw" / "a_qgen.elf",
    }
    if set(software) != set(packaged_paths):
        raise RuntimeError("Format-v2 software audit has an unexpected ELF set")
    if set(vitis_artifacts) != set(packaged_paths):
        raise RuntimeError("Format-v2 Vitis audit has an unexpected ELF set")
    verified = {}
    for name, path in packaged_paths.items():
        software_entry = software.get(name)
        vitis_entry = vitis_artifacts.get(name)
        if not isinstance(software_entry, dict) or not isinstance(vitis_entry, dict):
            raise RuntimeError(f"Format-v2 audit lacks {name} ELF provenance")
        if set(software_entry) != {"nbytes", "sha256"}:
            raise RuntimeError(f"Format-v2 software audit schema mismatch for {name}")
        if set(vitis_entry) != {
            "workspace_relative_path",
            "nbytes",
            "sha256",
        }:
            raise RuntimeError(f"Format-v2 Vitis audit schema mismatch for {name}")
        actual_size = path.stat().st_size
        actual_sha256 = sha256_file(path)
        for label, entry in (
            ("software", software_entry),
            ("Vitis", vitis_entry),
        ):
            if entry.get("nbytes") != actual_size:
                raise RuntimeError(
                    f"{label} audit size mismatch for packaged {name} ELF"
                )
            if entry.get("sha256") != actual_sha256:
                raise RuntimeError(
                    f"{label} audit SHA256 mismatch for packaged {name} ELF"
                )
        workspace_relative_path = vitis_entry.get("workspace_relative_path")
        if not isinstance(workspace_relative_path, str):
            raise RuntimeError(
                f"Vitis audit lacks workspace-relative path for {name} ELF"
            )
        parse_manifest_relative_path(
            workspace_relative_path,
            f"Vitis {name} workspace_relative_path",
        )
        verified[name] = {
            "nbytes": actual_size,
            "sha256": actual_sha256,
        }
    return verified


def verify_package(package: Path) -> dict[str, object]:
    package = package.resolve()
    if not package.is_dir():
        raise FileNotFoundError(f"Package directory is missing: {package}")
    manifest, expected_paths = verify_manifest_inventory(package)
    format_version = int(manifest["format_version"])

    required_paths = {
        "BOARD_TEST_README.md",
        "board_readiness_audit.json",
        "launch_qwen3_board.tcl",
        "run_board_smoke.ps1",
        "sw/fsbl.elf",
        "sw/a_qctl.elf",
        "sw/a_qmdl.elf",
    }
    if format_version >= 2:
        required_paths.add("sw/a_qgen.elf")
    missing_contract_paths = sorted(required_paths - expected_paths)
    if missing_contract_paths:
        raise RuntimeError(
            "Package manifest lacks required contract files: "
            f"{missing_contract_paths}"
        )

    audit = json.loads(
        (package / "board_readiness_audit.json").read_text(encoding="utf-8")
    )
    if not isinstance(audit, dict):
        raise RuntimeError("board_readiness_audit.json must contain an object")
    if audit.get("release_state") != RELEASE_STATE:
        raise RuntimeError(
            f"Unexpected board release state: {audit.get('release_state')}"
        )
    if audit.get("release_state") != manifest["release_state"]:
        raise RuntimeError("Manifest and readiness-audit release states differ")
    if audit.get("generated_utc") != manifest["generated_utc"]:
        raise RuntimeError("Manifest and readiness-audit timestamps differ")
    if format_version == 1:
        if audit.get("format_version") not in (None, 1):
            raise RuntimeError("Format-v1 package has an incompatible readiness audit")
        software_audit = None
    else:
        software_audit = verify_v2_software_audit(package, audit)

    launcher_contract = verify_launcher_contract(package, format_version)
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

    required_commands = [
        r"conda run -n llm_fpga python .\verify_board_release.py",
        r".\run_board_smoke.ps1 -Mode control",
        r".\run_board_smoke.ps1 -Mode model",
    ]
    if format_version >= 2:
        required_commands.append(r".\run_board_smoke.ps1 -Mode generate")
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

    return {
        "format_version": format_version,
        "file_count": manifest["file_count"],
        "total_file_bytes": manifest["total_file_bytes"],
        "release_state": RELEASE_STATE,
        "launcher": launcher_contract,
        "runtime_loader": loader_contract,
        "software": software_audit,
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
    result = verify_package(args.package)

    print(
        "PASS board release package: "
        f"format v{result['format_version']}, "
        f"{result['file_count']} files, {result['total_file_bytes']} bytes"
    )
    print(f"release_state={result['release_state']}")
    print(
        "launcher_default_device_filter="
        f"{result['launcher']['default_device_filter']}"
    )
    print(
        "runtime_loader="
        f"{result['runtime_loader']['segment_download_count']} x "
        f"{result['runtime_loader']['download_command']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
