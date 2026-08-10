from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

from make_prompt_demo_workbench import (
    EXPECTED_GENERATE_ELF_SHA256,
    LEGACY_LINEAGE_HASHES,
    LEGACY_LINEAGE_ID,
    HOST_ACCEPTANCE_CASE_COUNT,
    HOST_ACCEPTANCE_SOURCE_FILES,
    HOST_UI_SOURCE_FILES,
    PINNED_HOST_ACCEPTANCE_SHA256,
    PINNED_HOST_UI_SHA256,
    PINNED_RUNTIME_TOOL_SHA256,
    sha256_file,
    validate_runtime,
)


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SUPPORTED_FORMAT_VERSIONS = {2, 3, 4, 5}


def is_link_like(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def require_safe_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError("Manifest path must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError(f"Unsafe manifest path: {value!r}")
    if path.as_posix() != value or "\\" in value:
        raise RuntimeError(f"Non-canonical manifest path: {value!r}")
    return value


def require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise RuntimeError(f"{label} is not a lowercase SHA256 digest")
    return value


def require_fragments(text: str, fragments: tuple[str, ...], label: str) -> None:
    for fragment in fragments:
        if fragment not in text:
            raise RuntimeError(f"{label} is missing required contract: {fragment}")


def verify_host_ui(
    workbench: Path,
    manifest: dict[str, object],
    expected: dict[str, tuple[int, str]],
) -> int:
    audit = manifest.get("host_ui")
    if not isinstance(audit, dict):
        raise RuntimeError("Format v3+ requires a host_ui audit object")
    if set(audit) != {
        "transport",
        "requires_board_network",
        "server_entrypoint",
        "web_root",
        "files",
    }:
        raise RuntimeError("Unexpected host_ui audit schema")
    if audit.get("transport") != "PC_WEB_SERIAL_USB_UART":
        raise RuntimeError("Unexpected host UI transport")
    if audit.get("requires_board_network") is not False:
        raise RuntimeError("Host Web Serial UI must not require board networking")
    if audit.get("server_entrypoint") != "host_ui/serve_web_serial_ui.py":
        raise RuntimeError("Unexpected host UI server entry point")
    if audit.get("web_root") != "host_ui/web_serial_ui":
        raise RuntimeError("Unexpected host UI web root")

    entries = audit.get("files")
    if not isinstance(entries, list):
        raise RuntimeError("host_ui files must be a list")
    audited: dict[str, tuple[int, str]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {"path", "nbytes", "sha256"}:
            raise RuntimeError(f"Invalid host_ui file entry {index}")
        relative = require_safe_relative_path(entry.get("path"))
        if relative in audited:
            raise RuntimeError(f"Duplicate host_ui path: {relative}")
        nbytes = entry.get("nbytes")
        if not isinstance(nbytes, int) or isinstance(nbytes, bool) or nbytes <= 0:
            raise RuntimeError(f"Invalid host_ui byte count for {relative}")
        digest = require_sha256(entry.get("sha256"), f"host_ui SHA256 for {relative}")
        audited[relative] = (nbytes, digest)

    required_paths = set(HOST_UI_SOURCE_FILES)
    if set(audited) != required_paths:
        raise RuntimeError(
            "host_ui inventory mismatch: "
            f"missing={sorted(required_paths - set(audited))}, "
            f"unexpected={sorted(set(audited) - required_paths)}"
        )
    for relative, signature in audited.items():
        if expected.get(relative) != signature:
            raise RuntimeError(
                f"host_ui digest does not match main file manifest: {relative}"
            )
        path = workbench / relative
        if is_link_like(path):
            raise RuntimeError(f"host_ui file must not be a link: {relative}")
        if path.stat().st_size != signature[0] or sha256_file(path) != signature[1]:
            raise RuntimeError(f"host_ui file mismatch: {relative}")

    server = (workbench / "host_ui/serve_web_serial_ui.py").read_text(
        encoding="utf-8"
    )
    index = (workbench / "host_ui/web_serial_ui/index.html").read_text(
        encoding="utf-8"
    )
    app = (workbench / "host_ui/web_serial_ui/app.js").read_text(
        encoding="utf-8"
    )
    styles = (workbench / "host_ui/web_serial_ui/styles.css").read_text(
        encoding="utf-8"
    )
    readme = (workbench / "WORKBENCH_README.md").read_text(encoding="utf-8")

    require_fragments(
        server,
        (
            'default="127.0.0.1"',
            'default=8000',
            'Path(__file__).resolve().parent / "web_serial_ui"',
            "http.server.ThreadingHTTPServer",
        ),
        "Host UI server",
    )
    require_fragments(
        index,
        (
            "Qwen3 FPGA Prompt Console",
            '<link rel="stylesheet" href="styles.css">',
            '<script src="app.js"></script>',
            'id="connectButton"',
            'id="promptInput"',
            'id="generatedText"',
        ),
        "Host UI HTML",
    )
    require_fragments(
        app,
        (
            "const BAUD_RATE = 115200;",
            '"serial" in navigator',
            "navigator.serial.requestPort()",
            "await port.open({ baudRate: BAUD_RATE",
            "`TOKENS ${maxNew} ${ids.length} ${ids.join(\" \")}`",
            "`PROMPT ${maxNew} ${prompt}`",
            "/^BYTES\\s+",
            "不需要启用板卡网口",
            "Vitis Serial Monitor",
        ),
        "Host UI JavaScript",
    )
    if manifest.get("format_version") in {4, 5}:
        require_fragments(
            app,
            (
                'new TextDecoder("utf-8", { fatal: true })',
                'lineBuffer = "";',
                "模型输出 token #",
            ),
            "Format v4 strict Host UI JavaScript",
        )
    require_fragments(styles, (".connection", ".text-output", "#console"), "Host UI CSS")
    require_fragments(
        readme,
        (
            "Chrome/Edge Web Serial",
            "does **not** use or require the board's Ethernet port",
            "close/stop Vitis Serial Monitor",
            "python .\\host_ui\\serve_web_serial_ui.py",
        ),
        "Workbench README",
    )
    if re.search(r"https?://", index, flags=re.IGNORECASE):
        raise RuntimeError("Host UI HTML must not depend on remote assets")
    for forbidden in ("fetch(", "WebSocket(", "EventSource("):
        if forbidden in app:
            raise RuntimeError(
                f"Host UI unexpectedly depends on a network API: {forbidden}"
            )
    return len(audited)


def verify_host_acceptance(
    workbench: Path,
    manifest: dict[str, object],
    expected: dict[str, tuple[int, str]],
) -> int:
    audit = manifest.get("host_acceptance")
    if not isinstance(audit, dict):
        raise RuntimeError("Format v4+ requires a host_acceptance audit object")
    if set(audit) != {
        "transport",
        "requires_board_network",
        "entrypoint",
        "fixture",
        "baud_rate",
        "case_count",
        "files",
    }:
        raise RuntimeError("Unexpected host_acceptance audit schema")
    if audit.get("transport") != "PY_SERIAL_USB_UART":
        raise RuntimeError("Unexpected host acceptance transport")
    if audit.get("requires_board_network") is not False:
        raise RuntimeError("UART acceptance must not require board networking")
    if audit.get("entrypoint") != "host_tools/run_uart_board_acceptance.py":
        raise RuntimeError("Unexpected host acceptance entry point")
    if audit.get("fixture") != "host_tools/fixtures/uart_board_acceptance_pass.txt":
        raise RuntimeError("Unexpected host acceptance fixture")
    if audit.get("baud_rate") != 115200:
        raise RuntimeError("Unexpected host acceptance baud rate")
    if audit.get("case_count") != HOST_ACCEPTANCE_CASE_COUNT:
        raise RuntimeError("Unexpected host acceptance case count")

    entries = audit.get("files")
    if not isinstance(entries, list):
        raise RuntimeError("host_acceptance files must be a list")
    audited: dict[str, tuple[int, str]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {"path", "nbytes", "sha256"}:
            raise RuntimeError(f"Invalid host_acceptance file entry {index}")
        relative = require_safe_relative_path(entry.get("path"))
        if relative in audited:
            raise RuntimeError(f"Duplicate host_acceptance path: {relative}")
        nbytes = entry.get("nbytes")
        if not isinstance(nbytes, int) or isinstance(nbytes, bool) or nbytes <= 0:
            raise RuntimeError(f"Invalid host_acceptance byte count for {relative}")
        digest = require_sha256(
            entry.get("sha256"), f"host_acceptance SHA256 for {relative}"
        )
        audited[relative] = (nbytes, digest)

    required_paths = set(HOST_ACCEPTANCE_SOURCE_FILES)
    if set(audited) != required_paths:
        raise RuntimeError(
            "host_acceptance inventory mismatch: "
            f"missing={sorted(required_paths - set(audited))}, "
            f"unexpected={sorted(set(audited) - required_paths)}"
        )
    for relative, signature in audited.items():
        if expected.get(relative) != signature:
            raise RuntimeError(
                f"host_acceptance digest does not match main manifest: {relative}"
            )
        path = workbench / relative
        if is_link_like(path):
            raise RuntimeError(f"host_acceptance file must not be a link: {relative}")
        if path.stat().st_size != signature[0] or sha256_file(path) != signature[1]:
            raise RuntimeError(f"host_acceptance file mismatch: {relative}")

    tool = (workbench / "host_tools/run_uart_board_acceptance.py").read_text(
        encoding="utf-8"
    )
    test = (workbench / "host_tools/test_uart_board_acceptance.py").read_text(
        encoding="utf-8"
    )
    fixture = (workbench / "host_tools/fixtures/uart_board_acceptance_pass.txt").read_text(
        encoding="utf-8"
    )
    readme = (workbench / "WORKBENCH_README.md").read_text(encoding="utf-8")
    require_fragments(
        tool,
        (
            '"schema": "qot-uart-board-acceptance-v1"',
            '"TOKENS 2 1 374"',
            '"TOKENS 1 2 374 28458"',
            '"TOKENS 2 5 785 3853 315 89462 374"',
            '"PROMPT 2 The future of FPGA is"',
            '"TOKENS 1 1 151935"',
            '"TOKENS 1 1 151936"',
            '"ERROR PARSE RANGE offset=11"',
            '"--launch-workbench"',
            '"--verify-transcript"',
        ),
        "Host acceptance tool",
    )
    require_fragments(
        test,
        ("unittest", "verify_transcript_bytes", "uart_board_acceptance_pass.txt"),
        "Host acceptance tests",
    )
    require_fragments(
        fixture,
        (
            "Qwen3 text/token prompt demo",
            "READY vocab=151936 context=256",
            "DONE 2 MAX_NEW",
            "ERROR PARSE RANGE offset=11",
        ),
        "Host acceptance fixture",
    )
    require_fragments(
        readme,
        (
            "strict pyserial tool",
            ".\\host_tools\\run_uart_board_acceptance.py",
            "--launch-workbench .",
            "--verify-transcript",
        ),
        "Workbench README acceptance section",
    )
    return len(audited)


def verify_trusted_provenance(
    workbench: Path,
    manifest: dict[str, object],
    expected: dict[str, tuple[int, str]],
) -> bool:
    provenance = manifest.get("trusted_provenance")
    if not isinstance(provenance, dict) or set(provenance) != {
        "generate_elf_sha256",
        "runtime_tool_sha256",
        "host_ui_sha256",
        "host_acceptance_sha256",
    }:
        raise RuntimeError("Format v5 requires the exact trusted provenance schema")
    if provenance.get("generate_elf_sha256") != EXPECTED_GENERATE_ELF_SHA256:
        raise RuntimeError("Trusted a_qgen ELF provenance mismatch")
    if provenance.get("runtime_tool_sha256") != PINNED_RUNTIME_TOOL_SHA256:
        raise RuntimeError("Trusted launcher/wrapper provenance mismatch")
    if provenance.get("host_ui_sha256") != PINNED_HOST_UI_SHA256:
        raise RuntimeError("Trusted host UI provenance mismatch")
    if provenance.get("host_acceptance_sha256") != PINNED_HOST_ACCEPTANCE_SHA256:
        raise RuntimeError("Trusted UART acceptance provenance mismatch")

    pinned = {
        "sw/a_qgen.elf": EXPECTED_GENERATE_ELF_SHA256,
        **PINNED_RUNTIME_TOOL_SHA256,
        **PINNED_HOST_UI_SHA256,
        **PINNED_HOST_ACCEPTANCE_SHA256,
    }
    for relative, digest in pinned.items():
        if expected.get(relative, (None, None))[1] != digest:
            raise RuntimeError(f"Trusted file digest mismatch: {relative}")

    host_tools = workbench / "host_tools"
    with tempfile.TemporaryDirectory(prefix="qot-workbench-verify-") as temporary:
        completed = subprocess.run(
            [sys.executable, str(host_tools / "test_uart_board_acceptance.py"), "-q"],
            cwd=str(host_tools),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60.0,
            check=False,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        if completed.returncode != 0:
            tail = completed.stdout[-4000:]
            raise RuntimeError(
                "Packaged UART acceptance tests failed with "
                f"exit {completed.returncode}:\n{tail}"
            )
        replay = subprocess.run(
            [
                sys.executable,
                str(host_tools / "run_uart_board_acceptance.py"),
                "--verify-transcript",
                str(host_tools / "fixtures/uart_board_acceptance_pass.txt"),
                "--output-dir",
                str(Path(temporary) / "replay"),
            ],
            cwd=str(host_tools),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60.0,
            check=False,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        if replay.returncode != 0 or "PASS: 8 single-flight" not in replay.stdout:
            tail = replay.stdout[-4000:]
            raise RuntimeError(
                "Packaged UART transcript replay failed with "
                f"exit {replay.returncode}:\n{tail}"
            )
    return True


def verify_workbench(workbench: Path) -> dict[str, object]:
    workbench = workbench.absolute()
    if is_link_like(workbench):
        raise RuntimeError("Workbench root must not be a symbolic link or junction")
    workbench = workbench.resolve()
    if not workbench.is_dir():
        raise FileNotFoundError(workbench)
    manifest_path = workbench / "workbench_manifest.json"
    if is_link_like(manifest_path):
        raise RuntimeError("Workbench manifest must not be a symbolic link")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    format_version = manifest.get("format_version")
    if format_version not in SUPPORTED_FORMAT_VERSIONS:
        raise RuntimeError(
            f"Unsupported workbench manifest format: {format_version!r}"
        )
    if manifest.get("state") != "WORKBENCH_NOT_RELEASE":
        raise RuntimeError("Unexpected workbench state")
    if manifest.get("legacy_lineage_id") != LEGACY_LINEAGE_ID:
        raise RuntimeError("Unexpected legacy board lineage")
    if manifest.get("legacy_lineage_file_sha256") != LEGACY_LINEAGE_HASHES:
        raise RuntimeError("Legacy board lineage hash contract mismatch")

    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise RuntimeError("Manifest files must be a list")
    expected: dict[str, tuple[int, str]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise RuntimeError(f"Manifest file entry {index} is not an object")
        relative = require_safe_relative_path(entry.get("path"))
        if relative in expected:
            raise RuntimeError(f"Duplicate manifest path: {relative}")
        nbytes = entry.get("nbytes")
        if not isinstance(nbytes, int) or isinstance(nbytes, bool) or nbytes < 0:
            raise RuntimeError(f"Invalid byte count for {relative}")
        digest = require_sha256(entry.get("sha256"), f"SHA256 for {relative}")
        expected[relative] = (nbytes, digest)

    if manifest.get("file_count") != len(expected):
        raise RuntimeError("Manifest file_count mismatch")
    if manifest.get("total_file_bytes") != sum(v[0] for v in expected.values()):
        raise RuntimeError("Manifest total_file_bytes mismatch")

    actual: dict[str, Path] = {}
    for path in workbench.rglob("*"):
        if is_link_like(path):
            raise RuntimeError(
                f"Workbench must not contain symlinks or junctions: {path}"
            )
        if path.is_file() and path != manifest_path:
            actual[path.relative_to(workbench).as_posix()] = path
    if set(actual) != set(expected):
        raise RuntimeError(
            "Workbench inventory mismatch: "
            f"missing={sorted(set(expected) - set(actual))}, "
            f"unexpected={sorted(set(actual) - set(expected))}"
        )
    for relative, path in actual.items():
        nbytes, digest = expected[relative]
        if path.stat().st_size != nbytes or sha256_file(path) != digest:
            raise RuntimeError(f"Workbench file mismatch: {relative}")

    generate_hash = require_sha256(
        manifest.get("generate_elf_sha256"), "generate ELF SHA256"
    )
    if expected.get("sw/a_qgen.elf", (None, None))[1] != generate_hash:
        raise RuntimeError("a_qgen ELF does not match the manifest-level digest")

    runtime = validate_runtime(workbench / "runtime")
    if runtime != manifest.get("runtime"):
        raise RuntimeError("Runtime audit does not match the workbench manifest")

    launcher = (workbench / "launch_qwen3_board.tcl").read_text(
        encoding="utf-8"
    )
    wrapper = (workbench / "run_board_smoke.ps1").read_text(encoding="utf-8")
    for required in (
        'mode eq "generate"',
        "set generate_elf [file join $package_root sw a_qgen.elf]",
        "dow $generate_elf",
    ):
        if required not in launcher:
            raise RuntimeError(f"Launcher is missing generate contract: {required}")
    if "generate" not in wrapper or "ValidateSet" not in wrapper:
        raise RuntimeError("PowerShell wrapper is missing generate mode")

    if format_version == 2:
        if "host_ui" in manifest:
            raise RuntimeError("Legacy format v2 must not carry a v3 host_ui audit")
        if "host_acceptance" in manifest:
            raise RuntimeError("Legacy format v2 must not carry a v4 acceptance audit")
        host_ui_file_count = 0
    else:
        host_ui_file_count = verify_host_ui(workbench, manifest, expected)

    if format_version in {2, 3}:
        if "host_acceptance" in manifest:
            raise RuntimeError(
                "Legacy format must not carry a v4 host_acceptance audit"
            )
        host_acceptance_file_count = 0
    else:
        host_acceptance_file_count = verify_host_acceptance(
            workbench, manifest, expected
        )

    if format_version == 5:
        trusted_provenance_passed = verify_trusted_provenance(
            workbench, manifest, expected
        )
    else:
        if "trusted_provenance" in manifest:
            raise RuntimeError(
                "Legacy format must not carry a v5 trusted_provenance audit"
            )
        trusted_provenance_passed = False

    return {
        "format_version": format_version,
        "file_count": len(expected),
        "total_file_bytes": manifest["total_file_bytes"],
        "generate_elf_sha256": generate_hash,
        "runtime_segment_count": runtime["segment_count"],
        "host_ui_file_count": host_ui_file_count,
        "host_acceptance_file_count": host_acceptance_file_count,
        "trusted_provenance_passed": trusted_provenance_passed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify an interactive token-ID board workbench"
    )
    parser.add_argument("workbench", type=Path)
    return parser.parse_args()


def main() -> int:
    result = verify_workbench(parse_args().workbench)
    print(
        "PASS prompt demo workbench verification: "
        f"format v{result['format_version']}, {result['file_count']} files, "
        f"{result['total_file_bytes']} bytes, "
        f"{result['runtime_segment_count']} runtime segments, "
        f"{result['host_ui_file_count']} host UI files, "
        f"{result['host_acceptance_file_count']} host acceptance files, "
        f"trusted_provenance={result['trusted_provenance_passed']}"
    )
    print(f"generate_elf_sha256={result['generate_elf_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
