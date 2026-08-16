from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


LEGACY_LINEAGE_ID = "full28-board-pass-20260808"
WORKBENCH_FORMAT_VERSION = 5
EXPECTED_GENERATE_ELF_SHA256 = (
    "f37eb88d4e1b75fec815d01306ee85678c1e8555d02e5b42d3efca22fd337bbe"
)
PINNED_RUNTIME_TOOL_SHA256 = {
    "launch_qwen3_board.tcl":
        "d72a49b21175c4ae28264cf5a2962f478247d0f1df501545e883f574249c7bea",
    "run_board_smoke.ps1":
        "483def4143ab7d0530a6de688f9695f6c7740c20c67df3eaac3d7e8544dfadc6",
}
HOST_UI_SOURCE_FILES = {
    "host_ui/serve_web_serial_ui.py": "serve_web_serial_ui.py",
    "host_ui/web_serial_ui/index.html": "web_serial_ui/index.html",
    "host_ui/web_serial_ui/app.js": "web_serial_ui/app.js",
    "host_ui/web_serial_ui/styles.css": "web_serial_ui/styles.css",
}
PINNED_HOST_UI_SHA256 = {
    "host_ui/serve_web_serial_ui.py":
        "989b37fd2f596016cd6e8e128d717e35b4232ad7d164b11407a20fde1c0ad6f1",
    "host_ui/web_serial_ui/index.html":
        "43e807bf5e93b877792a3c06900612199f1125c5c771b59388275105e0cc7140",
    "host_ui/web_serial_ui/app.js":
        "09c212fe6a4a58c327c73b763e8c6d726696d270b5ace51944fe262df6a01944",
    "host_ui/web_serial_ui/styles.css":
        "34417ea346a3c85121c97610c9b67bc3db1b6be9d7b2908c2bec894ca983e5e1",
}
HOST_ACCEPTANCE_SOURCE_FILES = {
    "host_tools/run_uart_board_acceptance.py": "run_uart_board_acceptance.py",
    "host_tools/test_uart_board_acceptance.py": "test_uart_board_acceptance.py",
    "host_tools/fixtures/uart_board_acceptance_pass.txt":
        "fixtures/uart_board_acceptance_pass.txt",
}
PINNED_HOST_ACCEPTANCE_SHA256 = {
    "host_tools/run_uart_board_acceptance.py":
        "6f1c30fc31a33d9c9fe100df804896f7591d51153687665d622cdcfad24ab007",
    "host_tools/test_uart_board_acceptance.py":
        "4f47b3d2e6c95d58bf60ec31fa7b281ffb2af85143f54f6252954af8b106248e",
    "host_tools/fixtures/uart_board_acceptance_pass.txt":
        "b398734dc4ebe15cd5c1ec06f625e371012af6c07b8f2c1df03161113966b970",
}
HOST_ACCEPTANCE_CASE_COUNT = 8
LEGACY_LINEAGE_HASHES = {
    "hw/llm_system_qwen3_one_token_boardready.bit":
        "b4a4c6133de7af03f586e41bfc191b3a9379dab3953e518d2660f0e2cff7cc34",
    "hw/llm_system_qwen3_one_token_boardready.mmi":
        "f2b7a008d5a6561c1162ae811e19a68c018bfeda704b540da239f6032c000f86",
    "hw/llm_system_qwen3_one_token_boardready.xsa":
        "dc0e2ddf6064df29475bdf0e0a125223cee0b7af3209bad05d65b3a5a19d28c8",
    "hw/psu_init_gpl.c":
        "92a78c88b740486d3709d51e43a868bf18197303f8108231a1841f2a1cb1d1d0",
    "hw/psu_init_gpl.h":
        "793882b4ddc7b663bebf346b577d243597f4a5d0a1a925da007fa4526e8840f6",
    "hw/psu_init.c":
        "c111b8c5fdcf82b0e57140bda4f3b9e367ac708edadc3ed121c9b349435f9e1e",
    "hw/psu_init.h":
        "107135b806e6dce5a2038539f95e137856811ca4b1f3979a9c0172e56023c859",
    "hw/psu_init.html":
        "d1c58b691913553a4e757d50422a75b76806347b8fb917d124392a15fc7a5912",
    "hw/psu_init.tcl":
        "d7272e24620140428aa2556b9a107fa4cf65a42cf2d46e3f4ba495df0d9245e8",
    "sw/a_qctl.elf":
        "9f954f9a5364cbdb7e362754c049bf509cee9807d36ac67596dee68f7ef30e8b",
    "sw/a_qmdl.elf":
        "1ccd7b64d0481982586a0cc16477f7222568023ac06799f7494a666ec0e53c17",
    "sw/fsbl.elf":
        "e1c8a115a8e539868355987da6908b028ee7ced635f547933dca174d754f8e2f",
    "runtime/full_chain_manifest.json":
        "22c0be1631f62f88527719d4aa9686f5ef5784f85b05b4cb92f5ad97742ae730",
    "runtime/load_pl_ddr_runtime.tcl":
        "0b698f28165f05efe7dbc2a8c64bfa7379cbf10cc8dcb4b1ced9660eeabbecb1",
    "runtime/pl_ddr_binary_segments.json":
        "fa8981e71101def29970135df5e863da5634274dd9fb64905666f0cf1d47d3f2",
    "runtime/pl_ddr_runtime_load_plan.json":
        "1682370500fa3d404b58ea9024761b843bd3c544e493ff845335b4d6283182b9",
    "launch_qwen3_board.tcl":
        "29c206f27c2a4ab534c294ceb85d72cf938960b5e50553f36d33c638fe19acae",
    "run_board_smoke.ps1":
        "31bf2c626b255a4abec58ae98567b6368784802c80574c74c11bad637fe385ed",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_expected_hash(path: Path, expected: str, label: str) -> Path:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA256 mismatch: {actual} != trusted {expected}"
        )
    return path


def require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} is missing or empty: {path}")
    return path


def is_same_or_within(path: Path, parent: Path) -> bool:
    path = path.resolve()
    parent = parent.resolve()
    return path == parent or parent in path.parents


def require_safe_output(
    output: Path,
    *,
    repository_root: Path,
    inputs: tuple[Path, ...],
) -> None:
    if output == Path(output.anchor):
        raise RuntimeError(f"Refusing unsafe workbench output: {output}")
    if is_same_or_within(output, repository_root):
        raise RuntimeError(
            "Workbench output must be outside the source repository: "
            f"{output}"
        )
    for source in inputs:
        if is_same_or_within(output, source) or is_same_or_within(source, output):
            raise RuntimeError(
                "Workbench output must not overlap an input path: "
                f"output={output}, input={source}"
            )


def require_plain_file_name(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty file name")
    candidate = Path(value)
    if candidate.is_absolute() or candidate.name != value or value in {".", ".."}:
        raise RuntimeError(f"{label} is not a plain file name: {value!r}")
    return value


def validate_legacy_lineage(base: Path) -> dict[str, str]:
    actual_files = {
        path.relative_to(base).as_posix()
        for path in base.rglob("*")
        if path.is_file()
    }
    segment_manifest = json.loads(
        require_file(
            base / "runtime" / "pl_ddr_binary_segments.json",
            "legacy runtime manifest",
        ).read_text(encoding="utf-8")
    )
    segments = segment_manifest.get("segments")
    if not isinstance(segments, list):
        raise RuntimeError("Legacy runtime segment list is missing")
    segment_files = {
        "runtime/" + require_plain_file_name(
            segment.get("file"), f"legacy segment {index} file"
        )
        for index, segment in enumerate(segments)
    }
    expected_files = set(LEGACY_LINEAGE_HASHES) | segment_files
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        raise RuntimeError(
            "Legacy board package inventory mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )

    for relative, expected_hash in LEGACY_LINEAGE_HASHES.items():
        actual_hash = sha256_file(require_file(base / relative, relative))
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Legacy board package hash mismatch for {relative}: "
                f"{actual_hash} != {expected_hash}"
            )
    return dict(LEGACY_LINEAGE_HASHES)


def validate_runtime(runtime_dir: Path) -> dict[str, object]:
    manifest_path = require_file(
        runtime_dir / "pl_ddr_binary_segments.json",
        "PL-DDR segment manifest",
    )
    loader_path = require_file(
        runtime_dir / "load_pl_ddr_runtime.tcl",
        "PL-DDR XSDB loader",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    segments = manifest.get("segments")
    if not isinstance(segments, list) or len(segments) != 61:
        raise RuntimeError("Prompt workbench requires exactly 61 runtime segments")
    if manifest.get("segment_count") != len(segments):
        raise RuntimeError("Runtime segment_count does not match the segment list")

    expected_loader_operations: list[str] = []
    total_bytes = 0
    for index, segment in enumerate(segments):
        if segment.get("index") != index:
            raise RuntimeError(f"Runtime segment index mismatch at {index}")
        segment_name = require_plain_file_name(
            segment.get("file"), f"runtime segment {index} file"
        )
        segment_path = require_file(
            runtime_dir / segment_name,
            f"runtime segment {index}",
        )
        expected_size = int(segment["nbytes"])
        if segment_path.stat().st_size != expected_size:
            raise RuntimeError(f"Runtime segment size mismatch: {segment_path.name}")
        if sha256_file(segment_path).lower() != str(segment["sha256"]).lower():
            raise RuntimeError(f"Runtime segment SHA256 mismatch: {segment_path.name}")
        total_bytes += expected_size
        expected_loader_operations.extend(
            (
                'set segment_file [file join $script_dir '
                f'"{segment_name}"]',
                "dow -data -bypass-cache-sync $segment_file "
                f'0x{int(segment["address"]):016X}',
            )
        )

    loader_operations = [
        line.rstrip()
        for line in loader_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("set segment_file ") or line.startswith("dow ")
    ]
    if loader_operations != expected_loader_operations:
        raise RuntimeError("PL-DDR loader operations do not match the manifest")
    if total_bytes != int(manifest.get("total_segment_bytes", -1)):
        raise RuntimeError("Runtime total byte count mismatch")
    return {
        "segment_count": len(segments),
        "total_segment_bytes": total_bytes,
        "manifest_sha256": sha256_file(manifest_path),
        "loader_sha256": sha256_file(loader_path),
        "download_command": "dow -data -bypass-cache-sync",
    }


def build_readme() -> str:
    return """# Qwen3 FPGA interactive prompt workbench

State: `WORKBENCH_NOT_RELEASE`

This directory derives from the board-tested 2026-08-08 full28 hardware/runtime
package and adds the new interactive `a_qgen` application with an embedded,
hash-validated PS-native Qwen tokenizer. It does not rewrite the preserved
2026-08-08 package. Prompt-to-text is still pending physical-board acceptance.

This package contains two host-side demo interfaces:

- Web GUI: `host_ui/serve_web_serial_ui.py` plus the audited static files under
  `host_ui/web_serial_ui/`.
- UART TUI: any 115200 8N1 serial terminal can send the same line protocol.

The Web GUI uses the PC's USB-UART connection through Chrome/Edge Web Serial.
It does **not** use or require the board's Ethernet port. A COM port can have
only one owner, so close/stop Vitis Serial Monitor before clicking **Connect**
in Chrome or Edge.

For the physical acceptance gate, use the audited strict pyserial tool instead
of treating the display-only Web GUI as a PASS oracle. The tool owns the COM
port, launches this workbench, binds results to the new application startup,
checks all eight ordered cases, and saves both raw UART bytes and a JSON report:

```powershell
conda run -n llm_fpga python `
  .\\host_tools\\run_uart_board_acceptance.py `
  --port COM230 `
  --launch-workbench .
```

Close both Vitis Serial Monitor and the Web GUI before running it. A packaged
transcript-only self-check that does not require a board is also available:

```powershell
conda run -n llm_fpga python `
  .\\host_tools\\run_uart_board_acceptance.py `
  --verify-transcript .\\host_tools\\fixtures\\uart_board_acceptance_pass.txt
```

Board and Web GUI sequence:

1. Keep the board in JTAG boot mode and connect both JTAG and USB-UART. Do not
   leave the UART COM port open in Vitis Serial Monitor.
2. From this directory, launch the board application:

   ```powershell
   .\\run_board_smoke.ps1 -Mode generate
   ```

3. In a second terminal, start the local static-file server:

   ```powershell
   python .\\host_ui\\serve_web_serial_ui.py
   ```

4. Open `http://127.0.0.1:8000/` in desktop Chrome or Edge, click **Connect**,
   select the board's USB-UART COM port, and submit either text or token IDs.

For the UART TUI fallback, do not start the browser GUI. Instead open the COM
port in one serial terminal at 115200 8N1, wait for
`READY vocab=151936 context=256`, then enter:

   ```text
   PING
   TOKENS 2 1 374
   PROMPT 2 The future of FPGA is
   ```

The known first request must emit token `28458` first, feed that actual result
back at position 1, then emit token `64`, followed by `DONE 2 MAX_NEW`.
The text request must report prompt IDs `785 3853 315 89462 374`, then emit
`264/1296911292` and `26291/1225544557`; their PS-side decoded bytes form
` a fascinating`. Repeat the text request once and require bit-exact output.

General grammar:

```text
HELP
PING
TOKENS <max_new> <count> <token_id_0> ... <token_id_n>
PROMPT <max_new> <single-line UTF-8 text>
```

Both modes share the same retained-KV session and actual-result feedback loop.
Generated text is transported as per-token `BYTES <index> <hex>` records so a
browser can preserve UTF-8 decoder state across token boundaries.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a non-destructive interactive board workbench"
    )
    parser.add_argument("--base-package", required=True, type=Path)
    parser.add_argument("--generate-elf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    repository_root = script_dir.parents[2]
    runtime_source_dir = script_dir.parent / "qmap_one_token_runtime"
    host_ui_sources = {
        destination: require_file(script_dir / source, destination)
        for destination, source in HOST_UI_SOURCE_FILES.items()
    }
    host_acceptance_sources = {
        destination: require_file(script_dir / source, destination)
        for destination, source in HOST_ACCEPTANCE_SOURCE_FILES.items()
    }
    base = args.base_package.resolve()
    output = args.output.resolve()
    generate_elf = require_expected_hash(
        require_file(args.generate_elf, "interactive generation ELF"),
        EXPECTED_GENERATE_ELF_SHA256,
        "interactive generation ELF",
    )
    launcher_source = require_expected_hash(
        require_file(runtime_source_dir / "launch_qwen3_board.tcl", "launcher"),
        PINNED_RUNTIME_TOOL_SHA256["launch_qwen3_board.tcl"],
        "launcher",
    )
    wrapper_source = require_expected_hash(
        require_file(
            runtime_source_dir / "run_board_smoke.ps1", "PowerShell wrapper"
        ),
        PINNED_RUNTIME_TOOL_SHA256["run_board_smoke.ps1"],
        "PowerShell wrapper",
    )
    for relative, source in host_ui_sources.items():
        require_expected_hash(source, PINNED_HOST_UI_SHA256[relative], relative)
    for relative, source in host_acceptance_sources.items():
        require_expected_hash(
            source, PINNED_HOST_ACCEPTANCE_SHA256[relative], relative
        )
    require_safe_output(
        output,
        repository_root=repository_root,
        inputs=(base, generate_elf, runtime_source_dir),
    )
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing workbench: {output}")
    if not base.is_dir():
        raise FileNotFoundError(f"Base package is missing: {base}")
    base_hashes = validate_legacy_lineage(base)
    runtime_audit = validate_runtime(base / "runtime")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging-", dir=str(output.parent)
        )
    ).resolve()
    try:
        for relative in sorted(
            set(base_hashes)
            - {"launch_qwen3_board.tcl", "run_board_smoke.ps1"}
        ):
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(base / relative, destination)
        segment_manifest = json.loads(
            (base / "runtime" / "pl_ddr_binary_segments.json").read_text(
                encoding="utf-8"
            )
        )
        for index, segment in enumerate(segment_manifest["segments"]):
            segment_name = require_plain_file_name(
                segment.get("file"), f"runtime segment {index} file"
            )
            shutil.copy2(
                base / "runtime" / segment_name,
                staging / "runtime" / segment_name,
            )
        shutil.copy2(generate_elf, staging / "sw" / "a_qgen.elf")
        shutil.copy2(
            launcher_source,
            staging / "launch_qwen3_board.tcl",
        )
        shutil.copy2(
            wrapper_source,
            staging / "run_board_smoke.ps1",
        )
        for relative, source in host_ui_sources.items():
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        for relative, source in host_acceptance_sources.items():
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        (staging / "WORKBENCH_README.md").write_text(
            build_readme(), encoding="utf-8"
        )

        files = []
        for path in sorted(staging.rglob("*")):
            if path.is_file() and path != staging / "workbench_manifest.json":
                files.append(
                    {
                        "path": path.relative_to(staging).as_posix(),
                        "nbytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        host_ui_files = []
        for relative in HOST_UI_SOURCE_FILES:
            copied = staging / relative
            host_ui_files.append(
                {
                    "path": relative,
                    "nbytes": copied.stat().st_size,
                    "sha256": sha256_file(copied),
                }
            )
        host_acceptance_files = []
        for relative in HOST_ACCEPTANCE_SOURCE_FILES:
            copied = staging / relative
            host_acceptance_files.append(
                {
                    "path": relative,
                    "nbytes": copied.stat().st_size,
                    "sha256": sha256_file(copied),
                }
            )
        manifest = {
            "format_version": WORKBENCH_FORMAT_VERSION,
            "state": "WORKBENCH_NOT_RELEASE",
            "legacy_lineage_id": LEGACY_LINEAGE_ID,
            "legacy_lineage_file_sha256": base_hashes,
            "runtime": runtime_audit,
            "generate_elf_sha256": sha256_file(generate_elf),
            "host_ui": {
                "transport": "PC_WEB_SERIAL_USB_UART",
                "requires_board_network": False,
                "server_entrypoint": "host_ui/serve_web_serial_ui.py",
                "web_root": "host_ui/web_serial_ui",
                "files": host_ui_files,
            },
            "host_acceptance": {
                "transport": "PY_SERIAL_USB_UART",
                "requires_board_network": False,
                "entrypoint": "host_tools/run_uart_board_acceptance.py",
                "fixture": "host_tools/fixtures/uart_board_acceptance_pass.txt",
                "baud_rate": 115200,
                "case_count": HOST_ACCEPTANCE_CASE_COUNT,
                "files": host_acceptance_files,
            },
            "trusted_provenance": {
                "generate_elf_sha256": EXPECTED_GENERATE_ELF_SHA256,
                "runtime_tool_sha256": PINNED_RUNTIME_TOOL_SHA256,
                "host_ui_sha256": PINNED_HOST_UI_SHA256,
                "host_acceptance_sha256": PINNED_HOST_ACCEPTANCE_SHA256,
            },
            "file_count": len(files),
            "total_file_bytes": sum(item["nbytes"] for item in files),
            "files": files,
        }
        (staging / "workbench_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        for item in files:
            copied = staging / item["path"]
            if (copied.stat().st_size != item["nbytes"] or
                    sha256_file(copied) != item["sha256"]):
                raise RuntimeError(
                    f"Post-copy verification failed: {item['path']}"
                )
        os.replace(staging, output)
    except Exception:
        if staging.exists() and staging.parent == output.parent:
            shutil.rmtree(staging)
        raise

    print(
        "PASS prompt demo workbench: "
        f"{len(files)} files, {manifest['total_file_bytes']} bytes"
    )
    print(f"output={output}")
    print(f"generate_elf_sha256={manifest['generate_elf_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
