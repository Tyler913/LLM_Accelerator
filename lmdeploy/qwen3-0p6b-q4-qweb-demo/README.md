# Qwen3 0.6B Q4 FPGA QWEB Demo

This directory is the portable release location for the JTAG-loaded board-hosted
Web demo. The PS application tokenizes a prompt, submits the token sequence to
the PL full-28-layer inference path, detokenizes the returned token IDs, and
serves the result through the PS Ethernet interface.

The large Q4 runtime segments are deliberately not stored in GitHub. They are
restored from the Hugging Face model repository
[`Tyler01/qwen3-0p6b-fpga-q4-runtime`](https://huggingface.co/Tyler01/qwen3-0p6b-fpga-q4-runtime).

## Validation status

The source artifacts from which this package is assembled passed the physical
board flow in their original external locations (`F:\vwk` plus the accepted v13
runtime workbench). The new relative-path layout and portable `run_demo.ps1`
wrapper now pass local `-AuditOnly`; they still require **one new physical board
run** before this portable package is described as board-accepted. Moving
unchanged bitstreams, ELFs, or Q4 segments does not by itself validate a new
launcher.

The demo is JTAG-loaded. Autonomous `BOOT.BIN`, SD, or QSPI boot is not part of
this release.

## Directory layout

```text
lmdeploy/qwen3-0p6b-q4-qweb-demo/
|-- README.md
|-- release_manifest.json
|-- run_demo.ps1
|-- board/
|   |-- llm_system_qwen3_one_token_boardready.bit
|   |-- network_input_0af1257442a68a6beb31d94811713f2ae8e6af63e0a85dd497146405e37406cf.xsa
|   |-- fsbl.elf
|   `-- a_qweb.elf
|-- scripts/
|   |-- launch_qweb_board.tcl
|   `-- capture_qweb_uart.py
|-- assets/
|   `-- qwen3_tokenizer.qtk
|-- source/
|   `-- FPGA_Project/               # canonical RTL/Vivado/software source layout
`-- model/
    |-- README.md
    |-- LICENSE
    |-- load_pl_ddr_runtime.tcl
    |-- pl_ddr_binary_segments.json
    |-- full_chain_manifest.json
    |-- pl_ddr_runtime_load_plan.json
    `-- qwen3_runtime_00.bin ... qwen3_runtime_60.bin
```

The 61 `qwen3_runtime_*.bin` files appear only after the Hugging Face download.
The loader stays beside them because it resolves every segment relative to its
own script location.

The canonical development sources remain in the normal `FPGA_Project/` tree.
The `source/FPGA_Project/` directory is a clean final-demo snapshot in the same
relative layout as the development tree. It includes PS/PL implementation and
the non-test builder dependencies, while omitting host tests, traces, temporary
outputs, and regenerated Vitis workspaces. The checked-in bitstream, XSA, FSBL,
and application ELF are the run-ready path. Asset restore still uses the
repository-level `init/` scripts, and source rebuilds still require AMD tools.

## GitHub and Hugging Face split

GitHub stores:

- project source code and documentation;
- `run_demo.ps1`, the XSDB launcher, and UART capture tool;
- `release_manifest.json` and the segment/address/checksum manifests;
- the final bitstream, XSA, FSBL, and `a_qweb.elf`;
- the small model README and other non-weight metadata.

Hugging Face stores:

- exactly `qwen3_runtime_00.bin` through `qwen3_runtime_60.bin`;
- 61 segments totaling exactly `394547200` bytes;
- the model card and upstream-compatible license notice.

The GitHub manifest is the verification authority. The Hugging Face revision in
the init manifest must be replaced with the exact Hub commit after the owner
uploads the files; a floating `main` revision is not sufficient for a
reproducible release.

## Prerequisites

- Windows PowerShell or PowerShell 7.
- The repository's `llm_fpga` Conda environment, including
  `huggingface_hub` and `pyserial`.
- AMD Vitis 2025.1 with XSDB and a running or launchable `hw_server`.
- The supported target board connected through JTAG.
- The same board connected to the PC through its UART and `PS_ETH` ports.
- An unused host IPv4 address on the board subnet when using a direct Ethernet
  cable.

AMD tools are licensed external dependencies and are not downloaded by the
project init scripts. The host network adapter is also not modified by init.
If the Hub repository is private, authenticate before downloading:

```powershell
conda run -n llm_fpga hf auth login
conda run -n llm_fpga hf auth whoami
```

## Restore and verify the Q4 runtime

Run from the repository root:

```powershell
conda run -n llm_fpga python init/download_q4_runtime.py
conda run -n llm_fpga python init/verify_q4_runtime.py
```

Those are the normal post-publication commands and require an exact Hub commit
in `init/q4_runtime_assets.json`. During the one-time first upload only, the
owner verifies the already-local files with
`init/verify_q4_runtime.py --allow-unpinned`, uploads them, pins the returned
commit, and then runs `init/verify_q4_runtime.py --from-hub`.

The downloader restores the ignored `.bin` files under this directory's
`model/` folder. Verification must reject missing or extra segment names, a
segment count other than 61, a byte total other than `394547200`, and any size
or SHA-256 mismatch against `pl_ddr_binary_segments.json`.

Then audit the entire portable launch set without invoking XSDB or touching the
board:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\lmdeploy\qwen3-0p6b-q4-qweb-demo\run_demo.ps1 `
  -AuditOnly
```

A passing audit must include the four board artifacts, the loader and manifests,
and all `61 segments / 394547200 bytes`. Do not proceed to a physical launch if
this command fails.

## Physical board and UART flow

Use a controlled one-board bench. JTAG, UART, and Ethernet must all belong to
the same board. Connect `PS_ETH` before launching the image. Replace `COM230`
below if Windows assigned a different UART port. The evidence directory must be
new and must not already exist.

In terminal A, start UART capture first:

```powershell
$RepoRoot = (git rev-parse --show-toplevel).Trim()
$DemoRoot = Join-Path $RepoRoot 'lmdeploy\qwen3-0p6b-q4-qweb-demo'
$qwebEvidence = Join-Path ([IO.Path]::GetTempPath()) 'qweb_portable_rerun_01'
conda run -n llm_fpga python `
  "$DemoRoot\scripts\capture_qweb_uart.py" `
  --port COM230 `
  --output-dir $qwebEvidence `
  --timeout 3600
```

In terminal B, wait until capture owns the UART evidence file, then launch:

```powershell
$RepoRoot = (git rev-parse --show-toplevel).Trim()
$DemoRoot = Join-Path $RepoRoot 'lmdeploy\qwen3-0p6b-q4-qweb-demo'
$qwebEvidence = Join-Path ([IO.Path]::GetTempPath()) 'qweb_portable_rerun_01'
$deadline = (Get-Date).AddSeconds(10)
while (-not ((Test-Path -LiteralPath "$qwebEvidence\capture.claim.json") -and
             (Test-Path -LiteralPath "$qwebEvidence\capture.heartbeat.json") -and
             (Test-Path -LiteralPath "$qwebEvidence\uart_raw.bin"))) {
  if ((Get-Date) -ge $deadline) {
    throw 'UART capture did not create claim, heartbeat, and raw evidence files'
  }
  Start-Sleep -Milliseconds 200
}

powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  "$DemoRoot\run_demo.ps1" `
  -EvidenceDirectory $qwebEvidence
```

The launch programs the bitstream, loads the XSA and FSBL, downloads the 61
runtime segments into PL DDR4, checks the QMAP packet headers, and starts
`a_qweb.elf`. It writes `launch.json` and `xsdb.log` into the same evidence
directory. Keep terminal A open until it writes `startup.json` and prints the
current run's exact `QWEB READY http://.../` URL. Capture and launch reports are
bound by capture/run UUIDs, PID/start time, a live heartbeat, timestamps, and
the capture-claim SHA-256; a dead capture or stale READY line cannot close the
new physical gate. The wrapper also enforces the six-file AMD Vitis 2025.1.1
toolchain hash set recorded in the release manifest and refuses alternate
patched XSDB executables.

## PC Ethernet and Web UI flow

Use the IPv4 address printed by the current UART startup; do not rely on an old
browser bookmark. On the accepted direct-link bench the board used
`192.168.1.10`. If the new run reports that fallback address, configure the
connected PC adapter with an unused address such as `192.168.1.20/24`, with no
gateway required for the direct cable.

1. Confirm the UART log reports a resolved Ethernet link and a `QWEB READY`
   URL.
2. Ping the board address from the Ethernet adapter used for `PS_ETH`.
3. Open the exact UART-reported URL in a browser.
4. Enter a prompt in the Web UI and wait for the board job to finish.
5. Preserve the UART and launch evidence from the same run when performing the
   required portable-wrapper physical revalidation.

The first-version server is intentionally bounded and should be used by one UI
client and one inference job at a time.
