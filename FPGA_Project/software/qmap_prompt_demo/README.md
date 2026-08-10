# QMAP Prompt Demo: Text and Token-ID Generation

This directory is the first interactive PS application above the validated
full28 `run_one_token` datapath. It does not replace or modify the fixed board
smoke in `../qmap_one_token_runtime/main.c`.

## Current Scope

The application accepts either a bounded token-ID prompt or a single-line
UTF-8 prompt, then greedily emits later tokens. Text mode uses the PS-native
Qwen ByteLevel-BPE tokenizer/detokenizer in `tokenizer_runtime/`; model math
and retained KV remain in PL. Total prompt plus decode positions are bounded by
the current 256-position hardware context.

The UART command grammar is strict and case-sensitive:

```text
HELP
PING
TOKENS <max_new> <count> <token_id_0> ... <token_id_n>
PROMPT <max_new> <UTF-8 text>
```

Limits are `max_new=1..256`, `count=1..256`, and
`token_id=0..151935`. Every `TOKENS` command starts a fresh software session at
the configured position, normally position zero.

Typical output is:

```text
START prompt=2 max_new=3
PREFILL 1/2
TOKEN 0 28458 1227344433
BYTES 0 ...hexadecimal raw token bytes...
TOKEN 1 64 1015661901
BYTES 1 ...hexadecimal raw token bytes...
TOKEN 2 ... ...
DONE 3 MAX_NEW
```

`TOKEN` indices are zero-based. Stop reasons are `EOS`, `IM_END`, `MAX_NEW`,
`CONTEXT`, and `HW_ERROR`. Qwen3 defaults are EOS token `151643` and IM_END
token `151645`. `BYTES` records are the PS-side detokenized raw bytes; the Web
Serial page feeds them through a streaming UTF-8 decoder. Model-only IDs
`151669..151935` have no tokenizer string, so the application stops with an
explicit `ERROR DETOKENIZE` instead of silently substituting a token.

## Session Contract

`qot_session_step()` executes at most one full-model invocation:

1. Prompt ids run serially at consecutive positions and populate retained KV.
2. The result from the final prompt id is emitted as generated token zero.
3. Each later result token is used as the actual tied-embedding input at the
   next position.
4. The token event is emitted before its matching terminal `DONE` event.
5. A prompt may exactly fill the 256-position hardware context; its final run
   still predicts one output. If more output was requested, the session then
   stops with `CONTEXT` without an out-of-range launch.

The runner is injectable. The default runner calls `qot_run_token()`. Host tests
use a deterministic mock, and a future network runner can perform the same
single-token configure/start/poll sequence while pumping lwIP during polling.
No change to `qmap_one_token_runtime.h` is required for that extension.

The prompt token array is caller-owned and must remain valid until the session
reaches `DONE` or `ERROR`. Beginning another session resets software indices;
hardware correctness relies on the accelerator's established invariant that
attention only reads KV entries through the current position.

## Files

- `qot_session.[ch]`: prefill/decode state machine and runner injection seam.
- `qot_protocol.[ch]`: allocation-free, explicitly bounded command parser.
- `qot_uart.[ch]`: callback-based UART line input with CR/LF, echo, backspace,
  and overflow draining.
- `main_generate.c`: model readiness checks and the UART command loop.
- `tokenizer/`: deterministic QTKBPE1 exporter, locked Unicode provenance,
  asset verifier, and golden corpus.
- `tokenizer_runtime/`: allocation-free C parser, exact PS tokenizer, and raw
  byte streaming detokenizer.
- `test_session_host.c`: sequencing, feedback, stops, validation, and errors.
- `test_protocol_host.c`: grammar, numeric, count, vocabulary, and buffer bounds.
- `test_uart_host.c`: CR/LF, editing, overflow draining, EOF, and I/O errors.
- `test_prompt_chain_host.c`: exact text tokenize, injectable full-prompt
  session feedback, generated-token detokenize, and model-only-ID rejection.
- `web_serial_ui/`: browser GUI for text and token-ID modes over the same UART.
- `serve_web_serial_ui.py`: localhost static-file server for Web Serial.
- `run_uart_board_acceptance.py`: strict pyserial launcher, UART recorder, and
  exact eight-transaction physical acceptance oracle.
- `test_uart_board_acceptance.py` and `fixtures/`: transcript/live-driver unit
  tests plus the deterministic PASS replay fixture.

## Host Tests

From the repository root with GCC available:

```powershell
$out = 'Temp/qmap_prompt_demo_host'
New-Item -ItemType Directory -Force $out | Out-Null
gcc -std=c11 -Wall -Wextra -Werror -pedantic `
  -I FPGA_Project/software/qmap_one_token_runtime `
  -I FPGA_Project/software/qmap_prompt_demo `
  FPGA_Project/software/qmap_prompt_demo/qot_session.c `
  FPGA_Project/software/qmap_prompt_demo/test_session_host.c `
  -o "$out/test_session_host.exe"
& "$out/test_session_host.exe"

gcc -std=c11 -Wall -Wextra -Werror -pedantic `
  -I FPGA_Project/software/qmap_one_token_runtime `
  -I FPGA_Project/software/qmap_prompt_demo `
  FPGA_Project/software/qmap_prompt_demo/qot_protocol.c `
  FPGA_Project/software/qmap_prompt_demo/test_protocol_host.c `
  -o "$out/test_protocol_host.exe"
& "$out/test_protocol_host.exe"

gcc -std=c11 -Wall -Wextra -Werror -pedantic `
  -I FPGA_Project/software/qmap_prompt_demo `
  FPGA_Project/software/qmap_prompt_demo/qot_uart.c `
  FPGA_Project/software/qmap_prompt_demo/test_uart_host.c `
  -o "$out/test_uart_host.exe"
& "$out/test_uart_host.exe"

gcc -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror `
  -Wconversion -Wsign-conversion -Wshadow -Wstrict-prototypes `
  -Wmissing-prototypes `
  -I FPGA_Project/software/qmap_one_token_runtime `
  -I FPGA_Project/software/qmap_prompt_demo `
  -I FPGA_Project/software/qmap_prompt_demo/tokenizer_runtime `
  FPGA_Project/software/qmap_prompt_demo/qot_session.c `
  FPGA_Project/software/qmap_prompt_demo/tokenizer_runtime/qtk_tokenizer_runtime.c `
  FPGA_Project/software/qmap_prompt_demo/tokenizer_runtime/qtk_text_tokenizer.c `
  FPGA_Project/software/qmap_prompt_demo/test_prompt_chain_host.c `
  -o "$out/test_prompt_chain_host.exe"
& "$out/test_prompt_chain_host.exe" `
  Temp/qmap_prompt_demo_tokenizer/qwen3_tokenizer.qtk
```

These are host-only tests. They do not launch Vivado, Vitis, simulation, XSDB,
or board hardware.

The strict board-acceptance parser and live-driver tests are also host-only:

```powershell
conda run -n llm_fpga python -m unittest -v `
  FPGA_Project/software/qmap_prompt_demo/test_uart_board_acceptance.py
```

## Browser GUI Without Board Ethernet

Desktop Chrome or Edge can use Web Serial to talk directly to the board's
CH340 UART. The browser runs on the PC, so this GUI does not require GEM3 or a
network cable. Close Vitis Serial Monitor first because only one program can
own the COM port, then run:

```powershell
conda run -n llm_fpga python `
  FPGA_Project/software/qmap_prompt_demo/serve_web_serial_ui.py
```

Open `http://127.0.0.1:8000/`, click `连接串口`, and select the CH340 port. The
page enforces one in-flight request and unlocks the controls only after a
`DONE` or `ERROR` record. The browser performs no tokenization: text is sent
to the A53 and generated `BYTES` records are decoded only for display. UART and
generated-byte decoding are strict UTF-8; reconnect clears partial receive
state, and `UNDECODABLE` output is shown as an error instead of replacement
text. The GUI is still a display/demo client, not the physical PASS oracle.

## Automated Physical Acceptance

Close Vitis Serial Monitor and the Web GUI, because the strict tool must own the
CH340 COM port. With JTAG, UART, board power, and `hw_server` ready, run from the
verified workbench directory:

```powershell
conda run -n llm_fpga python `
  .\host_tools\run_uart_board_acceptance.py `
  --port COM230 `
  --launch-workbench .
```

The tool opens UART before reset/loading, requires the exact new `a_qgen`
startup sequence, then runs eight single-flight cases. It rejects stale logs,
missing command echoes, malformed or out-of-order records, incorrect BUSY
positions, wrong token/score/BYTES data, invalid UTF-8, nonzero launcher exit,
and unexpected PL-facing records after the out-of-range negative request. It
saves `uart_raw.bin`, `acceptance.json`, and `launcher.log`. Its timestamped
default output is outside the immutable workbench under the host temporary
directory; use `--output-dir` to choose a durable evidence directory. If the
launcher hangs, the Windows path first terminates the complete PowerShell/XSDB
process tree with `taskkill /T /F`, records the cleanup result, and fails closed;
its 21 host tests and a live parent/child cleanup check pass.

## Vitis and Board Integration Status

`create_vitis_workspace.py` now creates a third standalone A53 application,
`a_qgen`, without changing `a_qctl` or `a_qmdl`. The packaged XSDB launcher and
PowerShell wrapper accept `-Mode generate`, reuse the exact 61-segment PL-DDR
load and 281-header gate, and then start `a_qgen`.

On 2026-08-09 a clean Vitis 2025.1.1 workspace at `F:\vwi` built the platform,
both preserved smoke apps, and the text-capable `a_qgen` successfully. The ELF
is `4,050,704` bytes with SHA256
`F37EB88D4E1B75FEC815D01306EE85678C1E8555D02E5B42D3EFCA22FD337BBE`.
It embeds the deterministic 3,629,566-byte tokenizer asset with SHA256
`C20242603EF4144E3F3F2EC4BA97C0E9C315AADD41F1BD2C5740E2A7FFA03A7D`.
The launch keeps `runPsuInit=false` and `stopAtEntry=true`, and all durable
source hashes match the Vitis copies.

`make_prompt_demo_workbench.py` creates a non-destructive text/token workbench
that retains the 2026-08-08 validated hardware/runtime lineage but is
explicitly marked `WORKBENCH_NOT_RELEASE`. The next gate is physical UART
acceptance of both a general `TOKENS` request and a `PROMPT` request. A new
release should be promoted only after those board logs pass.

New workbenches use manifest format 5, pin every non-segment file from the
2026-08-08 board lineage, validate every segment against the pinned runtime
manifest, bind the exact trusted ELF/launcher/UI/acceptance provenance, execute
the packaged host tests and transcript replay, build in a temporary sibling
directory, and move into place only after verification. Formats 2 through 4
remain verifier-compatible. Verify the current workbench with:

```powershell
conda run -n llm_fpga python `
  FPGA_Project/software/qmap_prompt_demo/verify_prompt_demo_workbench.py `
  F:/qot_boardtest_prompt_text_v9_20260809
```

## First Board Acceptance Matrix

Run every command from a fresh software session at position zero. The first
two cases reuse the 2026-08-08 physical-board golden. The longer token/text and
vocabulary-boundary cases are current full28 Q4 fixed-point software goldens
and therefore need their first physical-board exact match; the text request is
also repeated immediately by the automated acceptance tool.

| Command | Exact generated token/score sequence | Purpose |
| --- | --- | --- |
| `TOKENS 2 1 374` | `28458/1227344433`, `64/1015661901` | Actual output feedback across two decode positions |
| `TOKENS 1 2 374 28458` | `64/1015661901` | Two-token prompt prefill and retained KV |
| `TOKENS 2 5 785 3853 315 89462 374` | `264/1296911292`, `26291/1225544557` | Multi-token text-derived prompt (`The future of FPGA is`) |
| `PROMPT 2 The future of FPGA is` | `PROMPT_IDS 5 785 3853 315 89462 374`, then the same `264/1296911292`, `26291/1225544557`; decoded ` a fascinating` | PS tokenizer + PL inference + PS detokenizer |
| `TOKENS 1 1 151935` | `28458/1224741478` | Highest legal model-vocabulary ID |

`TOKENS 1 1 151936` is the negative boundary case. It must return
`ERROR PARSE RANGE` before `START` and before any PL launch.
