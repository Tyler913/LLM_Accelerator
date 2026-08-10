# QMAP Standalone Web Demo Core

This directory contains the transport-independent, allocation-free request and
response core for the future standalone A53/lwIP `a_qweb` application. It does
not yet contain a lwIP adapter or board entry point.

## Files

- `qweb_http.[ch]`: bounded incremental HTTP/1.1 parser and complete-response
  formatter. The parser accepts fragmented headers and bodies, rejects transfer
  encoding and ambiguous content lengths, requires one `Host` header, and stops
  consuming bytes exactly at the end of one request.
- `qweb_api.[ch]`: strict JSON parser for `POST /api/generate`, a fixed-capacity
  `qweb_generate_request_t`, and bounded JSON writing helpers.
- `qweb_job.[ch]`: one allocation-free generation job. It tokenizes text with
  the exact Qwen asset, serially prefills the prompt, feeds each actual argmax
  result into the next position, and accumulates token IDs, scores, and raw
  detokenized bytes.
- `qweb_router.[ch]`: bounded routes for health, job submission, job status,
  and raw output bytes. The request workspace lives in the caller-owned router
  object so JSON decoding does not consume the small BSP stack.
- `test_qweb_core.c`: host tests for fragmentation, protocol limits, malformed
  requests, Unicode/escape handling, schema/range checks, response formatting,
  and the complete HTTP-to-JSON boundary.
- `test_qweb_job_host.c`: host test for exact prompt tokenization, actual token
  feedback, detokenization, busy rejection, range errors, and PL error
  propagation using the real tokenizer asset and an injected PL runner.
- `test_qweb_router_host.c`: host tests for routing, media types, busy and
  malformed requests, job status bounds, raw arbitrary-byte output, and the
  incremental HTTP-parser-to-job boundary.
- `run_host_tests.ps1`: strict GCC build and test runner using a disposable
  system-temporary directory. It pins the tokenizer asset by SHA-256 before
  running all three test executables and rejects any function whose optimized
  product-source stack frame exceeds 1 KiB (host test fixtures may use larger
  local objects).
- `audit_network_xsa.py`: read-only XSA/HWH gate for GEM3, MDIO, TTC0, clock,
  embedded provenance, and the preserved QMAP/status/PL-DDR address maps.
- `create_network_vitis_workspace.py`: delayed-import Vitis generator for an
  isolated `F:\vwn` workspace containing `p_net` and the AMD
  `lwip_echo_server` template as `a_net_echo`. It refuses a failed XSA audit,
  a missing embedded bitstream, source/proven-workspace overlap, or any
  existing output path; execution atomically claims a new workspace and passes Vitis a
  content-addressed XSA snapshot that is audited before and after the build.
- `test_audit_network_xsa.py` and
  `test_create_network_vitis_workspace.py`: synthetic fail-closed XSA/path/API
  tests that run without importing Vitis.

The accepted generation bodies are exactly one of:

```json
{"prompt":"The future of FPGA is","max_new_tokens":2}
```

```json
{"tokens":[785,3853,315,89462,374],"max_new_tokens":2}
```

Unknown or duplicate members are rejected. Prompt bytes are strict UTF-8 after
JSON escape decoding and are always length-delimited; callers must not use
`strlen()` on `request.prompt`. Token arrays and generation lengths are bounded
to the current 256-position model context, and token IDs must be below 151936.
The demo retains one job record: DONE/ERROR remains queryable until the next
successful submission replaces it; rejected requests leave that record intact.
The first Ethernet version therefore permits one UI client and one bounded
connection/job at a time.

Run the host-only checks from PowerShell:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\FPGA_Project\software\qmap_web_demo\run_host_tests.ps1
```

Use `-TokenizerAsset <path>` if the generated QTK asset is stored outside the
default `Temp/qmap_prompt_demo_tokenizer/qwen3_tokenizer.qtk` path. The runner
still requires the pinned, verified asset content.

Run the Python gates with the repository environment:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
conda run -n llm_fpga python -m unittest discover `
  -s .\FPGA_Project\software\qmap_web_demo -p 'test_*.py' -v
```

After a new network XSA exists, audit the exact planned workspace without
creating it:

```powershell
$env:QWEB_NETWORK_XSA = 'F:\path\to\new_network.xsa'
$env:QWEB_VITIS_WORKSPACE = 'F:\vwn'
conda run -n llm_fpga python `
  .\FPGA_Project\software\qmap_web_demo\create_network_vitis_workspace.py `
  --check-only --json
```

Only after that passes, run the same script with Vitis 2025.1 embedded Python
(`vitis -s ...create_network_vitis_workspace.py`). This builds only the new
network platform and echo app; it does not touch `F:\vwi` or build `a_qweb`.

The future raw-lwIP adapter should feed each received pbuf fragment into
`qweb_http_parser_feed()`, copy a completed `qweb_generate_request_t` into the
single-job state machine, and immediately return an HTTP 202 response. It must
not run a complete PL inference from inside a TCP receive callback. The
mandatory injected Web runner must also service `xemacif_input()` and the TTC
timer flags at bounded intervals during each long PL token poll; the Web job
rejects a null runner so the non-pumping default loop cannot be selected.
