# QMAP Standalone Web Demo

This directory contains the application-owned, allocation-free standalone
A53/lwIP `a_qweb` implementation. The HTTP/session core, raw-lwIP adapter,
board entry assembly, generated offline Web UI, network-XSA audit, and isolated
Vitis workspace builder all pass host checks. They have not yet been built from
a network-enabled XSA or run on the physical board, so this is not an Ethernet
hardware PASS.

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
- `qweb_lwip_adapter.[ch]`: one-connection raw-lwIP adapter with fragmented
  pbuf input, bounded request framing, exact Host authorities, incremental TX,
  request/connection deadlines, and static-asset routing. It returns HTTP 202
  before inference and never runs a complete model command in a TCP callback.
- `qweb_board_app.[ch]`: cooperative board loop and injected QMAP runner. The
  runner services lwIP input and TTC fast/slow timers at bounded intervals
  during every long PL token poll and validates the full layer/done/error
  result before returning it to the session.
- `qweb_board_entry.[ch]`: replacement for the AMD echo template's `echo.c`.
  It checks PL-DDR readiness and runtime sentinels, initializes the exact
  tokenizer/session/router/adapter chain, derives the accepted Host values and
  printed URL from the interface's actual DHCP/static IPv4 address, and serves
  port 80.
- `web/`, `web_assets_generator.py`, and `web_assets.[ch]`: self-contained
  responsive browser UI and deterministic ROM C table. The generator composes
  `/` and `/index.html` with the exact current CSS and JavaScript bytes inline,
  so the initial browser document needs only one TCP connection; exact
  `/styles.css` and `/app.js` routes remain available independently. The UI
  supports text or token IDs, defaults to two generated tokens, polls the
  asynchronous job, fetches cumulative raw output bytes, and renders their
  UTF-8 view without any CDN or external dependency.
- `test_qweb_core.c`: host tests for fragmentation, protocol limits, malformed
  requests, Unicode/escape handling, schema/range checks, response formatting,
  and the complete HTTP-to-JSON boundary.
- `test_qweb_job_host.c`: host test for exact prompt tokenization, actual token
  feedback, detokenization, busy rejection, range errors, and PL error
  propagation using the real tokenizer asset and an injected PL runner.
- `test_qweb_router_host.c`: host tests for routing, media types, busy and
  malformed requests, job status bounds, raw arbitrary-byte output, and the
  incremental HTTP-parser-to-job boundary.
- `test_qweb_lwip_adapter_host.c` and `test_qweb_board_entry_host.c`: raw-lwIP
  callback/TX/timeout tests and final AMD-template entry assembly tests,
  including DHCP-derived Host authorities and fail-closed startup.
- `test_web_assets.py`, `web_assets_host_test.c`, and
  `web_assets_ui_test.js`: deterministic generator, exact C lookup, one-document
  boot, browser logic, UTF-8 streaming, and injection-safety checks. The current
  three-source asset set has SHA256
  `D2FC8E450BEF24ABED3E44B3D05402972FF7AC33F813EC8F289ABE6725457090`.
- `run_host_tests.ps1`: strict GCC build and test runner using a disposable
  system-temporary directory. It pins the tokenizer asset by SHA-256 before
  running all five C test executables and rejects any function whose optimized
  product-source stack frame exceeds 1 KiB (host test fixtures may use larger
  local objects).
- `audit_network_xsa.py`: read-only XSA/HWH gate for the exact XCZU2EG
  part/top, `xsa.json`, parseable FULL_BIT/header binding, GEM3, MDIO, TTC0,
  clock, and the preserved QMAP/status/PL-DDR address maps.
- `create_network_vitis_workspace.py`: delayed-import Vitis generator for an
  isolated `F:\vwn` workspace containing `p_net` and the AMD
  `lwip_echo_server` template as `a_net_echo`. With explicit `--with-web-app`
  or `QWEB_VITIS_BUILD_WEB=1`, it also stages the exact audited sources and
  pinned tokenizer into `a_qweb`, patches the real AMD CMake source collector,
  raises its standalone stack/heap defaults, builds it, and verifies the full
  ELF structure, executable sections, required Web/session/tokenizer symbols,
  and exact tokenizer span. The real CLI path also pins the AMD Vitis 2025.1.1
  template inputs. It refuses a
  failed XSA audit, missing embedded bitstream, source/proven-workspace overlap,
  source drift, failed/in-progress build status, or any existing output path;
  execution atomically claims a new workspace and passes Vitis a
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
connection/job at a time. Its generated boot document is intentionally
self-contained so a browser does not need concurrent CSS or JavaScript fetches.

Run the host-only checks from PowerShell:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\FPGA_Project\software\qmap_web_demo\run_host_tests.ps1
```

Use `-TokenizerAsset <path>` if the generated QTK asset is stored outside the
default `Temp/qmap_prompt_demo_tokenizer/qwen3_tokenizer.qtk` path. The runner
still requires the pinned, verified asset content.

Run the 50 Python XSA/workspace/UI-asset gates with the repository environment:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
conda run -n llm_fpga python -m unittest discover `
  -s .\FPGA_Project\software\qmap_web_demo -p 'test_*.py' -v
```

The deterministic asset-only check is also available directly:

```powershell
conda run -n llm_fpga python -B `
  .\FPGA_Project\software\qmap_web_demo\web_assets_generator.py --check
```

After a new network XSA exists, audit the exact echo-plus-Web workspace without
creating it:

```powershell
$env:QWEB_NETWORK_XSA = 'F:\path\to\new_network.xsa'
$env:QWEB_VITIS_WORKSPACE = 'F:\vwn'
$env:QWEB_TOKENIZER_ASSET = `
  'F:\LLM_Accelerator\Temp\qmap_prompt_demo_tokenizer\qwen3_tokenizer.qtk'
conda run -n llm_fpga python `
  .\FPGA_Project\software\qmap_web_demo\create_network_vitis_workspace.py `
  --with-web-app --check-only --json
```

Only after that passes, run the same script with Vitis 2025.1 embedded Python
with `QWEB_VITIS_BUILD_WEB=1`, for example:

```powershell
$env:QWEB_VITIS_BUILD_WEB = '1'
vitis -s F:/LLM_Accelerator/FPGA_Project/software/qmap_web_demo/create_network_vitis_workspace.py
```

The default without that environment variable deliberately builds only
`p_net` and `a_net_echo` for the independent Ethernet gate. The explicit Web
mode adds `a_qweb`; neither mode touches the proven `F:\vwi` workspace. The
current 2026-08-08 XSA is expected to fail this flow because GEM3/MDIO/TTC0 are
still disabled. Do not weaken that rejection or treat the host tests as a
substitute for `a_net_echo` and `a_qweb` board acceptance.
