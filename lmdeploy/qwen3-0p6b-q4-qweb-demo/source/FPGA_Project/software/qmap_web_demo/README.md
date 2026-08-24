# QMAP Standalone Web Demo

This directory contains the application-owned, allocation-free standalone
A53/lwIP `a_qweb` implementation. The HTTP/session core, raw-lwIP adapter,
board entry assembly, generated offline Web UI, network-XSA audit, and isolated
Vitis workspace builder all pass host checks. A network-enabled Vivado/XSA
lineage and the real patched-lwIP `a_net_echo`/`a_qweb` Vitis builds also pass.
Ethernet Gate 1 now passes with YT8521 address 7/ID `0x0000011A`, 1000-Mb/s
full duplex, ping `10/10`, and exact port-7 echo. The final `F:\vwk` QWEB image
then passes one cold runtime load, two immediate strict HTTP jobs, and two real
browser jobs through the exact PS tokenize -> PL full28 -> PS detokenize chain.
Gate 2 is closed and the JTAG-loaded first-version Web demo is complete.

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
  request/connection deadlines, and static-asset routing. Every response byte
  is copied into lwIP; once all bytes are queued, the adapter gracefully closes
  and releases its single application slot without waiting for the final ACK.
  Retired PCBs use `TCP_PRIO_MIN`; an explicit `ERR_MEM` restores callbacks,
  state, and priority for retry. It returns HTTP 202 before inference and never
  runs a complete model command in a TCP callback.
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
  UTF-8 view without any CDN or external dependency. Generation polling is
  3000 ms; quiet health checks and submission are serialized, and one isolated
  quiet-health transport miss preserves the prior ready state and retries after
  500 ms before a second consecutive failure marks the board offline.
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
  `16ECE514A1489EFB3C4BC959BC983FA9229025AB412504F4349EDF9B8A2E8116`.
- `run_host_tests.ps1`: strict GCC build and test runner using a disposable
  system-temporary directory. It pins the tokenizer asset by SHA-256 before
  running all five C test executables and rejects any function whose optimized
  product-source stack frame exceeds 1 KiB (host test fixtures may use larger
  local objects).
- `audit_network_xsa.py`: read-only XSA/HWH gate for the exact XCZU2EG
  part/top, `xsa.json`, parseable FULL_BIT/header binding, GEM3, MDIO, TTC0,
  clock, and the preserved QMAP/status/PL-DDR address maps.
- `yt8521_lwip220_patch.py`: pins the installed AMD `lwip220_v1_2` source,
  stages a private EmbeddedSW repository, and adds exact Motorcomm YT8521
  identification, RGMII delay setup, and link-speed resolution without
  modifying the Vitis installation. `test_yt8521_lwip220_patch.py` checks the
  deterministic source/tree hashes and fail-closed library identity.
- `scan_network_phy.tcl`: bounded MDIO diagnostic that established physical
  PHY address 7 and ID `0x0000011A` without using the stock unknown/Marvell
  fallback.
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
- `launch_network_echo.tcl`, `run_network_echo_board.ps1`, and
  `capture_network_echo_uart.py`: manifest-audited echo launch plus the older
  two-terminal diagnostic capture for the independent Ethernet Gate 1.
- `run_network_echo_acceptance.py`: preferred one-command physical Gate 1.
  It opens and clears UART before launching the audited image, requires the
  exact YT8521 address/ID, ordered 1-Gbps full-duplex link/IP/port-7 records,
  ten address-bound ping replies, and a unique byte-exact TCP echo. Every live
  attempt claims a fresh directory and atomically preserves success or failure
  as `acceptance.json` plus raw UART, launcher, ping, TX, and RX evidence.
- `launch_qweb_board.tcl` and `run_qweb_board.ps1`: audit the formal network
  manifest, patched BSP/archive, final AArch64 Web ELF, board-accepted v13
  runtime, all 61 segment hashes/addresses, the exact cache-bypass loader, and
  all 281 QMAP headers before starting `a_qweb`. `-AuditOnly` performs the full
  artifact audit without touching XSDB or the board. A physical run requires
  the active UART evidence directory, atomically creates a pre-launch claim,
  pins the complete Vitis 2025.1 XSDB execution chain (`xsdb.bat`, `loader.bat`,
  `setupEnv.bat`, `rdiArgs.bat`, `xsdb.exe`, and its manifest), and runs it with
  `-no-ini` from a fresh isolated home/current directory after removing all
  inherited Xilinx/RDI/Tcl path overrides. It preserves the ordered PASS
  transcript in `xsdb.log` and writes one immutable `launch.json` with a UUID,
  the exact wrapper/Tcl hashes, argv/environment policy, and pinned artifacts.
- `capture_qweb_uart.py`: captures one ordered live startup and binds the exact
  YT8521 identity/resolved link, board IP, PL-DDR status, tokenizer metadata,
  `QWEB READY` URL, and same-run `launch.json` into raw UART plus JSON evidence.
  The arrival time of the UART READY bytes must be after this launch began;
  an old complete startup banner is therefore not accepted as same-run proof.
  Its default 3600-second timeout includes the measured cold JTAG runtime load;
  serial/open/framing failures still produce a structured, hash-bound report.
- `run_qweb_http_acceptance.py`: accepts only a passing live UART report, then
  reparses its raw UART and launch report, enforces an ordered same-run/two-hour
  evidence window, and verifies the exact ROM page/ETag, health, prompt
  submission, monotonic job status, an actual `running` response during PL
  inference, token IDs, Q26 scores, `MAX_NEW`, and exact binary output
  ` a fascinating`. The three corresponding Python test modules cover
  launcher, UART, mock-HTTP, and evidence failures.

Physical launch binding assumes a controlled bench with one target board: the
JTAG cable, `COM230`, and the connected PS Ethernet port must all belong to that
same board. The current image does not expose a board-unique nonce on UART.

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

Run all 110 Python XSA/workspace/PHY/launcher/UART/HTTP/UI gates with the
repository environment:

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

The final formal build is `F:\vwk\network_workspace_manifest.json`. It
records successful `p_net`, patched-YT8521 `a_net_echo`, and `a_qweb` builds;
manifest SHA256 is
`33CA1A825AFF72DD7A59C9938C0B0E838062C5C6E18EE1826432341E7A85E401`.
The Web ELF is 5,003,296 bytes with SHA256
`38A772F093CE3996177640863888B6700F1AC26F7BCB82E5F2159C0EF46F89DA`;
the BSP uses RAW API, 256 TCP PCBs, and pbuf pool 2048.
Before reproducing it, choose a different short workspace path that does not
exist and audit the exact network XSA/build plan without creating that path:

```powershell
$env:QWEB_NETWORK_XSA = `
  'F:\LLM_Accelerator\Temp\network_board_build_20260812_v1\llm_system_qwen3_one_token_boardready.xsa'
$env:QWEB_VITIS_WORKSPACE = 'F:\vwn_rebuild'
$env:QWEB_TOKENIZER_ASSET = `
  'F:\LLM_Accelerator\Temp\qmap_prompt_demo_tokenizer\qwen3_tokenizer.qtk'
$env:QWEB_VITIS_LWIP220_SOURCE = `
  'D:\Applications\Vivado_2025.1.1\2025.1.1\data\embeddedsw\ThirdParty\sw_services\lwip220_v1_2'
conda run -n llm_fpga python `
  .\FPGA_Project\software\qmap_web_demo\create_network_vitis_workspace.py `
  --with-web-app --check-only --json
```

Only after that passes, run the same script with Vitis 2025.1 embedded Python
and explicit Web mode:

```powershell
$env:QWEB_VITIS_BUILD_WEB = '1'
& 'D:\Applications\Vivado_2025.1.1\2025.1.1\Vitis\bin\vitis.bat' -s `
  F:/LLM_Accelerator/FPGA_Project/software/qmap_web_demo/create_network_vitis_workspace.py
```

The default without that environment variable deliberately builds only
`p_net` and `a_net_echo` for the independent Ethernet gate. The explicit Web
mode adds `a_qweb`; neither mode touches the proven `F:\vwi` workspace. The
historical 2026-08-08 XSA is correctly rejected because its GEM3/MDIO/TTC0 are
disabled, while the 2026-08-12 network XSA passes. Do not weaken either gate or
treat a build as physical Ethernet/Web acceptance.

Gate 1 is accepted. Its immutable physical-port oracle is
`Temp/network_echo_gate1_20260817_direct_v2/acceptance.json`, SHA256
`B7DFF37B2328B97242982A852A45EAC045C3AE3F5D90380F41F3969B040D9212`.
It records PHY 7/ID `0x0000011A`, 1000-Mb/s full duplex, board
`192.168.1.10`, ping `10/10`, and an exact 75-byte port-7 echo. The oracle is
hash-bound to the earlier `F:\vwc` echo image; final `F:\vwk` QWEB startup
independently re-proved the same PHY/link/IP. For a fresh reproduction, connect
`PS_ETH` before boot and use a new output directory:

```powershell
conda run -n llm_fpga python -B `
  .\FPGA_Project\software\qmap_web_demo\run_network_echo_acceptance.py `
  --port COM230 `
  --workspace F:\vwk `
  --output-dir `
    F:\LLM_Accelerator\Temp\network_echo_gate1_rerun_01
```

Do not insert the cable after launch: the current BSP has not closed its
hot-plug timing/state-machine boundary. A PASS requires PHY 7 at 1 Gbps full
duplex, a usable IPv4 address, ping `10/10`, and an exact TCP echo on port 7.
If a direct PC-to-board link falls back to `192.168.1.10`, first assign the
connected PC adapter an unused address such as `192.168.1.20/24`; the tool does
not modify host network configuration.

Audit the exact formal Web launch lineage without touching XSDB or the board:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\FPGA_Project\software\qmap_web_demo\run_qweb_board.ps1 `
  -Workspace F:\vwk `
  -RuntimeWorkbench F:\qot_boardtest_prompt_text_v13_20260812 `
  -AuditOnly
```

This must report the pinned network manifest/bit/XSA/FSBL/Web-ELF hashes and
`61 segments / 394547200 bytes`.

Gate 2 is accepted under
`Temp/qweb_board_acceptance_20260817_final_v4/`. Launch UUID
`aca9c471-93f1-414f-8e04-b1a431bf39ea` loaded all 61 runtime segments, checked
all 281 QMAP headers, and reached DDR `0x5`, YT8521 1000-full, tokenizer-ready,
and `QWEB READY http://192.168.1.10:80/`. Its `launch.json` SHA256 is
`FC7A183366765BA16AA9B0B6B86C3D4A086C3DCDFD92E7EA9DD775FEB943CAD1`;
`startup.json` SHA256 is
`DB8CCD18E2E9B1C72AB2B40E2834C6FF78A159814DC0112C6E5EB409A5049D1D`.
Two no-cooldown strict HTTP reports, SHA256
`7CE1716AFCD28221E38F9485804F67C94211D6B9092F084D5FEE6B0FACA5B5A1`
and `E8ACDCCE6AFFF467A93404F3B8F724B6A2D8E59470319C7C298D0A7F65DEECDB`,
both observed `running` and finished with IDs `264,26291`, scores
`1296911292,1225544557`, `MAX_NEW`, and exact ` a fascinating`. Live Edge was
observed to complete Jobs 3 and 4. The retained `browser_final.png`
independently proves Job 4 `Board ready / done / MAX_NEW` with SHA256
`D22D9D153B52940220143A88A5029868C6BC998573386F5CCDFE8C9AF711D9FC`;
a live console check returned zero messages.

This closes the requested JTAG-loaded first-version demo. BOOT.BIN/SD/QSPI
boot, automatic runtime/weight loading, production link lifecycle, persistent
or multi-client HTTP, and performance tuning are optional productization scope.
