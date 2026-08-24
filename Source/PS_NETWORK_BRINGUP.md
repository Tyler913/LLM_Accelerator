# PS Ethernet and Bare-Metal Web Bring-Up

Last updated: 2026-08-24

This runbook stages the user-facing path without changing the PL model math:

```text
a_net_echo -> ping/TCP echo -> a_qweb health page -> token-id generation API
           -> tokenizer/detokenizer -> text prompt and continuation UI
```

The accepted first target is a standalone A53/lwIP application. Linux remains
optional productization scope because PetaLinux, device-tree, boot-image,
driver, cache, and userspace integration are not required to re-prove the
already accepted network-to-accelerator boundary.

## Confirmed Hardware Route

- Use the board connector labelled `PS_ETH`, not `PL_ETH`.
- The MPSoC-P4 PS Ethernet PHY is a Motorcomm YT8521 connected to PS GEM3.
- RGMII uses MIO 64 through 75.
- GEM3 MDIO uses MIO 76 through 77.
- The vendor bare-metal example reports PHY address `7` and a 1000 Mb/s link.
- Dedicated PS MIO does not need PL package-pin constraints in the project
  XDC. The separate `PL_ETH` connector would require a different PL MAC/PHY
  design and is outside this bring-up.

Current checkpoint:

- Gate 0 is closed. The GEM3/MDIO/TTC0 configuration was applied, and a fresh
  no-simulation synthesis, implementation, bitstream, and include-bitstream XSA
  lineage is under `Temp/network_board_build_20260812_v1/`.
- The final routed timing is WNS `+0.208 ns` and WHS `+0.010 ns`; all 119,383
  routable nets are fully routed with zero routing error. The bitstream SHA256
  is `C926B3DB8021E976E5EA6CC2F71DA3C44899EA5C3DF61DA573475CE6D8C21239`,
  and the XSA SHA256 is
  `0AF1257442A68A6BEB31D94811713F2AE8E6AF63E0A85DD497146405E37406CF`.
- The network-XSA audit passes GEM3 RGMII MIO 64..75, MDIO MIO 76..77, TTC0,
  the 125 MHz IOPLL source, embedded-bitstream lineage, and all preserved
  QMAP/status/PL-DDR address checks.
- The original final clean `F:\vwk` Vitis workspace built the platform,
  patched-YT8521
  `a_net_echo`, and `a_qweb`; manifest SHA256
  `33CA1A825AFF72DD7A59C9938C0B0E838062C5C6E18EE1826432341E7A85E401`
  records build result zero for all three components.
- Gate 1 is closed on the physical board: YT8521 address 7/ID `0x0000011A`,
  1000-Mb/s full duplex, board `192.168.1.10`, ping `10/10`, and an exact
  75-byte port-7 echo. Its formal echo oracle is bound to the earlier `F:\vwc`
  build; the final `F:\vwk` QWEB startup independently re-proved the same PHY,
  link, and address.
- Gate 2 was closed by the original `F:\vwk` `final_v4` cold launch, two
  immediate strict
  HTTP jobs, and two live-browser jobs. Every accepted generation returns IDs
  `264,26291`, scores `1296911292,1225544557`, `MAX_NEW`, and exact 14-byte
  text ` a fascinating`; the final page remains `Board ready`.

## Portable Release Layout and Validation Boundary

The formal repository-contained board-hosted Web demo is
`lmdeploy/qwen3-0p6b-q4-qweb-demo/`:

```text
board/       network bitstream, XSA, FSBL, and a_qweb ELF
model/       runtime metadata and 61 downloaded Q4 binary segments
scripts/     relative-path Tcl launch and UART capture support
source/      final PS/PL source snapshot without tests or workspaces
run_demo.ps1
release_manifest.json
```

GitHub stores canonical source under `FPGA_Project/` and a clean source
snapshot, release manifests, wrapper, and selected final board artifacts under
`lmdeploy/`. Hugging Face repo
`Tyler01/qwen3-0p6b-fpga-q4-runtime` stores the 61
ignored `model/qwen3_runtime_*.bin` files (`394,547,200` bytes total). Restore
and audit a clone from its repository root before connecting to the board:

```powershell
conda run -n llm_fpga python init/download_q4_runtime.py
conda run -n llm_fpga python init/verify_q4_runtime.py
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\lmdeploy\qwen3-0p6b-q4-qweb-demo\run_demo.ps1 -AuditOnly
```

`run_demo.ps1` resolves every release input relative to `$PSScriptRoot`; it does
not require `F:\vwk` or `F:\qot_boardtest_prompt_text_v13_20260812`. Those
paths below record the original 2026-08-12/2026-08-17 physical acceptance and
are deliberately retained as provenance. The portable wrapper/copy currently
has local `-AuditOnly` evidence only. Hash equality proves that its board/model
inputs are the accepted bytes, but it does not prove that the new path and
wrapper have completed a physical launch. Use the following command for that
future revalidation after starting UART capture:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\lmdeploy\qwen3-0p6b-q4-qweb-demo\run_demo.ps1 `
  -EvidenceDirectory $qwebEvidence
```

Board-vendor references:

- Hardware overview: <https://blog.51cto.com/u_15046463/5900080>
- MPSoC-P4 standalone lwIP echo example:
  <https://blog.51cto.com/u_15046463/6107291>

## Gate 0: Review and Apply the Vivado PS Configuration - PASS

The following procedure is retained for reproducibility. The 2026-08-12
network lineage has already completed every step and passed the independent XSA
audit; reruns must use a new output directory rather than overwrite it.

The helper is:

`FPGA_Project/Vivado_Project/scripts/configure_ps_gem3_network.tcl`

It checks the exact project path/name/part, current block design, PS cell VLNV,
every property name, allowed current values, MIO 64..77 conflicts, IOPLL
source, and project/BD write protection. It does not guess a property name. It
never starts synthesis, implementation, bitstream generation, simulation,
output-product generation, or XSA export. Vivado 2025.1 does not expose a
reliable `IS_DIRTY` property for the current BD object, so the operator must
save or discard unrelated BD edits before using `--apply`.

In the existing Vivado GUI:

1. Open `FPGA_Project/Vivado_Project/LLM_FPGA.xpr`.
2. Open the `llm_system` block design.
3. In the Tcl Console, run the default audit:

   ```tcl
   source F:/LLM_Accelerator/FPGA_Project/Vivado_Project/scripts/configure_ps_gem3_network.tcl
   ```

4. Review every `current -> requested` line. The expected initial changes are
   GEM3 enable, RGMII `MIO 64 .. 75`, MDIO enable, MDIO
   `MIO 76 .. 77`, and TTC0 enable. The requested/actual GEM3 clock must be
   125 MHz.
5. Only after the audit passes and the BD is clean, apply and save explicitly:

   ```tcl
   llm_ps_net::run --apply
   ```

6. Reopen the BD and rerun the default audit. Every requested property should
   report `MATCH`.
7. Create a new hardware handoff lineage. Run a fresh no-simulation synthesis,
   implementation, and `write_bitstream`, then export a new include-bitstream
   XSA without overwriting the 2026-08-08 validated XSA. Do not use
   `--reuse-synth` or pair the changed PS handoff with an old routed artifact.
8. Before opening Vitis, run the read-only network-XSA gate:

   ```powershell
   conda run -n llm_fpga python `
     FPGA_Project/software/qmap_web_demo/audit_network_xsa.py `
     <new-network.xsa> --json
   ```

   It must pass the GEM3/MDIO/TTC0/clock checks and independently preserve the
   QMAP control aperture, DDR-status GPIO, and full PL-DDR aperture. The old
   2026-08-08 XSA is expected to fail only the network portion of this gate.

Do not proceed to Vitis if validation fails or the exported HWH still reports
GEM3/TTC0 disabled.

## Gate 1: `a_net_echo` - PASS

The original audited reference build is the short-path `F:\vwk` workspace with
components `p_net`, `a_net_echo`, and `a_qweb`; it does not mutate the proven
`F:\vwi` workspace. Use a different new short path for any reproduction because
the generator correctly rejects an existing output directory.

1. Build `p_net` from the new network-enabled XSA with an A53 standalone
   domain. The guarded generator is
   `FPGA_Project/software/qmap_web_demo/create_network_vitis_workspace.py`.
   Set `QWEB_NETWORK_XSA` and `QWEB_VITIS_WORKSPACE=F:\vwn`, run its conda
   `--check-only --json` path first, then run the same file with Vitis 2025.1
   embedded Python. Leave `QWEB_VITIS_BUILD_WEB` unset for this gate: the
   default creates only `p_net` and `a_net_echo`. It refuses the old non-network
   XSA, a missing embedded bitstream, protected-workspace overlap, or any
   existing output path. The execution path atomically claims a new workspace,
   snapshots the XSA by content hash, and audits that snapshot before and after
   the Vitis build.
2. Confirm the generated platform contains one `psu_ethernet_3`/`xemacps`
   instance and one `psu_ttc_0`/`xttcps` timer instance. Stop if either is
   absent; do not patch generated BSP headers by hand.
3. Require the generator's private EmbeddedSW repository and exact patched copy
   of the installed Vitis 2025.1 `lwip220_v1_2`; never modify the installed AMD
   tree. `yt8521_lwip220_patch.py` verifies the original source, stages the full
   library, adds exact YT8521 detection/configuration, and audits the BSP copy
   and exported archive. The accepted BSP PHY source SHA256 is
   `4900358C786793496501E0A19D0970E43AD70E378D909AB2C8D7458CC1BAC930`.
4. Connect a cable from `PS_ETH` to a router/switch, or directly to the PC.
   Keep the UART monitor open at 115200 8N1.
5. The AMD template enables DHCP. On a router/switch, use the address printed
   on UART. On a direct link with no DHCP server, its bounded fallback is board
   `192.168.1.10/24`; configure the PC as, for example,
   `192.168.1.20/24`. Use a unique locally administered MAC address; do not
   duplicate another device's MAC.
6. Run the app and record UART output for PHY ID/address, negotiated speed,
   link state, MAC address, and IP address.
7. From Windows, verify ICMP and TCP separately:

   ```powershell
   $boardIp = '<address printed on UART>'
   ping $boardIp
   Test-NetConnection $boardIp -Port 7
   ```

8. Use `ncat`, `telnet`, or a small TCP client to send bytes to port 7 and
   verify that exactly the same bytes return.

Acceptance for Gate 1:

- Stable 1000 Mb/s full-duplex link is reported. The current SDT clock path has
  not independently accepted 100/10 Mb/s operation.
- Exactly ten address-bound ping replies are observed.
- TCP port 7 accepts a unique nonempty payload and returns the exact bytes.
- UART remains usable for diagnostics.

The accepted Gate-1 report is
`Temp/network_echo_gate1_20260817_direct_v2/acceptance.json`, SHA256
`B7DFF37B2328B97242982A852A45EAC045C3AE3F5D90380F41F3969B040D9212`.
It records exact YT8521 address 7 and ID `0x0000011A`, 1000-Mb/s full duplex,
board `192.168.1.10`, ping `10/10`, and an exact 75-byte TCP echo whose TX/RX
SHA256 is
`E59A0DC34F0FF69AABDD4473C6139C771EEFDE0A31109715789149D370C444B3`.
This report is hash-bound to the earlier `F:\vwc` echo image, so it is the
accepted physical-port oracle rather than a claim that `F:\vwk` echo itself was
run. The `F:\vwk` `final_v4` QWEB UART later independently confirmed the same
PHY identity, resolved link, and IP before physical HTTP acceptance. The AMD
template's early banner mentions port 6001, but its actual `tcp_bind()` and this
acceptance gate use port 7.

### Gate 1 executable acceptance

Connect the `PS_ETH` cable before launching. The current BSP contains a link
recovery attempt, but its interrupt-context reset/wait path and DHCP address
refresh have not been accepted as stable hot-plug behavior. Close every other
program that owns the CH340 UART and inspect the physical adapters when using a
direct PC cable:

```powershell
Get-NetAdapter -Physical |
  Select-Object Name, InterfaceDescription, Status, LinkSpeed
```

The original Gate-1 acceptance flow remains one command. Its output directory
must not exist. It
opens and clears UART before it starts PowerShell/XSDB, pins the formal
workspace, launch scripts, network bit/XSA/FSBL/echo ELF, and Vitis 2025.1
execution files, then requires the ordered YT8521/link/IP/READY transcript,
ping `10/10`, and a byte-exact port-7 response:

```powershell
cd F:\LLM_Accelerator
conda run -n llm_fpga python -B `
  .\FPGA_Project\software\qmap_web_demo\run_network_echo_acceptance.py `
  --port COM230 `
  --workspace F:\vwk `
  --output-dir `
    F:\LLM_Accelerator\Temp\network_echo_gate1_rerun_01
```

Every attempt atomically writes `acceptance.json`, even for a failed serial,
launcher, link, ping, or TCP step. The same new directory also retains
`uart_raw.bin`, separate launcher logs, ping output, and exact TX/RX payloads.
Do not call Gate 1 PASS from a launcher message alone; require the final
`PASS Ethernet Gate 1` line and `passed=true` in that report.

With a DHCP router/switch, the tool uses the address printed by UART. For a
direct cable without DHCP, the image falls back to board
`192.168.1.10/24`; temporarily give the actual `Up` PC adapter an unused
address such as `192.168.1.20/24` with no gateway. The tool deliberately does
not change host network configuration. Do not configure a disconnected
adapter or assume its interface name.

The older two-terminal path remains useful only for diagnosis. Start
`capture_network_echo_uart.py` before `run_network_echo_board.ps1`, then inspect
the UART manually. That capture stops on the first PHY failure and does not by
itself prove ping or exact TCP echo. The launch inputs can also be audited
without XSDB or the board:

```powershell
& .\FPGA_Project\software\qmap_web_demo\run_network_echo_board.ps1 `
  -Workspace F:\vwk -AuditOnly
```

## Gate 2: `a_qweb` - PASS

The durable application is assembled, host-tested, built, and physically
accepted under `FPGA_Project/software/qmap_web_demo/`. It keeps lwIP in RAW mode
and uses an application-owned TCP adapter; the installed `lwip220_v1_2`
library does not link the optional application HTTPD implementation.

The original clean reference Web build exists in `F:\vwk`. To reproduce it,
create a different fresh workspace from the same audited XSA and explicitly
request the Web build:

```powershell
$env:QWEB_NETWORK_XSA = 'F:\path\to\new_network.xsa'
$env:QWEB_VITIS_WORKSPACE = 'F:\vwn_web'
$env:QWEB_VITIS_BUILD_WEB = '1'
$env:QWEB_TOKENIZER_ASSET = `
  'F:\LLM_Accelerator\Temp\qmap_prompt_demo_tokenizer\qwen3_tokenizer.qtk'
conda run -n llm_fpga python `
  .\FPGA_Project\software\qmap_web_demo\create_network_vitis_workspace.py `
  --with-web-app --check-only --json
vitis -s F:/LLM_Accelerator/FPGA_Project/software/qmap_web_demo/create_network_vitis_workspace.py
```

The builder creates `p_net`, keeps `a_net_echo` as an independent reference,
then stages the exact audited Web/session/tokenizer sources into `a_qweb`, sets
64 KiB stack and heap defaults, requires successful platform/application build
codes, audits the AArch64 ELF, and writes a provenance manifest. The original
`F:\vwk` manifest records platform, echo, and Web build result zero. Its
5,003,296-byte AArch64 `a_qweb.elf` has SHA256
`38A772F093CE3996177640863888B6700F1AC26F7BCB82E5F2159C0EF46F89DA`
and contains the required board, HTTP, session, and tokenizer symbols. The BSP
uses `MEMP_NUM_TCP_PCB=256`; all copied response bytes may be handed to lwIP and
gracefully closed without holding the single application slot for the final
ACK. The UI uses a 3000-ms generation poll and one-miss quiet-health grace.
A 128-connection diagnostic with distinct client source ports passed. A test
that deliberately rebinds the identical source port immediately after the
server's active close can still wait on lwIP TIME_WAIT; this did not occur in
the accepted browser flow. Persistent HTTP/keep-alive is optional future work
if a product must sustain aggressive connection churn.

The original accepted input lineage can still be audited directly without
touching XSDB or the board; this command is retained as historical provenance:

```powershell
& .\FPGA_Project\software\qmap_web_demo\run_qweb_board.ps1 `
  -Workspace F:\vwk `
  -RuntimeWorkbench F:\qot_boardtest_prompt_text_v13_20260812 `
  -AuditOnly
```

The accepted audit prints the pinned network manifest, bitstream, XSA, FSBL,
Web-ELF, and runtime-manifest hashes plus `61 segments / 394547200 bytes`.

For the formal portable release, use the download, verification, and
`run_demo.ps1 -AuditOnly` sequence under **Portable Release Layout and
Validation Boundary** instead of supplying `-Workspace` and
`-RuntimeWorkbench` paths.

### Gate 2 executable acceptance

#### Historical strict acceptance (canonical workspace)

The following pairing is the canonical flow used by the 2026-08-17 accepted
evidence. Start its canonical UART capture first; the output directory must not
exist:

```powershell
cd F:\LLM_Accelerator
$qwebEvidence = 'F:\LLM_Accelerator\Temp\qweb_board_rerun_01'
conda run -n llm_fpga python `
  .\FPGA_Project\software\qmap_web_demo\capture_qweb_uart.py `
  --port COM230 `
  --output-dir $qwebEvidence `
  --launch-json "$qwebEvidence\launch.json" `
  --timeout 3600
```

In terminal B, use the matching canonical launcher to reprogram the audited
network image, run FSBL, load all 61 runtime segments, check all 281 QMAP
headers, and start `a_qweb`:

```powershell
cd F:\LLM_Accelerator
$qwebEvidence = 'F:\LLM_Accelerator\Temp\qweb_board_rerun_01'
& .\FPGA_Project\software\qmap_web_demo\run_qweb_board.ps1 `
  -Workspace F:\vwk `
  -RuntimeWorkbench F:\qot_boardtest_prompt_text_v13_20260812 `
  -EvidenceDirectory $qwebEvidence
```

After terminal A reports `PASS QWEB UART startup`, the canonical
`run_qweb_http_acceptance.py` can consume that canonical `startup.json`. Run it
twice against the same live application. This proves an immediate repeated job
does not retain stale position, KV, token, score, or output state:

```powershell
conda run -n llm_fpga python `
  .\FPGA_Project\software\qmap_web_demo\run_qweb_http_acceptance.py `
  --startup-json "$qwebEvidence\startup.json" `
  --output-dir "$qwebEvidence\http_1"

conda run -n llm_fpga python `
  .\FPGA_Project\software\qmap_web_demo\run_qweb_http_acceptance.py `
  --startup-json "$qwebEvidence\startup.json" `
  --output-dir "$qwebEvidence\http_2"
```

The launcher must first preserve `launch.claim.json`, a hash-bound `xsdb.log`
with all ordered PASS markers, and a passing immutable `launch.json`; startup
must bind its UUID, full pinned XSDB execution chain, isolated startup
environment, wrapper/Tcl hashes, UART READY arrival time, and artifact hashes
before HTTP begins. Run this on a controlled one-board bench: JTAG, COM230, and
the connected PS Ethernet cable must belong to the same board because this
image does not yet publish a hardware-unique nonce.
Each HTTP run
must preserve `acceptance.json` and `output.bin`, observe an actual `running`
HTTP status during PL inference, and finish with exact
IDs `264,26291`, Q26 scores `1296911292,1225544557`, `MAX_NEW`, and raw UTF-8
` a fascinating`. The expected `output.bin` SHA256 is
`8E2B14930832293C41E4EFD76A667178A221082C8E254CE0FC01CF582CD8B55A`.

The accepted run is
`Temp/qweb_board_acceptance_20260817_final_v4/`. Its launch UUID is
`aca9c471-93f1-414f-8e04-b1a431bf39ea`; `launch.json` SHA256 is
`FC7A183366765BA16AA9B0B6B86C3D4A086C3DCDFD92E7EA9DD775FEB943CAD1`
and `startup.json` SHA256 is
`DB8CCD18E2E9B1C72AB2B40E2834C6FF78A159814DC0112C6E5EB409A5049D1D`.
The cold launch loaded 61/61 segments (394,547,200 bytes), checked 281/281 QMAP
headers, observed DDR status `0x5`, tokenizer metadata, YT8521 1000-full, and
`QWEB READY http://192.168.1.10:80/`. The two no-cooldown strict reports are
`http_pair_1/acceptance.json` SHA256
`7CE1716AFCD28221E38F9485804F67C94211D6B9092F084D5FEE6B0FACA5B5A1`
and `http_pair_2/acceptance.json` SHA256
`E8ACDCCE6AFFF467A93404F3B8F724B6A2D8E59470319C7C298D0A7F65DEECDB`.

Both observed an actual `running` response and finished with the exact values
above. Live Edge was observed to complete Jobs 3 and 4. The retained
`browser_final.png` independently proves Job 4 `Board ready / done / MAX_NEW`
and has SHA256
`D22D9D153B52940220143A88A5029868C6BC998573386F5CCDFE8C9AF711D9FC`,
while a live console check returned zero messages.

Open the exact live GUI URL from the UART-bound report for the final demo:

```powershell
$startup = Get-Content -Raw "$qwebEvidence\startup.json" | ConvertFrom-Json
Start-Process $startup.network.url
```

For a fresh reproduction, exercise it in this order:

1. Run `a_qweb`, open the exact `QWEB READY http://<actual-ip>:80/` URL printed
   on UART, and verify the generated static page plus `GET /api/health` without
   touching the PL. The Host allowlist is derived from the actual DHCP/static
   interface address, not hard-coded to the fallback IP.
2. Connect the existing bounded `POST /api/generate` route, first with a
   token-id request such as:

   ```json
   {"tokens":[374],"max_new_tokens":2}
   ```

3. Poll the returned job through `GET /api/generate/<id>` and retrieve arbitrary
   raw detokenized bytes through `GET /api/generate/<id>/output`. Confirm the
   already wired `qweb_job_step()`/QMAP runtime path advances outside the TCP
   callback and that the browser remains responsive throughout the PL run.
4. After token-ID hardware acceptance, exercise the already integrated exact
   tokenizer/detokenizer with the standard text prompt.
5. Verify the already implemented EOS/IM_END, length/context, malformed-input,
   error, and single-active-job behavior on the physical network path.

The implemented TCP callback only parses/copies a bounded request and returns
HTTP 202 with a job ID. The main loop advances at most one session step, while
the injected board runner services `xemacif_input`, adapter deadlines/TX, and
TTC fast/slow timer flags at bounded intervals inside every long single-token
PL poll. The job rejects a null runner, so the non-pumping default loop cannot
be selected accidentally. These properties pass host mocks and the final-v4
physical responsiveness checks during full28 execution.

Keep request/response buffers fixed and bounded. Reject oversized input before
tokenization, allow only one active job, and preserve UART logging for every
PL error/status transition.

Accepted Gate-2 observations and retained defensive gates:

- The page and idle/final `/api/health` requests succeed; job-status requests
  return `running` throughout each long PL inference, proving that the network
  pump remains active during computation.
- Two strict text-prompt jobs and two live-browser text-prompt jobs launch the
  validated PL path and return the exact expected continuation.
- Immediate repeated requests do not retain stale status, token, score,
  position, output, or KV-cache state.
- Invalid input and injected PL-error behavior remain bounded by the host C and
  Python suites; those fault cases were not separately injected into the
  accepted `final_v4` physical run.

#### Portable release revalidation (new path, pending physical run)

The release under `lmdeploy/qwen3-0p6b-q4-qweb-demo/` deliberately has its own
relative-path launcher and UART capture schema. Follow that package's
`README.md`: run `scripts/capture_qweb_uart.py` in terminal A, then
`run_demo.ps1 -EvidenceDirectory ...` in terminal B. The two reports bind one
capture ID, capture PID/live heartbeat, launch UUID, timestamps, toolchain,
artifacts, all 61 runtime segments, and the observed READY URL.

Do **not** pass the portable `startup.json` to the canonical
`run_qweb_http_acceptance.py`: that strict oracle validates the older canonical
launcher/capture schema and will correctly reject the portable schema. The
portable package currently has local `-AuditOnly` evidence only; its physical
launcher/UART revalidation and any new-path HTTP evidence remain pending. This
does not change the accepted 2026-08-17 canonical Gate 2 result above.

## Product Boundary After Gate 2

The UART CLI, PC Web Serial path, and preferred board-hosted Web UI now close
the physical arbitrary-prompt PS-to-PL-to-PS boundary for the JTAG-loaded first
version. Two immediate strict HTTP runs plus two live-browser runs prove the
exact PS tokenize -> PL full28 -> PS detokenize -> Web output chain. The
requested bench-demo project is therefore complete. A boot image, SD/QSPI
storage, autonomous runtime/weight loading, production link/DHCP recovery,
persistent or multi-client HTTP, and performance tuning are separate optional
power-on productization scope.
