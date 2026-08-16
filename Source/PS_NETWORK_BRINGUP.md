# PS Ethernet and Bare-Metal Web Bring-Up

Last updated: 2026-08-12

This runbook stages the user-facing path without changing the PL model math:

```text
a_net_echo -> ping/TCP echo -> a_qweb health page -> token-id generation API
           -> tokenizer/detokenizer -> text prompt and continuation UI
```

The first target is a standalone A53/lwIP application. Linux is intentionally
deferred: it would add PetaLinux, device-tree, boot-image, driver, cache, and
userspace integration before the network-to-accelerator boundary is proven.

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
- The clean `F:\vwc` Vitis workspace builds the platform, patched-YT8521
  `a_net_echo`, and `a_qweb`; its provenance manifest records build result zero
  for all three components.
- The patched echo image physically detects Motorcomm YT8521 at PHY address 7
  with ID `0x0000011A`, but auto-negotiation times out with the link and
  AN-complete bits clear. Both Windows wired adapters were `Disconnected` at
  that point. Gate 1 link/IP/ping/TCP acceptance and all physical `a_qweb`
  acceptance therefore remain open.

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

## Gate 1: `a_net_echo` - Physical Link Pending

The audited reference build is the short-path `F:\vwc` workspace with
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

- Stable 1000/100/10 Mb/s link is reported.
- Repeated ping succeeds without intermittent loss.
- TCP port 7 accepts a connection and echoes payload bytes.
- UART remains usable for diagnostics.

Current Gate 1 evidence is a PHY-detection PASS but a link failure. The audited
echo ELF SHA256 is
`93192F0D37741604036F1B502CC7BAE257EE8E8151103F3F09415C592FBA5C67`.
Its physical UART capture reports exact YT8521 address 7 and ID `0x0000011A`,
then an auto-negotiation timeout with `bmsr_latch=0x7949` and
`bmsr_now=0x7949`; link and AN-complete are clear. At the same time both PC
wired adapters were `Disconnected`. Establish an electrically connected host
or router path before changing PHY software again. The AMD template's early
banner mentions port 6001, but its actual `tcp_bind()` and this acceptance gate
use port 7. Do not start physical Web acceptance on an unstable link.

## Gate 2: `a_qweb` - Build PASS, Physical Acceptance Pending

The durable application is assembled, host-tested, and built under
`FPGA_Project/software/qmap_web_demo/`, but do not run it on the board until
Gate 1 proves the exact PHY/link/platform lineage. It keeps lwIP in RAW mode
and uses an application-owned TCP adapter; the installed `lwip220_v1_2`
library does not link the optional application HTTPD implementation.

The clean reference Web build already exists in `F:\vwc`. To reproduce it,
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
codes, audits the AArch64 ELF, and writes a provenance manifest. The real
`F:\vwc` manifest records platform, echo, and Web build result zero. Its
5,002,800-byte AArch64 `a_qweb.elf` has SHA256
`3B83026EC7647A79D6DF2D9FE584A2555157E5D87A212F7ADB478EBD036E8650`
and contains the required board, HTTP, session, and tokenizer symbols. This is
a build PASS only; no physical HTTP request has reached this ELF yet.

Bring it up in this order:

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
be selected accidentally. These properties pass host mocks but still require
physical responsiveness checks during full28 execution.

Keep request/response buffers fixed and bounded. Reject oversized input before
tokenization, allow only one active job, and preserve UART logging for every
PL error/status transition.

Acceptance for Gate 2:

- The browser page and `/api/health` remain responsive during a PL run.
- The token-id request launches the same validated PL path and returns the
  expected known smoke outputs before arbitrary text is attempted.
- Repeated requests do not retain stale status, token, score, position, or KV
  cache state.
- An invalid request or PL error produces a bounded HTTP error and a useful
  UART diagnostic instead of hanging the network stack.

## Product Boundary After Gate 2

The UART CLI and PC Web Serial path already close the physical arbitrary-prompt
PS-to-PL-to-PS boundary for the JTAG-loaded first version. Passing `a_qweb` with
token IDs will prove Ethernet-to-PS-to-PL-to-PS transport, and a repeated exact
text request plus bounded error recovery will close the preferred board-hosted
Web interface. Gate 1 link/ping/TCP and Gate 2 physical HTTP acceptance remain
open. A boot image, persistent storage, and autonomous runtime/weight loading
are a separate power-on productization boundary rather than part of the
existing bench-demo PASS.
