# PS Ethernet and Bare-Metal Web Bring-Up

Last updated: 2026-08-09

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

Current local evidence before configuration:

- `FPGA_Project/Vivado_Project/LLM_FPGA.srcs/sources_1/bd/llm_system/ip/llm_system_zynq_ultra_ps_e_0_0/llm_system_zynq_ultra_ps_e_0_0.xci`
  records GEM3, GEM3 MDIO, and TTC0 disabled.
- The same XCI records the GEM3 requested and actual reference frequencies as
  125 MHz, sourced from IOPLL.
- `FPGA_Project/Vivado_Project/LLM_FPGA.gen/sources_1/bd/llm_system/hw_handoff/llm_system.hwh`
  confirms the same exported state.
- The final 2026-08-08 XSA therefore does not yet expose a usable GEM3/TTC0
  platform to Vitis.

Board-vendor references:

- Hardware overview: <https://blog.51cto.com/u_15046463/5900080>
- MPSoC-P4 standalone lwIP echo example:
  <https://blog.51cto.com/u_15046463/6107291>

## Gate 0: Review and Apply the Vivado PS Configuration

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
7. Create a new hardware handoff lineage. Regenerate the required Vivado
   products and export a new XSA without overwriting the 2026-08-08 validated
   XSA. Record whether a newly generated or deliberately reused routed
   bitstream is embedded; keep the XSA and bitstream lineage unambiguous.
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

## Gate 1: `a_net_echo`

Use a new short-path Vitis platform/application so the proven model workspace
remains recoverable. Use a separate workspace such as `F:\vwn` with components
`p_net` and `a_net_echo`; do not mutate the proven `F:\vwi` workspace.

1. Build `p_net` from the new network-enabled XSA with an A53 standalone
   domain. The guarded generator is
   `FPGA_Project/software/qmap_web_demo/create_network_vitis_workspace.py`.
   Set `QWEB_NETWORK_XSA` and `QWEB_VITIS_WORKSPACE=F:\vwn`, run its conda
   `--check-only --json` path first, then run the same file with Vitis 2025.1
   embedded Python. It refuses the old non-network XSA, a missing embedded
   bitstream, protected-workspace overlap, or any existing output path. The
   execution path atomically claims a new workspace, snapshots the XSA by
   content hash, and audits that snapshot before and after the Vitis build.
2. Confirm the generated platform contains one `psu_ethernet_3`/`xemacps`
   instance and one `psu_ttc_0`/`xttcps` timer instance. Stop if either is
   absent; do not patch generated BSP headers by hand.
3. Create the AMD `lwIP Echo Server` standalone application and use the lwIP
   RAW API. The installed Vitis 2025.1 tree provides `lwip220_v1_2` for the
   A53 standalone domain.
4. Connect a cable from `PS_ETH` to a router/switch, or directly to the PC.
   Keep the UART monitor open at 115200 8N1.
5. Start with a fixed private subnet if DHCP is inconvenient. One direct-link
   example is PC `192.168.10.1/24`, board `192.168.10.2/24`, no gateway. Use a
   unique locally administered MAC address; do not duplicate another device's
   MAC.
6. Run the app and record UART output for PHY ID/address, negotiated speed,
   link state, MAC address, and IP address.
7. From Windows, verify ICMP and TCP separately:

   ```powershell
   ping 192.168.10.2
   Test-NetConnection 192.168.10.2 -Port 7
   ```

8. Use `ncat`, `telnet`, or a small TCP client to send bytes to port 7 and
   verify that exactly the same bytes return.

Acceptance for Gate 1:

- Stable 1000/100/10 Mb/s link is reported.
- Repeated ping succeeds without intermittent loss.
- TCP port 7 accepts a connection and echoes payload bytes.
- UART remains usable for diagnostics.

The local Xilinx lwIP PHY-speed source does not name YT8521 explicitly. It may
print an unrecognized-PHY warning and use its Marvell-compatible fallback.
The board vendor reports this path working with PHY address 7 at 1000 Mb/s,
but that is a hypothesis to validate on this exact board. If link negotiation
fails, first capture MDIO PHY ID/status registers; only then add a small
YT8521-specific reset/speed routine. Do not start the web layer on an unstable
link.

## Gate 2: Minimal `a_qweb`

Fork the proven network app into a durable application under
`FPGA_Project/software/` only after Gate 1 passes. Keep lwIP in RAW mode and
use the allocation-free core under
`FPGA_Project/software/qmap_web_demo/`. The installed `lwip220_v1_2` library
does not link the optional application HTTPD implementation, so the board app
should add a small raw-TCP adapter around `qweb_http.[ch]` rather than assuming
that `httpd_init()` alone provides a linkable server.

Bring it up in this order:

1. Serve a fixed static page and connect the already host-tested
   `GET /api/health` route without touching the PL.
2. Connect the existing bounded `POST /api/generate` route, first with a
   token-id request such as:

   ```json
   {"tokens":[374],"max_new_tokens":2}
   ```

3. Wire `qweb_job_step()` to the existing QMAP runtime and return status through
   `GET /api/generate/<id>`; retrieve arbitrary raw detokenized bytes through
   `GET /api/generate/<id>/output`. Do not duplicate the register map in the
   HTTP code.
4. After token-ID hardware acceptance, exercise the already integrated exact
   tokenizer/detokenizer with the standard text prompt.
5. Verify the already implemented EOS/IM_END, length/context, malformed-input,
   error, and single-active-job behavior on the physical network path.

The current model run is long and the existing board smoke busy-polls. A web
application must continue servicing Ethernet while inference is active. Use a
cooperative main loop or state machine that repeatedly services
`xemacif_input`/lwIP timers and advances one bounded inference step. Do not
block inside an HTTP callback for the complete generation. A practical first
API is `POST /api/generate` returning a job ID, followed by
`GET /api/generate/<id>` for status/result.

The Web job deliberately rejects a null runner. Its board runner must service
`xemacif_input` plus TTC fast/slow timer flags at bounded intervals inside each
long single-token PL poll; merely moving the default tight poll loop from an
HTTP callback into `main()` would still starve TCP.

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

Passing `a_qweb` with token IDs proves Ethernet-to-PS-to-PL-to-PS transport.
The bounded prompt-prefill path, board-side tokenizer asset/runtime,
detokenization, EOS/length rules, and session state machine already pass host
validation. Product completion still requires physical UART acceptance of
`a_qgen`, then physical network acceptance of those same behaviors through
`a_qweb`, including an end-to-end arbitrary-prompt run and repeatability/error
recovery evidence.
