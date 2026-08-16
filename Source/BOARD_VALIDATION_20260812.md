# Qwen3 Prompt-to-Text Board Validation - 2026-08-12

## Result

The first arbitrary prompt-to-text physical-board gate passed on the XCZU2EG
bench. One strict cold run completed this sequence without splitting the
acceptance claim:

1. program the preserved full28 bitstream;
2. run FSBL and require PL DDR status `0x00000005`;
3. load all 61 Qwen3 Q4 runtime segments (`394,547,200` bytes);
4. read and validate all 281 QMAP packet headers;
5. start the A53 `a_qgen` application;
6. require the exact tokenizer/startup banner through
   `READY vocab=151936 context=256`; and
7. pass eight ordered, single-flight UART transactions.

This establishes the physical chain

```text
UTF-8 prompt -> PS Qwen tokenizer -> token IDs -> PL full28 inference
-> generated IDs -> PS detokenizer -> UTF-8 output
```

for the current greedy, 256-position, JTAG-loaded first version.

## Trusted Inputs

- Workbench: `F:\qot_boardtest_prompt_text_v13_20260812`
- Workbench format: `5`
- Inventory: `88` files, `412,241,806` bytes
- Runtime: `61` segments, `394,547,200` bytes, `281` QMAP headers
- `a_qgen.elf`: `4,050,704` bytes
- `a_qgen.elf` SHA256:
  `F37EB88D4E1B75FEC815D01306EE85678C1E8555D02E5B42D3EFCA22FD337BBE`
- Tokenizer asset: `3,629,566` bytes
- Tokenizer SHA256:
  `C20242603EF4144E3F3F2EC4BA97C0E9C315AADD41F1BD2C5740E2A7FFA03A7D`
- Post-run workbench verifier: `trusted_provenance=True`

The workbench remains mechanically marked `WORKBENCH_NOT_RELEASE`; the physical
result is bound by the separate immutable evidence files below rather than by
mutating the package after use. XSDB-created `.Xil/.../llm_system.bda` and
`dfx_runtime.txt` were moved recoverably to
`Temp/workbench_runtime_debris_20260812_v13/` before the post-run verifier.

## Evidence

Directory:

`Temp/a_qgen_board_acceptance_20260812_v13_cold/`

| Artifact | Size | SHA256 |
| --- | ---: | --- |
| `acceptance.json` | structured report | `735C987EBE8AEB13C3243EB1FBA8D4CD057590DEF94C4BBEF5131A6A37798645` |
| `launcher.log` | cold-load/281-header evidence | `58D063F8ED5406919C3F4B5AFABD9B952B17A53592281174480F9375BBFF8EA1` |
| `uart_raw.bin` | 19,918 bytes | `8C2DCF4810BB53F39B6068166BDDBB15706E8B8D068760AC0FBB62076C0176D7` |

The report records `passed=true`, launcher return code `0`, all eight cases
passing, and the exact `READY` record.

## Accepted Transactions

| Case | Command | Exact result summary | Duration |
| --- | --- | --- | ---: |
| Liveness | `PING` | `PONG` | 0.110 s |
| Negative range | `TOKENS 1 1 151936` | `ERROR PARSE RANGE offset=11`, no PL launch | 0.109 s |
| Feedback | `TOKENS 2 1 374` | `28458/1227344433`, then actual-feedback `64/1015661901` | 13.438 s |
| Prefill | `TOKENS 1 2 374 28458` | retained-KV prefill, then `64/1015661901` | 13.437 s |
| Text-derived IDs | `TOKENS 2 5 785 3853 315 89462 374` | `264/1296911292`, `26291/1225544557` | 40.641 s |
| Text prompt | `PROMPT 2 The future of FPGA is` | exact IDs, results, and ` a fascinating` bytes | 40.656 s |
| Immediate repeat | same `PROMPT` | bit-exact repeat | 40.641 s |
| Vocabulary edge | `TOKENS 1 1 151935` | `28458/1224741478` | 6.750 s |

For the text cases, the board emitted:

```text
PROMPT_IDS 5 785 3853 315 89462 374
TOKEN 0 264 1296911292
BYTES 0 2061
TOKEN 1 26291 1225544557
BYTES 1 2066617363696e6174696e67
DONE 2 MAX_NEW
```

The concatenated byte records decode exactly as ` a fascinating`.

## Acceptance-Harness Corrections Proven During Bring-Up

- Invoke AMD `loader.bat -exec xsdb` on Windows so an uncaught Tcl error cannot
  be lost by `xsdb.bat` at `endlocal`.
- Allow 3600 seconds for a measured cold JTAG load; the old 900-second budget
  stopped at segment 23/61.
- Reject a nonzero launcher exit before `READY`, but allow return code zero
  because XSDB `con` resumes the A53 asynchronously and returns normally.
- Normalize only a leading terminal carriage return left between the FSBL
  banner and the first application line; all protocol content and whitespace
  remain exact.
- Preserve raw UART and structured failure reports for FSBL, DDR, DAP, timeout,
  and record-order failures.

## Remaining Product Boundary

The UART CLI and PC Web Serial GUI are now usable demo transports and do not
require board Ethernet. Since this acceptance, a fresh GEM3/MDIO/TTC0-enabled
Vivado/bitstream/XSA lineage, the network-XSA audit, and real patched-lwIP
`a_net_echo`/`a_qweb` Vitis builds have passed. Physical echo bring-up now
reaches exact Motorcomm YT8521 detection at PHY address 7, but link negotiation
times out; both host wired adapters were disconnected during that run. IP,
ping, TCP echo, and physical `a_qweb` HTTP prompt-to-text acceptance therefore
remain open. Power-on autonomy additionally requires a boot image, persistent
storage, and an autonomous PS runtime/weight loader; those are separate from
this JTAG-loaded prompt-chain PASS.
