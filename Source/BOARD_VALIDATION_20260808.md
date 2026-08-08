# Full28 Board Validation - 2026-08-08

## Result

The routed XCZU2EG Qwen3-0.6B candidate passed its first complete physical-board
validation on 2026-08-08. The accepted path is:

```text
token id
  -> tied-Q4 embedding
  -> 28 decoder layers
  -> retained PL-DDR K/V cache
  -> final RMSNorm
  -> 151936-row tied-Q4 LM-head argmax
  -> output token id and Q26 score
```

The control smoke and the full28 persistent two-token model smoke both passed
over JTAG/UART on the ALIENTEK/ATK MPSoC-P4 XCZU2EG board.

This is a hardware PASS for the fixed two-token validation sequence. It is not
yet proof of arbitrary natural-language prompt-to-text generation: board-side
tokenization, detokenization, general prompt prefill, EOS/stop handling, and a
user-facing serial generation protocol remain future integration work.

## Hardware and runtime lineage

- Vivado/Vitis: `2025.1.1`
- Bitstream SHA256:
  `B4A4C6133DE7AF03F586E41BFC191B3A9379DAB3953E518D2660F0E2CFF7CC34`
- XSA SHA256:
  `DC0E2DDF6064DF29475BDF0E0A125223CEE0B7AF3209BAD05D65B3A5A19D28C8`
- XSA embedded bitstream: exact SHA256 match to the external bitstream
- FSBL SHA256:
  `E1C8A115A8E539868355987DA6908B028EE7CED635F547933DCA174D754F8E2F`
- Control ELF SHA256:
  `9F954F9A5364CBDB7E362754C049BF509CEE9807D36AC67596DEE68F7EF30E8B`
- Model ELF SHA256:
  `1CCD7B64D0481982586A0CC16477F7222568023AC06799F7494A666EC0E53C17`
- Reconstructed runtime manifest SHA256:
  `FA8981E71101DEF29970135DF5E863DA5634274DD9FB64905666F0CF1D47D3F2`
- Runtime: `61` segments, `394,547,200` bytes,
  `0x4_0010_0000..0x4_1A13_FFFF`
- Runtime release checks: `281` QMAP headers and `397` zero-initialized mutable
  regions
- Bench working directory: `F:\qot_boardtest_full28_20260808`

The historical `Temp/boardready_qwen3_full28_20260801` directory was absent at
the bench. The runtime was reconstructed without simulation from the retained
final full-chain manifest and its complete source assets. The reconstructed
manifest SHA256 is
`22C0BE1631F62F88527719D4AA9686F5EF5784F85B05B4CB92F5AD97742AE730`.
Its generated model header matched the existing Vitis model header in all 280
per-layer QMAP bases and all eight global addresses; their only textual
difference was the source-manifest SHA256 provenance comment. The rebuilt
runtime then passed the existing runtime range, SHA256, zero-region, and model-
address consistency validators before board launch.

## JTAG launch evidence

The standalone XSDB flow reported:

```text
PASS programmed bitstream
PASS FSBL initialization
PASS PL DDR4 ready status=0x00000005 after 1 poll(s)
Qwen3 PL-DDR runtime image load complete.
PASS all 281 QMAP packet headers
PASS application downloaded and running
```

Two bench-only launcher issues were found and corrected in durable source:

1. The default `level==0` PL target filter matched both `PS TAP` and `PSU` on
   this ZynqMP JTAG topology. The verified default target is `name =~ "PL"`.
2. Plain `dow -data` attempted A53 cache synchronization and failed during
   segment 1 at `0x4_0511_4000` with an APB-AP/DAP error. PL-DDR preload is not
   an A53-cached buffer, so every binary download now uses
   `dow -data -bypass-cache-sync`. The complete 8,486,912-byte segment 1 was
   retransmitted successfully, and the original failure address read back the
   exact file value `0x10F3FB1E`, before the clean full 61-segment launch passed.

## UART acceptance evidence

UART configuration: `COM230`, `115200 8N1`, DTR/RTS disabled.

Control acceptance:

```text
PASS qot_run_no_memory_validation_smoke
```

Model setup acceptance:

```text
Qwen3-0.6B full28 persistent two-token board smoke
QOT_BASEADDR=0x00000000a0040000
DDR4 status=0x00000005
PASS PL DDR4 and runtime image sentinels
```

Per-position results:

| Position | Input | Output | Score Q26 | Layers | Done mask | Error mask | Memory reads req/words | Memory writes req/words |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 374 | 28458 | 1227344433 | 28/28 | `0x0fffffff` | `0x00000000` | 1140936 / 80138208 | 172343 / 718851 |
| 1 | 28458 | 64 | 1015661901 | 28/28 | `0x0fffffff` | `0x00000000` | 1255624 / 80252896 | 172343 / 718851 |

Position 1 issued exactly `114,688` more accepted 32-bit read words than
position 0. That delta equals `28 layers * 2 K/V tensors * 16 query heads *
128 dimensions`, which is the expected additional read footprint for one
retained context position. Together with the shared K/V base, the `position+1`
cache length, and the uninterrupted two-command application run, this proves
that the second tested step consumed the retained position-0 K/V context. It
does not claim a byte-for-byte readback of the complete K/V cache.

Exact final acceptance lines:

```text
PASS token position=0 output=28458 score=1227344433
PASS token position=1 output=64 score=1015661901
PASS Qwen3-0.6B full28 persistent two-token board smoke
```

The position-1 input ID equals the verified position-0 output ID, forming the
tested two-token-ID chain. In this smoke application the A53 software supplies
that expected ID explicitly; this result does not yet claim autonomous PL-side
feedback or unbounded generation.

The locally retained raw UART capture is:

```text
Temp/board_validation_full28_20260808/uart_full28_persistent_two_token_pass.txt
```

Its SHA256 is
`D194D88AC3D2E0038C8D19CDB486B592E9A53F13DCFD9225CB150B8D936B2A8E`.

## Next capability slice

With the fixed two-token full28 hardware gate closed, the next correctness
slice is general token-id prompt prefill plus bounded decode/EOS handling.
Tokenizer, detokenizer, and a serial prompt/result protocol follow that token-id
path. Performance changes remain deferred because the routed design uses
`8,799 / 8,820` CLBs (`99.76%`).
