# Release Validation Summary

Last updated: 2026-08-26

This is the release-facing validation summary. It distinguishes the original
physical-board evidence from the current portable package; it is not a claim
that the portable wrapper has already been physically revalidated.

## Original Physical Validation

The original controlled XCZU2EG bench closed these first-version gates:

| Date | Scope | Result |
| --- | --- | --- |
| 2026-08-08 | Full28 PL path, persistent two-token model smoke | PASS |
| 2026-08-12 | UART arbitrary prompt-to-text, including PS tokenizer and detokenizer | PASS |
| 2026-08-17 | Board-hosted QWEB prompt-to-text path over `PS_ETH` | PASS |

The accepted model path is:

```text
prompt -> PS tokenizer -> PL Q4 full28 inference -> PS detokenizer -> output
```

The first-version implementation uses serial prefill, cached single-token
decode, greedy argmax, a 256-token context, and the custom Q4 weight-only
PL-DDR runtime. It is a correctness-first JTAG-loaded demonstration, not a
power-on autonomous product or a performance benchmark.

## Portable Package Status

`lmdeploy/qwen3-0p6b-q4-qweb-demo/` contains the selected bitstream, XSA, FSBL,
application ELF, tokenizer asset, manifests, and relative-path launch support.
The 61 Q4 runtime segments are restored separately and verified against the
tracked manifests.

The package has passed a local integrity audit, but it has **not** yet completed
a new physical-board run through `run_demo.ps1`. It must therefore be described
as a release candidate, not as a portable board-accepted release.

## Required Checks

From the repository root:

```bash
conda run -n llm_fpga python init/download_q4_runtime.py
conda run -n llm_fpga python init/verify_q4_runtime.py
```

On Windows, audit the complete package before using hardware:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\lmdeploy\qwen3-0p6b-q4-qweb-demo\run_demo.ps1 -AuditOnly
```

Only a new controlled bench run, with JTAG, UART, and `PS_ETH` connected to the
same board, can close the remaining portable-wrapper validation gate.

## Scope Exclusions

The validation does not establish autonomous boot, SD/QSPI deployment,
automatic runtime loading, persistent or multi-client HTTP, broad prompt
coverage, or performance targets. Those are separate productization work.

Detailed incremental logs, old workbench paths, temporary outputs, and raw
acceptance artifacts are intentionally excluded from the release tree. Preserve
them in a separately managed evidence archive when they are needed for an
engineering or audit review.
