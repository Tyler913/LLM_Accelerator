---
license: apache-2.0
base_model: Qwen/Qwen3-0.6B-Base
tags:
  - qwen3
  - fpga
  - q4
  - qmap
  - xilinx
---

# Qwen3 0.6B Custom QMAP/Q4 FPGA Runtime

This repository contains the Q4 runtime segments used by the
`qwen3-0p6b-q4-qweb-demo` FPGA board demo.

It is a hardware-specific, prepacked PL-DDR runtime image derived from
[`Qwen/Qwen3-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-0.6B-Base). It is
not a standard Transformers checkpoint, GGUF model, ONNX model, or LMDeploy
model package. Generic model loaders cannot consume these files.

## Target integration

The matching source, board artifacts, launcher, and canonical verification
manifests live in the GitHub project:

[`Tyler913/LLM_Accelerator`](https://github.com/Tyler913/LLM_Accelerator/tree/main/lmdeploy/qwen3-0p6b-q4-qweb-demo)

The intended Hub repository is:

```text
Tyler01/qwen3-0p6b-fpga-q4-runtime
```

The files are consumed by the custom XSDB loader and the matching FPGA
bitstream/application pair. They must not be mixed with artifacts from another
bitstream, address map, quantization export, or software build.

## Runtime contents

The complete runtime consists of exactly:

```text
qwen3_runtime_00.bin
qwen3_runtime_01.bin
...
qwen3_runtime_60.bin
```

Contract:

- segment count: `61`;
- total segment bytes: `394547200`;
- Q4 weight grouping: symmetric groups of `64` weights;
- layout: custom QMAP/Q4 packets plus the supporting FPGA runtime data required
  by the accepted full-28-layer image;
- load addresses, exact byte counts, and per-file SHA-256 values: defined by
  the GitHub `model/pl_ddr_binary_segments.json` manifest.

These 61 files form one atomic runtime package. Do not omit, rename, concatenate,
reorder, or edit individual segments.

## Download through the GitHub project

After cloning the GitHub repository, run from its root:

```powershell
conda run -n llm_fpga python init/download_q4_runtime.py
conda run -n llm_fpga python init/verify_q4_runtime.py
```

The project downloads the segments into:

```text
lmdeploy/qwen3-0p6b-q4-qweb-demo/model/
```

The init manifest records the Hub repository and must pin an exact Hub commit
revision after upload. If the repository is private, authenticate with the
Hugging Face CLI first.

## Verification

The Hub copy alone is not the verification authority. Use the manifest and
verification tools tracked in GitHub. Verification must confirm:

1. the exact names `qwen3_runtime_00.bin` through
   `qwen3_runtime_60.bin` and no extra matching segment files;
2. exactly 61 segments and exactly `394547200` total bytes;
3. every per-file size and SHA-256 value;
4. the manifest hash pinned by the release manifest;
5. the recorded PL-DDR addresses, alignment, aperture, and non-overlap rules.

After model verification, audit the complete board package without touching
hardware:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\lmdeploy\qwen3-0p6b-q4-qweb-demo\run_demo.ps1 `
  -AuditOnly
```

## Scope and limitations

- This artifact is for the matching custom FPGA QWEB demo, not for CPU or GPU
  inference frameworks.
- It does not contain the upstream tokenizer/configuration needed by a standard
  Transformers pipeline.
- It does not provide a generic Q4 interchange format.
- No performance claim is made by this model card.
- The original external-location artifacts passed a physical board flow, but
  the new portable wrapper and directory layout require one new physical board
  validation before being labeled board-accepted.

Retain the upstream attribution and Apache-2.0 license information when
redistributing this derived runtime.
