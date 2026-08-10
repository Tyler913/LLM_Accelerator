# QTKBPE1 C Runtime

This directory contains the allocation-free C11 consumer for the generated
`QTKBPE1` tokenizer asset. The runtime itself uses only standard C headers and
does not use a filesystem, `mmap`, global mutable state, or dynamic allocation.
The caller supplies a persistent asset buffer and BPE workspace.

Implemented and tested:

- structural parsing of the little-endian container;
- payload and per-section SHA-256 verification;
- table-bound, padding, overlap, offset, flag, and merge invariants;
- zero-copy token-ID to raw-byte slices;
- callback-based streaming detokenization with optional special-token skip;
- ByteLevel mapping and BPE for one already-normalized, already-pretokenized
  raw-byte piece, using binary search over the sorted merge table;
- strict UTF-8 validation;
- literal leftmost-longest added/special-token isolation;
- exact Unicode-9 NFC and Hangul normalization from pinned dependency tables;
- the fixed Qwen split expression with Unicode-16 letter, number, whitespace,
  and simple case-fold semantics;
- complete caller-buffer UTF-8 text tokenization with staged exact capacity
  requirements.

`qtk_encode_raw_piece()` remains the lower-level pre-split API. Full prompt text
uses `qtk_tokenize_utf8()`, which performs the complete fixed Qwen pipeline.
The post-processor adds no BOS/EOS tokens, matching this model configuration.

The streaming detokenizer emits raw bytes rather than decoding each token as
UTF-8. This is intentional: one generated token may end in the middle of a
multibyte UTF-8 sequence, so the UART/UI layer must preserve decoder state
across callbacks.

## Host Test

First generate the ignored asset as described in `../tokenizer/README.md`, then
run from the repository root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  FPGA_Project/software/qmap_prompt_demo/tokenizer_runtime/run_host_tests.ps1
```

The runner uses `conda run -n llm_fpga` to build a temporary Hugging Face
reference fixture, compiles the C runtime under strict GCC warnings, and checks
every token slice plus selected raw-byte pieces. The fixture, executable, and
asset remain under ignored `Temp/qmap_prompt_demo_tokenizer/`; only the host
test harness uses `malloc` to load those files.

The tracked differential corpus adds 71 multilingual/Unicode cases to the 8
tracked tokenizer goldens. Tests compare every token ID and both normal and
skip-special decoded byte streams against local Hugging Face `AutoTokenizer`.

## Model-only Output IDs

The LM head exposes IDs `0..151935`, while tokenizer decode data ends at
`151668`. `151669..151935` return `QTK_ERR_MODEL_ONLY_TOKEN`. A generation loop
must stop and report this condition; remapping, dropping, or treating these IDs
as EOS would hide a model/export mismatch.
