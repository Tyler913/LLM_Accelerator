# Cross-Platform Sync Runbook

Use this runbook when moving the project between macOS, Linux, and Windows.
It is written for a future AI assistant or developer operating from the
repository root.

## Goal

Recreate the local project state without storing large model files in the Git
repository.

GitHub should contain:

- project source files
- PC testing scripts
- project documentation
- model metadata and bootstrap scripts
- small Hugging Face config/tokenizer files if desired

GitHub should not contain:

- `model.safetensors`
- the 61 `qwen3_runtime_*.bin` Q4 PL-DDR segments used by the final QWEB demo
- future Q4/GGUF/GPTQ/FPGA binary weight packs
- generated build directories
- local cache folders

Large assets should be restored through this `init/` flow.

For Hugging Face upload/download and versioning practice, read
`init/HUGGINGFACE_WORKFLOW.md`.

Current restore sources:

- Baseline model mirror: `Tyler01/qwen3-0p6b-base-llm-accelerator`
  - Repo type: `model`
  - Revision: `d297782df3b18206f4b1caea202cf6272bae3aa9`
  - Source model: `Qwen/Qwen3-0.6B-Base`
- Final Q4 QWEB runtime: `Tyler01/qwen3-0p6b-fpga-q4-runtime`
  - Repo type: `model`
  - Revision: the exact commit SHA recorded in `init/q4_runtime_assets.json`
  - Local directory: `lmdeploy/qwen3-0p6b-q4-qweb-demo/model/`

If either repository is private, a fresh machine must be logged in to Hugging
Face with a token that can read it before downloading model assets.

## Fresh Machine Checklist

1. Clone the GitHub repository.

```bash
git clone <repo-url> LLM_Accelerator
cd LLM_Accelerator
```

2. Create or activate the required conda environment.

If the environment does not exist:

```bash
conda create -n llm_fpga python=3.12 -y
```

Install the Python packages needed for the current PC-side flow:

```bash
conda run -n llm_fpga python -m pip install -U -r init/requirements.txt
```

3. Log in to Hugging Face when the required repositories are private.

Windows PowerShell:

```powershell
conda run -n llm_fpga hf auth login
conda run -n llm_fpga hf auth whoami
```

Bash or zsh:

```bash
conda run -n llm_fpga hf auth login
conda run -n llm_fpga hf auth whoami
```

Keep `HF_TOKEN` outside the repository for non-interactive use. Never commit a
token, `.env` file, credential helper output, or Hugging Face cache.

4. Download and verify the upstream Qwen3 development model assets.

```bash
conda run -n llm_fpga python init/download_model_assets.py
conda run -n llm_fpga python init/verify_assets.py
```

This downloads the files listed in `init/model_assets.json` into
`Qwen3-0.6B-Base/`.

Use strict size checks only when reproducing the exact recorded binary
snapshot. Text files may have platform-dependent LF/CRLF working-tree sizes.

```bash
conda run -n llm_fpga python init/verify_assets.py --strict-sizes
```

5. Download and verify the board-accepted Q4 QWEB runtime.

Windows PowerShell:

```powershell
conda run -n llm_fpga python init/download_q4_runtime.py
conda run -n llm_fpga python init/verify_q4_runtime.py
```

Bash or zsh:

```bash
conda run -n llm_fpga python init/download_q4_runtime.py
conda run -n llm_fpga python init/verify_q4_runtime.py
```

The Q4 scripts use `init/q4_runtime_assets.json`, restore exactly 61
`qwen3_runtime_*.bin` files under
`lmdeploy/qwen3-0p6b-q4-qweb-demo/model/`, and verify their names, sizes, total
byte count, and SHA256 values. The committed manifest must contain an exact Hub
commit SHA, not `main`, before it is used as a reproducible release input.

6. Run a software smoke test when the upstream development model was restored.

```bash
conda run -n llm_fpga python Qwen3-0.6B-Base/pc_testing/01_test_generate.py
```

Then run the serial decode reference:

```bash
conda run -n llm_fpga python Qwen3-0.6B-Base/pc_testing/02_serial_decode.py
```

## Expected Local Layout

After setup, the local tree should include:

```text
Qwen3-0.6B-Base/
  config.json
  generation_config.json
  tokenizer.json
  tokenizer_config.json
  vocab.json
  merges.txt
  model.safetensors
  README.md
  LICENSE
  pc_testing/

lmdeploy/
  qwen3-0p6b-q4-qweb-demo/
    model/
      qwen3_runtime_00.bin
      ...
      qwen3_runtime_60.bin
```

`model.safetensors` and all 61 Q4 runtime `.bin` files are intentionally ignored
by Git. The Q4 manifests, loader, source code, and documentation remain tracked.

## Vendor Tool Boundary

The `init/` flow creates the Python environment and restores Hugging Face
assets. It does not install AMD Vivado/Vitis, XSDB, board/cable drivers,
licensed IP, or a local Vitis workspace. Install the project-compatible AMD
tools separately before building or launching the physical FPGA demo. A model
download or Python asset verification does not prove that the vendor toolchain
or development board is available.

## If the Download Fails

Check:

- network access to Hugging Face
- whether Hugging Face login can read
  `Tyler01/qwen3-0p6b-base-llm-accelerator`
- whether Hugging Face login can read
  `Tyler01/qwen3-0p6b-fpga-q4-runtime`
- whether `huggingface_hub` is installed in `llm_fpga`
- whether each Hugging Face repo ID and revision still match
  `init/model_assets.json` and `init/q4_runtime_assets.json`
- whether a proxy or login token is required on the target machine

Retry:

```bash
conda run -n llm_fpga python init/download_model_assets.py
conda run -n llm_fpga python init/download_q4_runtime.py
```

If the target machine already has the model files, copy them into
`Qwen3-0.6B-Base/` and run:

```bash
conda run -n llm_fpga python init/verify_assets.py
```

If the target machine already has the 61 Q4 runtime segments, copy them into
`lmdeploy/qwen3-0p6b-q4-qweb-demo/model/` and run:

```bash
conda run -n llm_fpga python init/verify_q4_runtime.py
```

## Q4 Artifact Publication Policy

The board-accepted Q4 runtime belongs in
`Tyler01/qwen3-0p6b-fpga-q4-runtime`, not in the main Git repository. Publish a
new Q4 runtime in this order:

1. Verify all 61 local `.bin` files.
2. Upload those 61 files with the current `hf upload` command.
3. Read the resulting Hugging Face commit SHA.
4. Pin that exact SHA with
   `conda run -n llm_fpga python init/pin_q4_runtime_revision.py <sha>`.
5. Run `init/verify_q4_runtime.py --from-hub`; it must download that exact Hub
   commit into a clean temporary directory and pass every hash.
6. Commit and push the source, manifests, scripts, and documentation to GitHub.

Never commit the 61 `.bin` files to GitHub, and never upload `HF_TOKEN`, local
Hub caches, Vitis workspaces, or generated credential files to either service.

When adding a new required large artifact, update:

- `.gitignore`
- `init/model_assets.json` or `init/q4_runtime_assets.json`
- `init/HUGGINGFACE_WORKFLOW.md` if the Hugging Face location or versioning
  practice changes
- `init/git-hooks/pre-commit` if the artifact introduces a new large-file
  extension or path pattern that should be blocked from Git
- this runbook
- `Source/CURRENT_STATE.md`
