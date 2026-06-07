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
- future Q4/GGUF/GPTQ/FPGA binary weight packs
- generated build directories
- local cache folders

Large assets should be restored through this `init/` flow.

For Hugging Face upload/download and versioning practice, read
`init/HUGGINGFACE_WORKFLOW.md`.

Current model restore source:

- Hugging Face repo: `Tyler01/qwen3-0p6b-base-llm-accelerator`
- Repo type: `model`
- Revision: `d297782df3b18206f4b1caea202cf6272bae3aa9`
- Source model: `Qwen/Qwen3-0.6B-Base`

This repository is private. A fresh machine must be logged in to Hugging Face
with a token that can read this repo before downloading model assets.

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

3. Download the Qwen3 model assets.

If this is a fresh machine, log in to Hugging Face first:

```bash
conda run -n llm_fpga hf auth login
conda run -n llm_fpga hf auth whoami
```

```bash
conda run -n llm_fpga python init/download_model_assets.py
```

This downloads the files listed in `init/model_assets.json` into
`Qwen3-0.6B-Base/`.

4. Verify local assets.

```bash
conda run -n llm_fpga python init/verify_assets.py
```

Use strict size checks only when reproducing the exact current snapshot:

```bash
conda run -n llm_fpga python init/verify_assets.py --strict-sizes
```

5. Run a smoke test.

```bash
conda run -n llm_fpga python Qwen3-0.6B-Base/pc_testing/01_test_generate.py
```

Then run the serial decode reference:

```bash
conda run -n llm_fpga python Qwen3-0.6B-Base/pc_testing/02_serial_decode.py
```

6. Install the local Git safety hook.

```bash
bash init/install_git_safety_hooks.sh
```

This hook blocks commits that accidentally stage large model weights or common
generated artifact extensions. It is a local safety net; it does not replace
`.gitignore`.

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
```

`model.safetensors` is intentionally ignored by Git.

## If the Download Fails

Check:

- network access to Hugging Face
- whether Hugging Face login can read
  `Tyler01/qwen3-0p6b-base-llm-accelerator`
- whether `huggingface_hub` is installed in `llm_fpga`
- whether the Hugging Face repo ID and revision still match
  `init/model_assets.json`
- whether a proxy or login token is required on the target machine

Retry:

```bash
conda run -n llm_fpga python init/download_model_assets.py
```

If the target machine already has the model files, copy them into
`Qwen3-0.6B-Base/` and run:

```bash
conda run -n llm_fpga python init/verify_assets.py
```

## Future Artifact Policy

When Q4 or FPGA-specific weight files are generated, do not commit them to the
main Git repository. Store them in one of these places:

- a separate Hugging Face model or dataset repository
- an internal artifact store
- a release asset if the file is small and intentionally distributed
- local storage only, with a manifest and regeneration script

When adding a new required large artifact, update:

- `.gitignore`
- `init/model_assets.json` or a new manifest
- `init/HUGGINGFACE_WORKFLOW.md` if the Hugging Face location or versioning
  practice changes
- `init/git-hooks/pre-commit` if the artifact introduces a new large-file
  extension or path pattern that should be blocked from Git
- this runbook
- `Source/CURRENT_STATE.md`
