# Hugging Face Hub Workflow

This note explains how to use Hugging Face Hub for large model and artifact
sync in this project.

## Recommended Split

Use GitHub for:

- source code
- PC-side validation scripts
- Markdown documentation
- manifests and setup scripts
- small configuration files

Use Hugging Face Hub for:

- upstream model downloads
- generated Q4/INT8 weight packs
- future FPGA-specific binary artifacts
- large operator test vectors, if they become too large for Git

Do not upload Hugging Face tokens, local caches, or generated secrets to either
GitHub or Hugging Face repositories.

## Mental Model

Hugging Face repositories are git-backed, but this project should usually use
the `hf` CLI or `huggingface_hub` Python APIs instead of manually cloning model
repositories.

Each upload creates a Hub commit. Later downloads can be pinned by:

- branch name, such as `main`
- tag name, such as `q4-v0.1`
- exact commit hash

For reproducible FPGA work, record the exact `repo_id`, `revision`, file names,
sizes, and checksums in a manifest whenever an artifact becomes required.

## Login

Create a Hugging Face user access token from:

```text
https://huggingface.co/settings/tokens
```

Use a token with write permission if you need to upload artifacts.

Login from the project environment:

```bash
conda run -n llm_fpga hf auth login
```

Check the active account:

```bash
conda run -n llm_fpga hf auth whoami
```

For scripts or CI, prefer an environment variable instead of pasting a token
into command history. Set `HF_TOKEN` outside the repository, then run:

```bash
conda run -n llm_fpga hf auth login --token "$HF_TOKEN"
```

Do not commit `.env` files or shell history containing tokens.

## Download From Hugging Face

The current upstream model is:

```text
Qwen/Qwen3-0.6B-Base
```

The project download script already wraps `huggingface_hub.snapshot_download`:

```bash
conda run -n llm_fpga python init/download_model_assets.py
```

You can also use the CLI directly:

```bash
conda run -n llm_fpga hf download Qwen/Qwen3-0.6B-Base --local-dir Qwen3-0.6B-Base
```

Download a pinned revision:

```bash
conda run -n llm_fpga hf download Qwen/Qwen3-0.6B-Base \
  --revision <branch-tag-or-commit> \
  --local-dir Qwen3-0.6B-Base
```

Preview a download without fetching files:

```bash
conda run -n llm_fpga hf download Qwen/Qwen3-0.6B-Base --dry-run
```

## Create a Project Artifact Repository

Do not re-upload the original official Qwen model unless there is a clear
reason. The first useful private Hub repository for this project should be for
derived artifacts, such as custom Q4 weights.

Example model artifact repository:

```bash
conda run -n llm_fpga hf repos create <hf-username>/qwen3-0p6b-fpga-artifacts \
  --repo-type model \
  --private \
  --exist-ok
```

Use a dataset repository instead if the files are mostly test vectors,
calibration data, traces, or logs:

```bash
conda run -n llm_fpga hf repos create <hf-username>/llm-accelerator-test-vectors \
  --repo-type dataset \
  --private \
  --exist-ok
```

## Upload Artifacts

For a normal single-commit upload:

```bash
conda run -n llm_fpga hf upload <hf-username>/qwen3-0p6b-fpga-artifacts \
  artifacts/qwen3_0p6b_base_q4 \
  qwen3_0p6b_base_q4 \
  --repo-type model \
  --commit-message "Add Qwen3 0.6B base Q4 FPGA artifact v0.1"
```

For very large or interruption-prone uploads, use the resumable large-folder
command:

```bash
conda run -n llm_fpga hf upload-large-folder <hf-username>/qwen3-0p6b-fpga-artifacts \
  artifacts/qwen3_0p6b_base_q4 \
  --repo-type model
```

Use clear commit messages. A good commit message should describe:

- model base
- artifact format
- quantization format
- version or experiment id

## Restore Project Artifacts on Another Machine

After cloning the GitHub repo and installing requirements, download the large
artifact from Hugging Face:

```bash
conda run -n llm_fpga hf download <hf-username>/qwen3-0p6b-fpga-artifacts \
  --repo-type model \
  --revision <branch-tag-or-commit> \
  --local-dir artifacts/hf/qwen3-0p6b-fpga-artifacts
```

Then verify files with a project manifest or checksum file before using them.

## Versioning Practice

For each artifact that matters, keep these records in Git:

- artifact Hub `repo_id`
- `repo_type`
- exact `revision`
- file list
- file sizes
- checksums
- generation script or command
- source model revision
- quantization settings

Example future manifest fields:

```json
{
  "name": "qwen3_0p6b_base_q4_v0_1",
  "repo_id": "<hf-username>/qwen3-0p6b-fpga-artifacts",
  "repo_type": "model",
  "revision": "<commit-hash-or-tag>",
  "source_model": "Qwen/Qwen3-0.6B-Base",
  "format": "custom_groupwise_symmetric_q4",
  "group_size": 64,
  "files": [
    "qwen3_0p6b_base_q4/model_meta.json",
    "qwen3_0p6b_base_q4/weights_q4.bin",
    "qwen3_0p6b_base_q4/scales_fp16.bin",
    "qwen3_0p6b_base_q4/checksums.txt"
  ]
}
```

The GitHub repository should track this manifest. Hugging Face should store the
large files referenced by the manifest.

## When To Use Direct Git

Direct `git clone`, `git add`, `git commit`, and `git push` can work with
Hugging Face repositories, but it adds Git LFS and credential-management
friction. Prefer the `hf` CLI and Python APIs for this project unless there is
a specific need to inspect full repo history offline.

Useful references:

- Hugging Face CLI guide: https://huggingface.co/docs/huggingface_hub/guides/cli
- Download guide: https://huggingface.co/docs/huggingface_hub/guides/download
- Git vs HTTP concept: https://huggingface.co/docs/huggingface_hub/concepts/git_vs_http
