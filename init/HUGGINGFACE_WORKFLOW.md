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
- the private project mirror of the current Qwen3 baseline model
- generated Q4/INT8 weight packs
- future FPGA-specific binary artifacts
- large operator test vectors, if they become too large for Git

Do not upload Hugging Face tokens, local caches, or generated secrets to either
GitHub or Hugging Face repositories.

## Current Project Model Mirror

The current private project mirror for cross-machine restore is:

```text
Tyler01/qwen3-0p6b-base-llm-accelerator
```

It contains the model and tokenizer files from `Qwen/Qwen3-0.6B-Base`, but not
the project validation scripts under `Qwen3-0.6B-Base/pc_testing/` or
`Qwen3-0.6B-Base/python_each_module/`. Those scripts live in GitHub.

Current pinned revision:

```text
d297782df3b18206f4b1caea202cf6272bae3aa9
```

The exact restore source is recorded in `init/model_assets.json`.

## Final Q4 QWEB Runtime

The board-accepted Q4 PL-DDR runtime will be published separately from the
baseline model mirror:

```text
Tyler01/qwen3-0p6b-fpga-q4-runtime
```

It consists of exactly 61 files named `qwen3_runtime_00.bin` through
`qwen3_runtime_60.bin`, totaling `394,547,200` bytes. On a project checkout,
they are restored to:

```text
lmdeploy/qwen3-0p6b-q4-qweb-demo/model/
```

The repository ID, repository type, and local directory are recorded in
`init/q4_runtime_assets.json`. Before the first upload its revision is the
temporary value `main`; the upload procedure below replaces it with the exact
Hugging Face commit. The tracked Q4 segment manifest is the authority for each
file name, size, and SHA256 value. GitHub contains that metadata, the loader,
source code, and documentation; after publication Hugging Face contains the 61
large `.bin` files.

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

Login from the project environment on Windows PowerShell:

```powershell
conda run -n llm_fpga hf auth login
conda run -n llm_fpga hf auth whoami
```

Or from Bash or zsh:

```bash
conda run -n llm_fpga hf auth login
conda run -n llm_fpga hf auth whoami
```

For scripts or CI, prefer an environment variable instead of pasting a token
into command history. Set `HF_TOKEN` outside the repository, then run:

Bash or zsh:

```bash
conda run -n llm_fpga hf auth login --token "$HF_TOKEN"
```

Windows PowerShell:

```powershell
conda run -n llm_fpga hf auth login --token $env:HF_TOKEN
```

Do not commit `.env` files or shell history containing tokens.

## Download From Hugging Face

The current project restore repo is:

```text
Tyler01/qwen3-0p6b-base-llm-accelerator
```

It is a private model repo, so log in before downloading on a fresh machine:

```bash
conda run -n llm_fpga hf auth login
conda run -n llm_fpga hf auth whoami
```

The project download script wraps `huggingface_hub.snapshot_download` and uses
the pinned repo/revision from `init/model_assets.json`:

```bash
conda run -n llm_fpga python init/download_model_assets.py
```

You can also use the CLI directly:

```bash
conda run -n llm_fpga hf download Tyler01/qwen3-0p6b-base-llm-accelerator \
  --repo-type model \
  --local-dir Qwen3-0.6B-Base
```

Download a pinned revision:

```bash
conda run -n llm_fpga hf download Tyler01/qwen3-0p6b-base-llm-accelerator \
  --repo-type model \
  --revision d297782df3b18206f4b1caea202cf6272bae3aa9 \
  --local-dir Qwen3-0.6B-Base
```

Preview a download without fetching files:

```bash
conda run -n llm_fpga hf download Tyler01/qwen3-0p6b-base-llm-accelerator \
  --repo-type model \
  --dry-run
```

Restore the final Q4 QWEB runtime on Windows PowerShell:

```powershell
conda run -n llm_fpga python init/download_q4_runtime.py
conda run -n llm_fpga python init/verify_q4_runtime.py
```

Restore it from Bash or zsh:

```bash
conda run -n llm_fpga python init/download_q4_runtime.py
conda run -n llm_fpga python init/verify_q4_runtime.py
```

Both commands use `init/q4_runtime_assets.json`. A release-ready manifest must
pin an exact Hugging Face commit SHA. A temporary `main` revision is permitted
only during the first upload, before a Hub commit exists.

## Create the Q4 Runtime Repository

Do not re-upload the original official Qwen model unless there is a clear
reason. The first useful private Hub repository for this project should be for
derived artifacts, such as custom Q4 weights.

The exception currently in use is the private baseline mirror
`Tyler01/qwen3-0p6b-base-llm-accelerator`, created for cross-platform sync.
The derived Q4 runtime uses its own model repository.

Windows PowerShell:

```powershell
function Assert-NativeSuccess([string]$Step) {
  if ($LASTEXITCODE -ne 0) { throw "$Step failed (exit $LASTEXITCODE)" }
}
$RepoRoot = (git rev-parse --show-toplevel).Trim()
Assert-NativeSuccess "locate Git repository"
Set-Location $RepoRoot
$HfRepo = "Tyler01/qwen3-0p6b-fpga-q4-runtime"
conda run -n llm_fpga hf repos create $HfRepo `
  --repo-type model `
  --private `
  --exist-ok
Assert-NativeSuccess "create Hugging Face repository"
```

Bash or zsh:

```bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
HF_REPO="Tyler01/qwen3-0p6b-fpga-q4-runtime"
conda run -n llm_fpga hf repos create "$HF_REPO" \
  --repo-type model \
  --private \
  --exist-ok
```

Use `--public` instead of `--private` only when publication is intentional and
the model card, attribution, and licensing information are ready.

## Upload or Update the 61 Q4 Segments

Run the local verifier before every upload. The upload command deliberately
includes only `qwen3_runtime_*.bin`, the public `README.md` model card, and its
`LICENSE`; it cannot upload a token, `.env` file, Hugging Face cache, Vitis
workspace, or unrelated local file.

Windows PowerShell:

```powershell
function Assert-NativeSuccess([string]$Step) {
  if ($LASTEXITCODE -ne 0) { throw "$Step failed (exit $LASTEXITCODE)" }
}
$RepoRoot = (git rev-parse --show-toplevel).Trim()
Assert-NativeSuccess "locate Git repository"
Set-Location $RepoRoot
$HfRepo = "Tyler01/qwen3-0p6b-fpga-q4-runtime"
$ModelDir = "lmdeploy/qwen3-0p6b-q4-qweb-demo/model"

conda run -n llm_fpga python init/verify_q4_runtime.py --allow-unpinned
Assert-NativeSuccess "verify local Q4 runtime"

$Segments = @(Get-ChildItem -LiteralPath $ModelDir -File -Filter "qwen3_runtime_*.bin")
if ($Segments.Count -ne 61) {
    throw "Expected 61 Q4 runtime segments, found $($Segments.Count)"
}

conda run -n llm_fpga hf upload $HfRepo `
  $ModelDir `
  . `
  --repo-type model `
  --include "qwen3_runtime_*.bin" `
  --include "README.md" `
  --include "LICENSE" `
  --commit-message "Publish board-accepted Qwen3 0.6B Q4 QWEB runtime"
Assert-NativeSuccess "upload Q4 runtime"

$HfRevision = (conda run -n llm_fpga python -c "from huggingface_hub import HfApi; print(HfApi().model_info('$HfRepo').sha)").Trim()
Assert-NativeSuccess "read Hugging Face revision"
conda run -n llm_fpga python init/pin_q4_runtime_revision.py $HfRevision
Assert-NativeSuccess "pin Hugging Face revision"
conda run -n llm_fpga python init/verify_q4_runtime.py --from-hub
Assert-NativeSuccess "redownload and verify pinned Hub revision"
```

Bash or zsh:

```bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
HF_REPO="Tyler01/qwen3-0p6b-fpga-q4-runtime"
MODEL_DIR="lmdeploy/qwen3-0p6b-q4-qweb-demo/model"

conda run -n llm_fpga python init/verify_q4_runtime.py --allow-unpinned

SEGMENT_COUNT="$(conda run -n llm_fpga python -c 'from pathlib import Path; import sys; print(len(list(Path(sys.argv[1]).glob("qwen3_runtime_*.bin"))))' "$MODEL_DIR" | tr -d '\r\n')"
test "$SEGMENT_COUNT" -eq 61 || {
  echo "Expected 61 Q4 runtime segments, found $SEGMENT_COUNT" >&2
  exit 1
}

conda run -n llm_fpga hf upload "$HF_REPO" \
  "$MODEL_DIR" \
  . \
  --repo-type model \
  --include 'qwen3_runtime_*.bin' \
  --include 'README.md' \
  --include 'LICENSE' \
  --commit-message "Publish board-accepted Qwen3 0.6B Q4 QWEB runtime"

HF_REVISION="$(conda run -n llm_fpga python -c "from huggingface_hub import HfApi; print(HfApi().model_info('$HF_REPO').sha)" | tr -d '\r\n')"
conda run -n llm_fpga python init/pin_q4_runtime_revision.py "$HF_REVISION"
conda run -n llm_fpga python init/verify_q4_runtime.py --from-hub
```

The project no longer uses the older `hf upload-large-folder` recommendation.
Use the current `hf upload` command above for both first publication and later
updates.

## Pin the Upload and Push GitHub

Publishing is a two-repository transaction. Complete it in this order:

1. Upload all 61 `.bin` files to Hugging Face.
2. Read the resulting Hub commit SHA as shown above.
3. Run `init/pin_q4_runtime_revision.py <sha>` to replace the temporary
   `"revision": "main"` with that exact SHA.
4. Run `init/verify_q4_runtime.py --from-hub`; it downloads that exact revision
   into a temporary directory and must pass all 61 hashes before GitHub update.
5. Confirm that Git ignores the `.bin` files.
6. Commit and push the tracked manifest, scripts, source, and documentation to
   GitHub.

Windows PowerShell verification and Git update:

```powershell
function Assert-NativeSuccess([string]$Step) {
  if ($LASTEXITCODE -ne 0) { throw "$Step failed (exit $LASTEXITCODE)" }
}
$RepoRoot = (git rev-parse --show-toplevel).Trim()
Assert-NativeSuccess "locate Git repository"
Set-Location $RepoRoot
$ExistingStaged = @(git diff --cached --name-only)
Assert-NativeSuccess "inspect existing staging area"
if ($ExistingStaged.Count -ne 0) {
  throw "Refusing to mix this release with already-staged files"
}
git check-ignore "lmdeploy/qwen3-0p6b-q4-qweb-demo/model/qwen3_runtime_00.bin"
Assert-NativeSuccess "confirm Q4 Git ignore"
conda run -n llm_fpga python init/verify_q4_runtime.py --from-hub
Assert-NativeSuccess "redownload and verify pinned Hub revision"
conda run -n llm_fpga python init/verify_q4_runtime.py
Assert-NativeSuccess "verify pinned local Q4 runtime"
git status --short

git add .gitignore README.md `
  Source/CURRENT_STATE.md `
  Source/PROJECT_CONTEXT.md `
  Source/PS_NETWORK_BRINGUP.md `
  init/q4_runtime_assets.json `
  init/download_q4_runtime.py `
  init/verify_q4_runtime.py `
  init/pin_q4_runtime_revision.py `
  init/requirements.txt `
  init/README.md `
  init/CROSS_PLATFORM_SYNC.md `
  init/HUGGINGFACE_WORKFLOW.md `
  lmdeploy/qwen3-0p6b-q4-qweb-demo
Assert-NativeSuccess "stage GitHub release"
$StagedQ4 = @(git diff --cached --name-only | Select-String `
  '^lmdeploy/qwen3-0p6b-q4-qweb-demo/model/qwen3_runtime_[0-9]{2}\.bin$')
if ($StagedQ4.Count -ne 0) { throw "Q4 runtime binaries were staged unexpectedly" }
git diff --cached --check
Assert-NativeSuccess "check staged diff"
git status --short
git commit -m "Add reproducible Q4 QWEB runtime restore"
Assert-NativeSuccess "commit GitHub release"
git push
Assert-NativeSuccess "push GitHub release"
```

Bash/zsh verification and Git update:

```bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
test -z "$(git diff --cached --name-only)" || {
  echo "Refusing to mix this release with already-staged files" >&2
  exit 1
}
git check-ignore 'lmdeploy/qwen3-0p6b-q4-qweb-demo/model/qwen3_runtime_00.bin'
conda run -n llm_fpga python init/verify_q4_runtime.py --from-hub
conda run -n llm_fpga python init/verify_q4_runtime.py
git status --short

git add .gitignore README.md \
  Source/CURRENT_STATE.md \
  Source/PROJECT_CONTEXT.md \
  Source/PS_NETWORK_BRINGUP.md \
  init/q4_runtime_assets.json \
  init/download_q4_runtime.py \
  init/verify_q4_runtime.py \
  init/pin_q4_runtime_revision.py \
  init/requirements.txt \
  init/README.md \
  init/CROSS_PLATFORM_SYNC.md \
  init/HUGGINGFACE_WORKFLOW.md \
  lmdeploy/qwen3-0p6b-q4-qweb-demo
if git diff --cached --name-only | grep -Eq \
  '^lmdeploy/qwen3-0p6b-q4-qweb-demo/model/qwen3_runtime_[0-9]{2}\.bin$'; then
  echo "Q4 runtime binaries were staged unexpectedly" >&2
  exit 1
fi
git diff --cached --check
git status --short
git commit -m "Add reproducible Q4 QWEB runtime restore"
git push
```

Review `git status --short` before committing. Do not use `git add -f` to force
the ignored Q4 binaries into Git.

## Restore Project Artifacts on Another Machine

After cloning GitHub and installing `init/requirements.txt`, restore the final
runtime with the project scripts rather than a floating direct download:

```bash
conda run -n llm_fpga python init/download_q4_runtime.py
conda run -n llm_fpga python init/verify_q4_runtime.py
```

The download script uses the pinned commit from `init/q4_runtime_assets.json`;
the verifier checks all 61 files before the board launcher may consume them.

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

The Q4 runtime manifest must contain, at minimum:

```json
{
  "schema_version": 1,
  "artifact": {
    "repo_id": "Tyler01/qwen3-0p6b-fpga-q4-runtime",
    "repo_type": "model",
    "revision": "<exact-hugging-face-commit-sha>",
    "local_dir": "lmdeploy/qwen3-0p6b-q4-qweb-demo/model",
    "source_model": "Qwen/Qwen3-0.6B-Base",
    "format": "custom_qmap_groupwise_symmetric_q4",
    "group_size": 64,
    "segment_manifest": "lmdeploy/qwen3-0p6b-q4-qweb-demo/model/pl_ddr_binary_segments.json",
    "segment_manifest_sha256": "fa8981e71101def29970135df5e863da5634274dd9fb64905666f0cf1d47d3f2",
    "expected_segment_count": 61,
    "expected_total_bytes": 394547200,
    "pl_ddr_aperture": {
      "start": "0x0000000400000000",
      "end_exclusive": "0x0000000420000000",
      "alignment_bytes": 4
    }
  }
}
```

GitHub tracks this manifest and the authoritative per-segment checksum
manifest. Hugging Face stores the 61 large files referenced by them.

## Vendor Tool Boundary

The `init/` scripts install Python dependencies and restore model artifacts.
They do not install AMD Vivado/Vitis, XSDB, cable or board drivers, licensed IP,
or a Vitis workspace. Install the compatible vendor toolchain separately. Do
not upload a Vitis installation or generated workspace to GitHub or Hugging
Face.

## When To Use Direct Git

Direct `git clone`, `git add`, `git commit`, and `git push` can work with
Hugging Face repositories, but it adds Git LFS and credential-management
friction. Prefer the `hf` CLI and Python APIs for this project unless there is
a specific need to inspect full repo history offline.

Useful references:

- Hugging Face CLI guide: https://huggingface.co/docs/huggingface_hub/guides/cli
- Download guide: https://huggingface.co/docs/huggingface_hub/guides/download
- Git vs HTTP concept: https://huggingface.co/docs/huggingface_hub/concepts/git_vs_http
