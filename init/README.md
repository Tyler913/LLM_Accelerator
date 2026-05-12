# Init Directory

This directory contains the cross-platform bootstrap flow for this project.
It is meant to let a future AI assistant or developer recreate the local
working state on macOS, Linux, or Windows without committing large model
artifacts to Git.

Start here:

- `CROSS_PLATFORM_SYNC.md`: human/AI runbook for a fresh machine.
- `HUGGINGFACE_WORKFLOW.md`: Hugging Face Hub workflow for large artifacts,
  uploads, downloads, and revision pinning.
- `model_assets.json`: manifest for the Qwen3 local model assets.
- `requirements.txt`: Python packages needed for the current PC-side flow.
- `download_model_assets.py`: downloads the model files from Hugging Face.
- `verify_assets.py`: checks that required local assets are present.
- `install_git_safety_hooks.sh`: installs local Git hooks that block accidental
  commits of model weights and other large generated artifacts.
- `git-hooks/pre-commit`: the tracked source for the local pre-commit hook.

Repository policy:

- Keep source code, scripts, small configs, and documentation in Git.
- Do not commit model weights or generated FPGA weight packs.
- Restore large assets from Hugging Face or another artifact store.
- Run Python commands through `conda run -n llm_fpga ...`.
- Install the Git safety hook on each clone:

```bash
bash init/install_git_safety_hooks.sh
```
