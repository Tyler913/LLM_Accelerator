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
- `q4_runtime_assets.json`: Hugging Face location and pinned revision for the
  board-accepted 61-file Q4 PL-DDR runtime.
- `requirements.txt`: Python packages needed for the current PC-side flow.
- `download_model_assets.py`: downloads the model files from Hugging Face.
- `verify_assets.py`: checks that required local assets are present.
- `download_q4_runtime.py`: downloads the 61 Q4 runtime segments into
  `lmdeploy/qwen3-0p6b-q4-qweb-demo/model/`.
- `verify_q4_runtime.py`: checks the Q4 segment count, names, sizes, and SHA256
  values against the tracked runtime manifest.
- `pin_q4_runtime_revision.py`: replaces the temporary `main` revision with
  the exact Hugging Face commit SHA returned after upload.

Repository policy:

- Keep source code, scripts, small configs, and documentation in Git.
- Do not commit model weights or generated FPGA weight packs.
- Restore large assets from Hugging Face or another artifact store.
- Store the QWEB demo's 61 `qwen3_runtime_*.bin` files only in
  `Tyler01/qwen3-0p6b-fpga-q4-runtime`; keep their manifests and loader source
  in GitHub.
- Never upload Hugging Face tokens, `.env` files, Hugging Face caches, or local
  Vitis workspaces.
- Run Python commands through `conda run -n llm_fpga ...`.
On a fresh clone, restore and verify the final Q4 QWEB runtime with:

```bash
conda run -n llm_fpga python init/download_q4_runtime.py
conda run -n llm_fpga python init/verify_q4_runtime.py
```

The `init/` flow installs Python dependencies and restores model assets. It
does not install AMD Vivado/Vitis, cable drivers, board support, or licensed
vendor tools; install those separately when physical-board development is
required.
