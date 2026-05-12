# Agent Instructions

Before making changes in this repository, read `PROJECT_CONTEXT.md` and
`CURRENT_STATE.md`.

Hard rules:

- This project is an FPGA-based LLM accelerator effort inspired by the
  Hummingbird+ paper in `paper/3748173.3779189.pdf`.
- Use the conda environment `llm_fpga` for every Python-related command:
  `conda run -n llm_fpga ...`.
- Do not use system Python, base conda Python, or another environment unless
  the user explicitly requests it.
- The local model baseline is `Qwen3-0.6B-Base/`, downloaded from
  `Qwen/Qwen3-0.6B-Base`.
- Preserve model artifacts and paper files unless the user explicitly asks to
  remove them.
- Do not run broad staging commands such as `git add .`, `git add -A`, or
  `git stash --include-untracked` unless the user explicitly asks for that Git
  operation. If staging is requested, stage explicit safe paths and verify that
  large model/artifact files remain ignored.

Project workflow notes:

- Keep `CURRENT_STATE.md` updated automatically after meaningful progress.
  Meaningful progress includes a new script, a successful or failed validation,
  an important tensor/shape/memory finding, a model/quantization/design
  decision, a blocker, or a changed next-step plan.
- Update `PROJECT_CONTEXT.md` when the new information is durable project
  context rather than just current progress. Update `AGENTS.md` only when the
  workflow rules themselves change.
- Do not paste long chat transcripts into project docs. Summarize the finding,
  evidence, command, result, and next action in a form future work can reuse.
- Write repository code, code comments, docstrings, script output labels, and
  project documentation in English unless the user explicitly asks for another
  language. Chinese is fine in chat with the user, but persisted project files
  should stay English.
- Keep large model weights and generated FPGA/model artifacts out of normal Git
  history. Use `init/CROSS_PLATFORM_SYNC.md` and `init/` scripts to restore
  assets on new machines.
- Keep the local Git safety hook installed with
  `init/install_git_safety_hooks.sh` when possible; it blocks accidental
  commits of model weights and generated large artifacts.
- PC-side Python scripts should be runnable from the repository root and by
  absolute script path. Prefer resolving the model directory relative to the
  script file, for example `Path(__file__).resolve().parents[1]` for scripts
  under `Qwen3-0.6B-Base/pc_testing/`.
- The current first-version target is functionality over performance: run a
  small Qwen3 base model with serial prefill, single-token decode, greedy
  argmax, and a simple future Q4 weight format.
