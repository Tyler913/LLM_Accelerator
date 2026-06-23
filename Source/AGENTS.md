# Agent Instructions

Before making changes in this repository, read `Source/PROJECT_CONTEXT.md` and
`Source/CURRENT_STATE.md`.

Hard rules:

- This project is an FPGA-based LLM accelerator effort inspired by the
  Hummingbird+ paper in `paper/3748173.3779189.pdf`.
- Use the conda environment `llm_fpga` for every Python-related command:
  `conda run -n llm_fpga ...`.
- Do not use system Python, base conda Python, or another environment unless
  the user explicitly requests it.
- The local model baseline is `Qwen3-0.6B-Base/`, downloaded from
  `Qwen/Qwen3-0.6B-Base`.
- The first deployable PL weight path must use the project's custom Q4
  weight-only quantization for all large model weights stored in PL DDR4. Do
  not design BF16/FP32 full-weight storage or full-precision GEMV datapaths as
  the default implementation path; FP32/BF16 data is for software reference and
  bring-up comparison only.
- Preserve model artifacts and paper files unless the user explicitly asks to
  remove them.
- Do not run broad staging commands such as `git add .`, `git add -A`, or
  `git stash --include-untracked` unless the user explicitly asks for that Git
  operation. If staging is requested, stage explicit safe paths and verify that
  large model/artifact files remain ignored.

Project workflow notes:

- Core project handoff Markdown files live under `Source/`; the root
  `README.md` is the entry point.
- Keep `Source/CURRENT_STATE.md` updated automatically after meaningful progress.
  Meaningful progress includes a new script, a successful or failed validation,
  an important tensor/shape/memory finding, a model/quantization/design
  decision, a blocker, or a changed next-step plan.
- Update `Source/PROJECT_CONTEXT.md` when the new information is durable project
  context rather than just current progress. Update `Source/AGENTS.md` only
  when the workflow rules themselves change.
- Treat the QMAP dot64 and row1024 PL-master board passes as sufficient to
  close the isolated Q4 GEMV smoke-test phase. Do not propose additional
  row-loop or small multi-row smoke-test detours as the default next step unless
  they are needed to debug a specific integration blocker.
- Do not paste long chat transcripts into project docs. Summarize the finding,
  evidence, command, result, and next action in a form future work can reuse.
- Write repository code, code comments, docstrings, script output labels, and
  project documentation in English unless the user explicitly asks for another
  language. Chinese is fine in chat with the user, but persisted project files
  should stay English.
- Keep large model weights and generated FPGA/model artifacts out of normal Git
  history. Use `init/CROSS_PLATFORM_SYNC.md` and `init/` scripts to restore
  assets on new machines.
- On Windows, prefer a short Vitis workspace path such as `F:\vws` and short
  Vitis component names. Long workspace paths can make Vitis/CMake/Ninja fail
  generated BSP builds when opening `.obj.d` dependency files.
- Keep the local Git safety hook installed with
  `init/install_git_safety_hooks.sh` when possible; it blocks accidental
  commits of model weights and generated large artifacts.
- PC-side Python scripts should be runnable from the repository root and by
  absolute script path. Prefer resolving the model directory relative to the
  script file, for example `Path(__file__).resolve().parents[1]` for scripts
  under `Qwen3-0.6B-Base/pc_testing/`.
- The current first-version target is functionality over performance: run a
  small Qwen3 base model with serial prefill, single-token decode, greedy
  argmax, and the required custom Q4 weight-only path.
- The user's main goal is to learn and practice Verilog/RTL and FPGA PL
  development. Prefer hand-written Verilog/SystemVerilog RTL for PL compute
  blocks. Do not introduce High-Level Synthesis (HLS) flows or HLS C/C++ IP
  unless the user explicitly asks for them.
- PS-side bare-metal code is support infrastructure for control, loading, and
  validation. Keep the project focus on PL design and hardware understanding.
