# Final Demo Source Snapshot

This directory is the compact source handoff for the portable QWEB demo.

- `FPGA_Project/rtl/` contains the handwritten PL inference RTL.
- `FPGA_Project/Vivado_Project/` preserves the source-bearing XPR, block-design,
  constraint, IP-wrapper, and Tcl layout.
- `FPGA_Project/software/qmap_one_token_runtime/` contains the PS-to-PL runtime
  contract.
- `FPGA_Project/software/qmap_prompt_demo/` contains prompt tokenization,
  session, UART, and PC Web Serial fallback source.
- `FPGA_Project/software/qmap_web_demo/` contains the board-hosted HTTP/Web UI,
  Vitis workspace builder dependencies, and Motorcomm YT8521 lwIP patch source.

Host-only tests, temporary traces, generated Vitis workspaces, and intermediate
build products are intentionally excluded. Component READMEs are copied from
the canonical tree and can mention tests that remain available only in the full
development checkout. The canonical development copies remain at the repository
root under `FPGA_Project/`; this snapshot is the clean demo handoff surface.

Before attempting a Web-app rebuild, audit the copied XSA, source set, and the
release tokenizer from the repository root. The workspace path below is a new,
nonexistent path outside the repository; choose another new path for a real
build:

```powershell
$DemoRoot = (Resolve-Path `
  '.\lmdeploy\qwen3-0p6b-q4-qweb-demo').Path
$env:QWEB_NETWORK_XSA = Join-Path $DemoRoot `
  'board\network_input_0af1257442a68a6beb31d94811713f2ae8e6af63e0a85dd497146405e37406cf.xsa'
$env:QWEB_TOKENIZER_ASSET = Join-Path $DemoRoot `
  'assets\qwen3_tokenizer.qtk'
$env:QWEB_VITIS_WORKSPACE = Join-Path ([IO.Path]::GetTempPath()) `
  ('qweb_vitis_' + [Guid]::NewGuid().ToString('N'))
conda run -n llm_fpga python `
  "$DemoRoot\source\FPGA_Project\software\qmap_web_demo\create_network_vitis_workspace.py" `
  --with-web-app --check-only --json
```

Remove `--check-only` only after installing the compatible AMD Vitis tools and
choosing a new persistent workspace path outside the repository.
