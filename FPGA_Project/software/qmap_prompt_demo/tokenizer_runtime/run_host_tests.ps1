param(
    [string]$AssetPath = ""
)

$ErrorActionPreference = "Stop"
$runtimeDir = $PSScriptRoot
$repoRoot = (Resolve-Path (Join-Path $runtimeDir "..\..\..\..")).Path
$tempDir = Join-Path $repoRoot "Temp\qmap_prompt_demo_tokenizer"

if ([string]::IsNullOrWhiteSpace($AssetPath)) {
    $AssetPath = Join-Path $tempDir "qwen3_tokenizer.qtk"
}
$AssetPath = (Resolve-Path $AssetPath).Path
$referencePath = Join-Path $tempDir "qtk_host_reference.bin"
$textReferencePath = Join-Path $tempDir "qtk_text_reference.bin"
$testExecutable = Join-Path $tempDir "test_qtk_tokenizer_runtime.exe"
$textTestExecutable = Join-Path $tempDir "test_qtk_text_tokenizer.exe"
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

& conda run -n llm_fpga python `
    (Join-Path $runtimeDir "generate_host_reference.py") `
    --output $referencePath
if ($LASTEXITCODE -ne 0) {
    throw "Python reference generation failed with exit code $LASTEXITCODE"
}
& conda run -n llm_fpga python `
    (Join-Path $runtimeDir "generate_text_reference.py") `
    --output $textReferencePath
if ($LASTEXITCODE -ne 0) {
    throw "Python text reference generation failed with exit code $LASTEXITCODE"
}

$warningFlags = @(
    "-std=c11",
    "-O2",
    "-Wall",
    "-Wextra",
    "-Wpedantic",
    "-Werror",
    "-Wconversion",
    "-Wsign-conversion",
    "-Wshadow",
    "-Wstrict-prototypes",
    "-Wmissing-prototypes"
)
& gcc @warningFlags `
    (Join-Path $runtimeDir "qtk_tokenizer_runtime.c") `
    (Join-Path $runtimeDir "test_tokenizer_runtime_host.c") `
    -I $runtimeDir `
    -o $testExecutable
if ($LASTEXITCODE -ne 0) {
    throw "Strict GCC build failed with exit code $LASTEXITCODE"
}

& $testExecutable $AssetPath $referencePath
if ($LASTEXITCODE -ne 0) {
    throw "C host test failed with exit code $LASTEXITCODE"
}

& gcc @warningFlags `
    (Join-Path $runtimeDir "qtk_tokenizer_runtime.c") `
    (Join-Path $runtimeDir "qtk_text_tokenizer.c") `
    (Join-Path $runtimeDir "test_text_tokenizer_host.c") `
    -I $runtimeDir `
    -o $textTestExecutable
if ($LASTEXITCODE -ne 0) {
    throw "Strict GCC text-tokenizer build failed with exit code $LASTEXITCODE"
}

& $textTestExecutable $AssetPath $textReferencePath
if ($LASTEXITCODE -ne 0) {
    throw "C text-tokenizer host test failed with exit code $LASTEXITCODE"
}
