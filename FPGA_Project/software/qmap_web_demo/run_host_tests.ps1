param(
    [string]$TokenizerAsset = ''
)

$ErrorActionPreference = 'Stop'

$sourceDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $sourceDirectory '..\..\..'))
$promptDirectory = Join-Path $repositoryRoot 'FPGA_Project\software\qmap_prompt_demo'
$runtimeDirectory = Join-Path $repositoryRoot 'FPGA_Project\software\qmap_one_token_runtime'
$tokenizerRuntimeDirectory = Join-Path $promptDirectory 'tokenizer_runtime'
if ([string]::IsNullOrWhiteSpace($TokenizerAsset)) {
    $TokenizerAsset = Join-Path $repositoryRoot `
        'Temp\qmap_prompt_demo_tokenizer\qwen3_tokenizer.qtk'
}
$TokenizerAsset = [System.IO.Path]::GetFullPath($TokenizerAsset)
$expectedTokenizerSha256 = `
    'C20242603EF4144E3F3F2EC4BA97C0E9C315AADD41F1BD2C5740E2A7FFA03A7D'
if (-not (Test-Path -LiteralPath $TokenizerAsset -PathType Leaf)) {
    throw "Tokenizer asset not found: $TokenizerAsset"
}
$actualTokenizerSha256 = (Get-FileHash -LiteralPath $TokenizerAsset `
    -Algorithm SHA256).Hash
if ($actualTokenizerSha256 -ne $expectedTokenizerSha256) {
    throw "Tokenizer SHA256 mismatch: expected $expectedTokenizerSha256, got $actualTokenizerSha256"
}
$gcc = Get-Command gcc -ErrorAction Stop
$temporaryDirectory = Join-Path ([System.IO.Path]::GetTempPath()) (
    'qweb_core_' + [System.Guid]::NewGuid().ToString('N'))
$coreTestExecutable = Join-Path $temporaryDirectory 'test_qweb_core.exe'
$jobTestExecutable = Join-Path $temporaryDirectory 'test_qweb_job_host.exe'
$routerTestExecutable = Join-Path $temporaryDirectory 'test_qweb_router_host.exe'
$adapterTestExecutable = Join-Path $temporaryDirectory `
    'test_qweb_lwip_adapter_host.exe'
$entryTestExecutable = Join-Path $temporaryDirectory `
    'test_qweb_board_entry_host.exe'

$warningFlags = @(
    '-std=c11',
    '-O2',
    '-Wall',
    '-Wextra',
    '-Werror',
    '-Wconversion',
    '-Wsign-conversion',
    '-Wshadow',
    '-Wpedantic',
    '-Wstrict-prototypes',
    '-Wmissing-prototypes'
)
$productWarningFlags = $warningFlags + '-Wframe-larger-than=1024'
$productSources = @(
    (Join-Path $promptDirectory 'qot_session.c'),
    (Join-Path $tokenizerRuntimeDirectory 'qtk_tokenizer_runtime.c'),
    (Join-Path $tokenizerRuntimeDirectory 'qtk_text_tokenizer.c'),
    (Join-Path $sourceDirectory 'qweb_http.c'),
    (Join-Path $sourceDirectory 'qweb_api.c'),
    (Join-Path $sourceDirectory 'qweb_job.c'),
    (Join-Path $sourceDirectory 'qweb_router.c'),
    (Join-Path $sourceDirectory 'qweb_lwip_adapter.c'),
    (Join-Path $sourceDirectory 'qweb_board_app.c'),
    (Join-Path $sourceDirectory 'qweb_board_entry.c'),
    (Join-Path $sourceDirectory 'web_assets.c')
)

New-Item -ItemType Directory -Path $temporaryDirectory | Out-Null
try {
    $productIndex = 0
    foreach ($productSource in $productSources) {
        $productObject = Join-Path $temporaryDirectory (
            'product_' + $productIndex.ToString() + '.o')
        & $gcc.Source @productWarningFlags `
            '-c' $productSource `
            '-I' $runtimeDirectory `
            '-I' $promptDirectory `
            '-I' $tokenizerRuntimeDirectory `
            '-I' $sourceDirectory `
            '-I' (Join-Path $sourceDirectory 'test_lwip_mock') `
            '-o' $productObject
        if ($LASTEXITCODE -ne 0) {
            throw "Strict product stack-frame build failed for $productSource with exit code $LASTEXITCODE"
        }
        ++$productIndex
    }

    & $gcc.Source @warningFlags `
        (Join-Path $sourceDirectory 'qweb_http.c') `
        (Join-Path $sourceDirectory 'qweb_api.c') `
        (Join-Path $sourceDirectory 'test_qweb_core.c') `
        '-I' $sourceDirectory `
        '-o' $coreTestExecutable
    if ($LASTEXITCODE -ne 0) {
        throw "Strict qweb core GCC build failed with exit code $LASTEXITCODE"
    }

    & $coreTestExecutable
    if ($LASTEXITCODE -ne 0) {
        throw "qweb core host tests failed with exit code $LASTEXITCODE"
    }

    & $gcc.Source @warningFlags `
        (Join-Path $promptDirectory 'qot_session.c') `
        (Join-Path $tokenizerRuntimeDirectory 'qtk_tokenizer_runtime.c') `
        (Join-Path $tokenizerRuntimeDirectory 'qtk_text_tokenizer.c') `
        (Join-Path $sourceDirectory 'qweb_job.c') `
        (Join-Path $sourceDirectory 'test_qweb_job_host.c') `
        '-I' $runtimeDirectory `
        '-I' $promptDirectory `
        '-I' $tokenizerRuntimeDirectory `
        '-I' $sourceDirectory `
        '-o' $jobTestExecutable
    if ($LASTEXITCODE -ne 0) {
        throw "Strict qweb job GCC build failed with exit code $LASTEXITCODE"
    }

    & $jobTestExecutable $TokenizerAsset
    if ($LASTEXITCODE -ne 0) {
        throw "qweb job host tests failed with exit code $LASTEXITCODE"
    }

    & $gcc.Source @warningFlags `
        (Join-Path $promptDirectory 'qot_session.c') `
        (Join-Path $tokenizerRuntimeDirectory 'qtk_tokenizer_runtime.c') `
        (Join-Path $tokenizerRuntimeDirectory 'qtk_text_tokenizer.c') `
        (Join-Path $sourceDirectory 'qweb_http.c') `
        (Join-Path $sourceDirectory 'qweb_api.c') `
        (Join-Path $sourceDirectory 'qweb_job.c') `
        (Join-Path $sourceDirectory 'qweb_router.c') `
        (Join-Path $sourceDirectory 'test_qweb_router_host.c') `
        '-I' $runtimeDirectory `
        '-I' $promptDirectory `
        '-I' $tokenizerRuntimeDirectory `
        '-I' $sourceDirectory `
        '-o' $routerTestExecutable
    if ($LASTEXITCODE -ne 0) {
        throw "Strict qweb router GCC build failed with exit code $LASTEXITCODE"
    }

    & $routerTestExecutable
    if ($LASTEXITCODE -ne 0) {
        throw "qweb router host tests failed with exit code $LASTEXITCODE"
    }

    & $gcc.Source @warningFlags `
        (Join-Path $sourceDirectory 'qweb_http.c') `
        (Join-Path $sourceDirectory 'qweb_api.c') `
        (Join-Path $sourceDirectory 'qweb_lwip_adapter.c') `
        (Join-Path $sourceDirectory 'qweb_board_app.c') `
        (Join-Path $sourceDirectory 'test_qweb_lwip_adapter_host.c') `
        '-I' $runtimeDirectory `
        '-I' $promptDirectory `
        '-I' $tokenizerRuntimeDirectory `
        '-I' $sourceDirectory `
        '-I' (Join-Path $sourceDirectory 'test_lwip_mock') `
        '-o' $adapterTestExecutable
    if ($LASTEXITCODE -ne 0) {
        throw "Strict qweb raw-lwIP adapter GCC build failed with exit code $LASTEXITCODE"
    }

    & $adapterTestExecutable
    if ($LASTEXITCODE -ne 0) {
        throw "qweb raw-lwIP adapter host tests failed with exit code $LASTEXITCODE"
    }

    & $gcc.Source @warningFlags `
        '-DQWEB_BOARD_ENTRY_HOST_TEST' `
        (Join-Path $sourceDirectory 'qweb_board_entry.c') `
        (Join-Path $sourceDirectory 'test_qweb_board_entry_host.c') `
        '-I' $runtimeDirectory `
        '-I' $promptDirectory `
        '-I' $tokenizerRuntimeDirectory `
        '-I' $sourceDirectory `
        '-I' (Join-Path $sourceDirectory 'test_lwip_mock') `
        '-o' $entryTestExecutable
    if ($LASTEXITCODE -ne 0) {
        throw "Strict qweb board-entry GCC build failed with exit code $LASTEXITCODE"
    }

    & $entryTestExecutable
    if ($LASTEXITCODE -ne 0) {
        throw "qweb board-entry host assembly test failed with exit code $LASTEXITCODE"
    }
} finally {
    if (Test-Path -LiteralPath $temporaryDirectory) {
        $resolvedTemporary = (Resolve-Path -LiteralPath $temporaryDirectory).Path
        $resolvedSystemTemp = (Resolve-Path -LiteralPath (
            [System.IO.Path]::GetTempPath())).Path.TrimEnd('\')
        if (-not $resolvedTemporary.StartsWith(
                $resolvedSystemTemp + '\',
                [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to remove non-temporary path: $resolvedTemporary"
        }
        Remove-Item -LiteralPath $resolvedTemporary -Recurse -Force
    }
}
