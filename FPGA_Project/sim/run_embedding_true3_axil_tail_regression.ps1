[CmdletBinding()]
param(
    [string]$RunRoot = "",
    [string]$VectorDir = "",
    [int]$TokenId = -1,
    [string]$CondaEnvironment = "llm_fpga"
)

$ErrorActionPreference = "Stop"

function Invoke-LoggedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [Parameter(Mandatory = $true)]
        [string]$LogPath
    )

    $savedErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $output = @(& $Executable @Arguments 2>&1 |
        ForEach-Object { "$_" } |
        Tee-Object -FilePath $LogPath)
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedErrorActionPreference
    if ($exitCode -ne 0) {
        throw "Command failed with exit code $exitCode. See $LogPath"
    }
    return ,$output
}

function Assert-UnderTemp {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$TempRoot
    )

    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $tempFull = [System.IO.Path]::GetFullPath($TempRoot)
    $prefix = $tempFull.TrimEnd([System.IO.Path]::DirectorySeparatorChar) +
        [System.IO.Path]::DirectorySeparatorChar
    if (($fullPath -ne $tempFull) -and
        !$fullPath.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Path must stay under $tempFull`: $fullPath"
    }
    return $fullPath
}

function Add-VectorLink {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Source,
        [Parameter(Mandatory = $true)]
        [string]$Destination
    )

    if (!(Test-Path -LiteralPath $Source -PathType Leaf)) {
        throw "Missing vector source: $Source"
    }
    $parent = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    if (Test-Path -LiteralPath $Destination) {
        return
    }
    New-Item -ItemType HardLink -Path $Destination -Target $Source | Out-Null
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$tempRoot = Join-Path $repoRoot "Temp"
if ([string]::IsNullOrWhiteSpace($RunRoot)) {
    $RunRoot = Join-Path $tempRoot "embedding_true3_axil_tail_regression"
} elseif (![System.IO.Path]::IsPathRooted($RunRoot)) {
    $RunRoot = Join-Path $repoRoot $RunRoot
}
$runRootFull = Assert-UnderTemp -Path $RunRoot -TempRoot $tempRoot

$runId = Get-Date -Format "yyyyMMdd_HHmmss"
$sessionDir = Join-Path $runRootFull $runId
$xsimDir = Join-Path $sessionDir "xsim"
$pycacheDir = Join-Path $sessionDir "pycache"
New-Item -ItemType Directory -Force -Path $sessionDir,$xsimDir,$pycacheDir | Out-Null

if ([string]::IsNullOrWhiteSpace($VectorDir)) {
    $vectorDirFull = Join-Path $sessionDir "vectors"
} elseif (![System.IO.Path]::IsPathRooted($VectorDir)) {
    $vectorDirFull = Join-Path $repoRoot $VectorDir
} else {
    $vectorDirFull = $VectorDir
}
$vectorDirFull = Assert-UnderTemp -Path $vectorDirFull -TempRoot $tempRoot

$rtlDir = Join-Path $repoRoot "FPGA_Project\rtl"
$simDir = Join-Path $repoRoot "FPGA_Project\sim"
$tbPath = Join-Path $simDir "tb_qmap_one_token_layer_scheduler.sv"
$exporter = Join-Path $repoRoot "Qwen3-0.6B-Base\python_each_module\52_export_embedding_true3_final_chain.py"
$timingChecker = Join-Path $simDir "check_embedding_true3_axil_tail_timing.py"
. (Join-Path $simDir "load_rtl_manifest.ps1")
$rtlBuild = Get-RtlBuildManifest -RtlDir $rtlDir
$rtlFiles = $rtlBuild.SourceFiles
$rtlIncludeDirs = $rtlBuild.IncludeDirs

$conda = (Get-Command conda -ErrorAction Stop).Source
$xvlog = (Get-Command xvlog -ErrorAction Stop).Source
$xelab = (Get-Command xelab -ErrorAction Stop).Source
$xsim = (Get-Command xsim -ErrorAction Stop).Source
$summary = [System.Collections.Generic.List[string]]::new()
$summary.Add("AXI-Lite tied-Q4 embedding -> true3 -> final tail regression")
$summary.Add("run_id=$runId")
$summary.Add("session_dir=$sessionDir")
$summary.Add("vector_dir=$vectorDirFull")

$savedPycachePrefix = $env:PYTHONPYCACHEPREFIX
$env:PYTHONPYCACHEPREFIX = $pycacheDir
Push-Location $sessionDir
try {
    $manifestPath = Join-Path $vectorDirFull "true3_final_chain_manifest.json"
    if (!(Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        New-Item -ItemType Directory -Force -Path $vectorDirFull | Out-Null
        $exportArgs = @(
            "run", "--no-capture-output", "-n", $CondaEnvironment,
            "python", "-B", $exporter, "--output-dir", $vectorDirFull
        )
        if ($TokenId -ge 0) {
            $exportArgs += @("--token-id", "$TokenId")
        }
        Write-Host "[export]  tied-Q4 embedding through true3 and full-vocabulary tail"
        $exportOutput = Invoke-LoggedCommand -Executable $conda -Arguments $exportArgs `
            -LogPath (Join-Path $sessionDir "export.log")
        if (($exportOutput -join "`n") -notmatch
            "PASS: exported tied-Q4 embedding through three complete layers and final tail") {
            throw "True3 exporter did not print its PASS marker"
        }
        $summary.Add("PASS vector_export")
    } else {
        $summary.Add("REUSE vector_export :: $manifestPath")
    }

    $syntheticVectors = Join-Path $xsimDir "FPGA_Project\sim\vectors"
    $syntheticArtifacts = Join-Path $xsimDir "artifacts\test_vectors\qwen3_0p6b_qmap_v1"
    New-Item -ItemType Directory -Force -Path $syntheticVectors,$syntheticArtifacts | Out-Null

    $layer0Sim = Join-Path $vectorDirFull "embedding_layer0\layer0\sim_vectors"
    foreach ($source in Get-ChildItem -LiteralPath $layer0Sim -File) {
        $destinationName = $null
        if ($source.Name.StartsWith("qmap_layer0_chained_")) {
            $destinationName = "qmap_" + $source.Name.Substring("qmap_layer0_chained_".Length)
        } elseif ($source.Name.StartsWith("layer0_chained_")) {
            $destinationName = $source.Name.Substring("layer0_chained_".Length)
        }
        if ($null -ne $destinationName) {
            Add-VectorLink -Source $source.FullName `
                -Destination (Join-Path $syntheticVectors $destinationName)
        }
    }

    foreach ($layerName in @("layer1", "layer2")) {
        $layerSim = Join-Path $vectorDirFull "$layerName\sim_vectors"
        foreach ($source in Get-ChildItem -LiteralPath $layerSim -File) {
            Add-VectorLink -Source $source.FullName `
                -Destination (Join-Path $syntheticVectors $source.Name)
        }
    }

    $embeddingDir = Join-Path $vectorDirFull "embedding_layer0\embedding"
    foreach ($name in @(
        "embedding_weight_words32.hex",
        "embedding_scale_words32.hex",
        "embedding_expected_q14_10.hex",
        "embedding_token_id.hex"
    )) {
        Add-VectorLink -Source (Join-Path $embeddingDir $name) `
            -Destination (Join-Path $syntheticVectors $name)
    }

    $layer0Qmap = Join-Path $vectorDirFull "embedding_layer0\layer0\qmap"
    $layer1Qmap = Join-Path $vectorDirFull "layer1\qmap"
    $layer2Qmap = Join-Path $vectorDirFull "layer2\qmap"
    $qkvLinks = @{
        "layer0_qkv_projection_full_image_words32.hex" =
            (Join-Path $layer0Qmap "layer0_qkv_from_embedding_rmsnorm_full_image_words32.hex")
        "layer0_qkv_projection_full_expected_words32.hex" =
            (Join-Path $layer0Qmap "layer0_qkv_from_embedding_rmsnorm_full_expected_words32.hex")
        "layer1_qkv_from_layer0_rtl_full_image_words32.hex" =
            (Join-Path $layer1Qmap "layer1_qkv_from_layer0_rtl_full_image_words32.hex")
        "layer1_qkv_from_layer0_rtl_full_expected_words32.hex" =
            (Join-Path $layer1Qmap "layer1_qkv_from_layer0_rtl_full_expected_words32.hex")
        "layer2_qkv_from_layer1_rtl_full_image_words32.hex" =
            (Join-Path $layer2Qmap "layer2_qkv_from_layer1_rtl_full_image_words32.hex")
        "layer2_qkv_from_layer1_rtl_full_expected_words32.hex" =
            (Join-Path $layer2Qmap "layer2_qkv_from_layer1_rtl_full_expected_words32.hex")
    }
    foreach ($entry in $qkvLinks.GetEnumerator()) {
        Add-VectorLink -Source $entry.Value `
            -Destination (Join-Path $syntheticArtifacts $entry.Key)
    }
    Add-VectorLink `
        -Source (Join-Path $layer0Qmap "layer0_qkv_from_embedding_rmsnorm_full_image_words32.hex") `
        -Destination (Join-Path $syntheticVectors "qmap_qkv_projection_from_input_rmsnorm_image_words32.hex")
    Add-VectorLink `
        -Source (Join-Path $layer0Qmap "layer0_qkv_from_embedding_rmsnorm_full_expected_words32.hex") `
        -Destination (Join-Path $syntheticVectors "qmap_qkv_projection_expected_from_input_rmsnorm_words32.hex")

    $tailSim = Join-Path $vectorDirFull "final_tail\sim_vectors"
    $tailLinks = @{
        "qmap_final_token_tail_layer2_chained_full_vocab_image_words32.hex" =
            "qmap_final_token_tail_embedding_true3_image_words32.hex"
        "qmap_final_token_tail_layer2_chained_full_vocab_expected_words32.hex" =
            "qmap_final_token_tail_embedding_true3_expected_words32.hex"
        "final_rmsnorm_layer2_chained_stage_real_expected.hex" =
            "final_rmsnorm_from_embedding_true3_expected.hex"
        "lm_head_argmax_layer2_chained_full_vocab_real_weight_words32.hex" =
            "lm_head_full_vocab_from_embedding_true3_weight_words32.hex"
        "lm_head_argmax_layer2_chained_full_vocab_real_scale_words32.hex" =
            "lm_head_full_vocab_from_embedding_true3_scale_words32.hex"
        "lm_head_argmax_layer2_chained_full_vocab_real_expected_scan_logits_q26.hex" =
            "lm_head_full_vocab_from_embedding_true3_expected_scan_logits_q26.hex"
        "lm_head_argmax_layer2_chained_full_vocab_real_scan_base_token.hex" =
            "lm_head_full_vocab_from_embedding_true3_scan_base_token.hex"
        "lm_head_argmax_layer2_chained_full_vocab_real_weight_base_addr.hex" =
            "lm_head_full_vocab_from_embedding_true3_weight_base_addr.hex"
        "lm_head_argmax_layer2_chained_full_vocab_real_scale_base_addr.hex" =
            "lm_head_full_vocab_from_embedding_true3_scale_base_addr.hex"
    }
    foreach ($entry in $tailLinks.GetEnumerator()) {
        Add-VectorLink -Source (Join-Path $tailSim $entry.Value) `
            -Destination (Join-Path $syntheticVectors $entry.Key)
    }

    $tbText = Get-Content -LiteralPath $tbPath -Raw
    $vectorMatches = [regex]::Matches(
        $tbText,
        '\$readmemh\("FPGA_Project/sim/vectors/([^"\r\n]+)"'
    )
    foreach ($match in $vectorMatches) {
        $required = Join-Path $syntheticVectors $match.Groups[1].Value
        if (!(Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "Synthetic vector map is incomplete: $required"
        }
    }
    $artifactMatches = [regex]::Matches(
        $tbText,
        '\$readmemh\("artifacts/test_vectors/qwen3_0p6b_qmap_v1/([^"\r\n]+)"'
    )
    foreach ($match in $artifactMatches) {
        $required = Join-Path $syntheticArtifacts $match.Groups[1].Value
        if (!(Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "Synthetic artifact map is incomplete: $required"
        }
    }
    $summary.Add("PASS Temp-only vector map")

    Push-Location $xsimDir
    try {
        $snapshot = "embedding_true3_axil_tail_xsim"
        $xvlogArgs = @("--sv", "--relax")
        foreach ($includeDir in $rtlIncludeDirs) {
            $xvlogArgs += @("-i", $includeDir)
        }
        foreach ($define in @(
            "QMAP_ONE_TOKEN_TB_WITH_FINAL_TAIL",
            "QMAP_ONE_TOKEN_TB_USE_TOP",
            "QMAP_ONE_TOKEN_TB_USE_MMIO_CONTROL",
            "QMAP_ONE_TOKEN_TB_USE_AXIL_TOP"
        )) {
            $xvlogArgs += @("-d", $define)
        }
        $xvlogArgs += @("--log", (Join-Path $xsimDir "xvlog.log"))
        $xvlogArgs += $rtlFiles
        $xvlogArgs += @($tbPath)
        Write-Host "[xvlog]  compile AXI-Lite embedding true3 top-to-tail snapshot"
        Invoke-LoggedCommand -Executable $xvlog -Arguments $xvlogArgs `
            -LogPath (Join-Path $xsimDir "xvlog_console.log") | Out-Null

        $xelabArgs = @(
            "--relax", "--snapshot", $snapshot,
            "--log", (Join-Path $xsimDir "xelab.log"),
            "tb_qmap_one_token_layer_scheduler"
        )
        Write-Host "[xelab]  elaborate $snapshot"
        Invoke-LoggedCommand -Executable $xelab -Arguments $xelabArgs `
            -LogPath (Join-Path $xsimDir "xelab_console.log") | Out-Null

        $xsimArgs = @(
            $snapshot,
            "-testplusarg", "true3_mmio_top_tail_only",
            "-testplusarg", "embedding_true3",
            "-testplusarg", "fastmem",
            "-testplusarg", "progress",
            "-testplusarg", "notrace",
            "--log", (Join-Path $xsimDir "xsim.log"),
            "--runall"
        )
        Write-Host "[xsim]   run embedding -> true3 -> full-vocabulary tail"
        $xsimOutput = Invoke-LoggedCommand -Executable $xsim -Arguments $xsimArgs `
            -LogPath (Join-Path $xsimDir "xsim_console.log")
        $combinedOutput = $xsimOutput -join "`n"
        if ($combinedOutput -match "(?m)^FAIL:") {
            throw "XSim printed a FAIL line. See $(Join-Path $xsimDir 'xsim.log')"
        }
        if ($combinedOutput -notmatch
            "(?m)^PASS: AXI-Lite tied-Q4 embedding ran through three complete layers and the full-vocabulary final-token tail exactly\.$") {
            throw "XSim did not print the embedding true3 PASS marker"
        }
        $summary.Add("PASS xsim")
    } finally {
        Pop-Location
    }

    $eventTrace = Join-Path $xsimDir "embedding_true3_events.csv"
    $auditPath = Join-Path $xsimDir "timing_audit.json"
    $auditArgs = @(
        "run", "--no-capture-output", "-n", $CondaEnvironment,
        "python", "-B", $timingChecker, $eventTrace,
        "--manifest", $manifestPath,
        "--xsim-log", (Join-Path $xsimDir "xsim.log"),
        "--output", $auditPath
    )
    Write-Host "[audit]  verify exact addresses, counts, and response ordering"
    $auditOutput = Invoke-LoggedCommand -Executable $conda -Arguments $auditArgs `
        -LogPath (Join-Path $xsimDir "timing_audit_console.log")
    if (($auditOutput -join "`n") -notmatch
        "PASS: AXI-Lite embedding true3 final-tail timing and counts are exact\.") {
        throw "Timing auditor did not print its PASS marker"
    }
    $summary.Add("PASS timing_audit :: $auditPath")

    $summaryPath = Join-Path $sessionDir "summary.txt"
    [System.IO.File]::WriteAllLines($summaryPath, [string[]]$summary)
    $summary | ForEach-Object { Write-Host $_ }
    Write-Host "All vectors, links, compiler products, logs, traces, and audits are under $sessionDir"
} finally {
    Pop-Location
    $env:PYTHONPYCACHEPREFIX = $savedPycachePrefix
}
