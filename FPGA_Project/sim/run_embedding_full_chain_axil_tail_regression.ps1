[CmdletBinding()]
param(
    [string]$RunRoot = "",
    [string]$VectorDir = "",
    [ValidateRange(3, 28)]
    [int]$LayerCount = 28,
    [int]$TokenId = -1,
    [string]$CondaEnvironment = "llm_fpga"
)

$ErrorActionPreference = "Stop"

function Invoke-LoggedCommand {
    param(
        [Parameter(Mandatory = $true)] [string]$Executable,
        [Parameter(Mandatory = $true)] [string[]]$Arguments,
        [Parameter(Mandatory = $true)] [string]$LogPath
    )

    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $output = @(& $Executable @Arguments 2>&1 |
        ForEach-Object { "$_" } |
        Tee-Object -FilePath $LogPath)
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedPreference
    if ($exitCode -ne 0) {
        throw "Command failed with exit code $exitCode. See $LogPath"
    }
    return ,$output
}

function Assert-UnderTemp {
    param(
        [Parameter(Mandatory = $true)] [string]$Path,
        [Parameter(Mandatory = $true)] [string]$TempRoot
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
        [Parameter(Mandatory = $true)] [string]$Source,
        [Parameter(Mandatory = $true)] [string]$Destination
    )

    if (!(Test-Path -LiteralPath $Source -PathType Leaf)) {
        throw "Missing vector source: $Source"
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
    if (!(Test-Path -LiteralPath $Destination)) {
        New-Item -ItemType HardLink -Path $Destination -Target $Source | Out-Null
    }
}

function Resolve-VectorFile {
    param(
        [Parameter(Mandatory = $true)] [string]$VectorRoot,
        [Parameter(Mandatory = $true)] [string]$RelativePath
    )

    $path = [System.IO.Path]::GetFullPath((Join-Path $VectorRoot $RelativePath))
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Manifest references a missing vector: $path"
    }
    return $path
}

function Format-SvAddress {
    param([Parameter(Mandatory = $true)] [UInt64]$Value)
    return "64'h{0:X16}" -f $Value
}

function Format-SvPath {
    param([Parameter(Mandatory = $true)] [string]$Path)
    return $Path.Replace('\', '/')
}

function Add-ReadMemLine {
    param(
        [Parameter(Mandatory = $true)] [System.Collections.Generic.List[string]]$Lines,
        [Parameter(Mandatory = $true)] [string]$Path,
        [Parameter(Mandatory = $true)] [string]$ArrayName,
        [Parameter(Mandatory = $true)] [int]$Start,
        [Parameter(Mandatory = $true)] [int]$WordCount
    )

    $svPath = Format-SvPath -Path $Path
    $end = $Start + $WordCount - 1
    $Lines.Add(('                $readmemh("{0}", {1}, {2}, {3});' -f
        $svPath, $ArrayName, $Start, $end))
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$tempRoot = Join-Path $repoRoot "Temp"
if ([string]::IsNullOrWhiteSpace($RunRoot)) {
    $RunRoot = Join-Path $tempRoot "embedding_full_chain_axil_tail_regression"
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
$exporter = Join-Path $repoRoot "Qwen3-0.6B-Base\python_each_module\53_export_embedding_full28_final_chain.py"
$timingChecker = Join-Path $simDir "check_embedding_full_chain_axil_tail_timing.py"
$rtlFiles = @(Get-ChildItem -LiteralPath $rtlDir -Filter "*.sv" -File |
    Sort-Object FullName |
    ForEach-Object { $_.FullName })

$conda = (Get-Command conda -ErrorAction Stop).Source
$xvlog = (Get-Command xvlog -ErrorAction Stop).Source
$xelab = (Get-Command xelab -ErrorAction Stop).Source
$xsim = (Get-Command xsim -ErrorAction Stop).Source
$summary = [System.Collections.Generic.List[string]]::new()
$summary.Add("AXI-Lite tied-Q4 embedding -> full decoder chain -> final tail regression")
$summary.Add("run_id=$runId")
$summary.Add("session_dir=$sessionDir")
$summary.Add("vector_dir=$vectorDirFull")

$savedPycachePrefix = $env:PYTHONPYCACHEPREFIX
$env:PYTHONPYCACHEPREFIX = $pycacheDir
Push-Location $sessionDir
try {
    $manifestPath = Join-Path $vectorDirFull "full_chain_manifest.json"
    if (!(Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        New-Item -ItemType Directory -Force -Path $vectorDirFull | Out-Null
        $exportArgs = @(
            "run", "--no-capture-output", "-n", $CondaEnvironment,
            "python", "-B", $exporter,
            "--output-dir", $vectorDirFull,
            "--layer-count", "$LayerCount"
        )
        if ($TokenId -ge 0) {
            $exportArgs += @("--token-id", "$TokenId")
        }
        Write-Host "[export]  tied-Q4 embedding through $LayerCount layers and full-vocabulary tail"
        $exportOutput = Invoke-LoggedCommand -Executable $conda -Arguments $exportArgs `
            -LogPath (Join-Path $sessionDir "export.log")
        if (($exportOutput -join "`n") -notmatch
            "PASS: exported tied-Q4 embedding through $LayerCount complete layers") {
            throw "Full-chain exporter did not print its PASS marker"
        }
        $summary.Add("PASS vector_export")
    } else {
        $summary.Add("REUSE vector_export :: $manifestPath")
    }

    $manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
    $manifestLayerCount = [int]$manifest.layer_count
    if ($manifestLayerCount -ne $LayerCount) {
        throw "Requested $LayerCount layers but manifest contains $manifestLayerCount"
    }
    if (($manifestLayerCount -lt 3) -or ($manifestLayerCount -gt 28)) {
        throw "Manifest layer_count must be in 3..28"
    }
    if ($manifest.address_audit.status -ne "PASS") {
        throw "Manifest address audit did not pass"
    }

    $syntheticVectors = Join-Path $xsimDir "FPGA_Project\sim\vectors"
    New-Item -ItemType Directory -Force -Path $syntheticVectors | Out-Null
    foreach ($entry in @{
        "embedding_weight_words32.hex" = $manifest.embedding.files.weight
        "embedding_scale_words32.hex" = $manifest.embedding.files.scale
        "embedding_expected_q14_10.hex" = $manifest.embedding.files.expected
        "embedding_token_id.hex" = $manifest.embedding.files.token
    }.GetEnumerator()) {
        Add-VectorLink -Source (Resolve-VectorFile $vectorDirFull $entry.Value) `
            -Destination (Join-Path $syntheticVectors $entry.Key)
    }

    $tailDestinations = @{
        "qmap_final_token_tail_layer2_chained_full_vocab_image_words32.hex" = "qmap_image"
        "qmap_final_token_tail_layer2_chained_full_vocab_expected_words32.hex" = "expected_output"
        "final_rmsnorm_layer2_chained_stage_real_expected.hex" = "final_norm_expected"
        "lm_head_argmax_layer2_chained_full_vocab_real_weight_words32.hex" = "lm_head_weight"
        "lm_head_argmax_layer2_chained_full_vocab_real_scale_words32.hex" = "lm_head_scale"
        "lm_head_argmax_layer2_chained_full_vocab_real_expected_scan_logits_q26.hex" = "expected_logits"
        "lm_head_argmax_layer2_chained_full_vocab_real_scan_base_token.hex" = "scan_base_token"
        "lm_head_argmax_layer2_chained_full_vocab_real_weight_base_addr.hex" = "weight_base_addr"
        "lm_head_argmax_layer2_chained_full_vocab_real_scale_base_addr.hex" = "scale_base_addr"
    }
    foreach ($entry in $tailDestinations.GetEnumerator()) {
        $relativePath = [string]$manifest.final_tail.files.($entry.Value)
        Add-VectorLink -Source (Resolve-VectorFile $vectorDirFull $relativePath) `
            -Destination (Join-Path $syntheticVectors $entry.Key)
    }

    $includeLines = [System.Collections.Generic.List[string]]::new()
    $includeLines.Add("                // Generated from full_chain_manifest.json; keep under Temp.")
    $includeLines.Add("                full_chain_layer_count = $manifestLayerCount;")
    $includeLines.Add("                full_chain_kv_cache_base_addr = $(Format-SvAddress ([UInt64]$manifest.address_formula.kv_cache_base));")
    $includeLines.Add("                tail_qmap_base_addr = $(Format-SvAddress ([UInt64]$manifest.final_tail.qmap_base));")

    $qmapLoads = @(
        @("qkv_image", "full_qkv_qmap", 568320),
        @("input_norm_image", "full_input_norm_qmap", 5120),
        @("frontend_image", "full_frontend_qmap", 8192),
        @("score_image", "full_score_qmap", 5120),
        @("oproj_image", "full_oproj_qmap", 5120),
        @("post_image", "full_post_qmap", 8192),
        @("gate_image", "full_gate_qmap", 14336),
        @("silu_image", "full_silu_qmap", 14336),
        @("down_image", "full_down_qmap", 6144),
        @("residual_image", "full_residual_qmap", 5120)
    )
    $dataLoads = @(
        @("k_cache", "full_k_cache_mem", 5120),
        @("v_cache", "full_v_cache_mem", 5120),
        @("oproj_weight", "full_oproj_weight_mem", 262144),
        @("oproj_scale", "full_oproj_scale_mem", 16384),
        @("gate_weight", "full_gate_weight_mem", 393216),
        @("gate_scale", "full_gate_scale_mem", 24576),
        @("up_weight", "full_up_weight_mem", 393216),
        @("up_scale", "full_up_scale_mem", 24576),
        @("down_weight", "full_down_weight_mem", 393216),
        @("down_scale", "full_down_scale_mem", 24576),
        @("expected_qkv", "full_expected_qkv", 4096),
        @("expected_input_norm", "full_expected_input_norm", 1024),
        @("expected_q_rope", "full_expected_q_rope", 2048),
        @("expected_cache_addr", "full_expected_cache_addr", 2048),
        @("expected_cache_data", "full_expected_cache_data", 2048),
        @("expected_cache_kind", "full_expected_cache_kind", 2048),
        @("expected_attn_out", "full_expected_attn_out", 2048),
        @("expected_oproj", "full_expected_o_proj", 1024),
        @("expected_post_hidden", "full_expected_post_hidden", 1024),
        @("expected_post_norm", "full_expected_post_norm", 1024),
        @("expected_gate", "full_expected_gate", 3072),
        @("expected_up", "full_expected_up", 3072),
        @("expected_silu", "full_expected_silu_hidden", 3072),
        @("expected_down", "full_expected_down", 1024),
        @("expected_layer", "full_expected_layer", 1024)
    )

    for ($layerIndex = 0; $layerIndex -lt $manifestLayerCount; $layerIndex++) {
        $layer = $manifest.layers[$layerIndex]
        $includeLines.Add("                // Layer $layerIndex")
        foreach ($assignment in @{
            "full_qkv_qmap_base" = $layer.qmap_bases.qkv
            "full_input_norm_qmap_base" = $layer.qmap_bases.input_rmsnorm
            "full_frontend_qmap_base" = $layer.qmap_bases.attn_frontend
            "full_score_qmap_base" = $layer.qmap_bases.attn_score_value
            "full_oproj_qmap_base" = $layer.qmap_bases.o_proj
            "full_post_qmap_base" = $layer.qmap_bases.post_attn_norm
            "full_gate_qmap_base" = $layer.qmap_bases.mlp_gate_up
            "full_silu_qmap_base" = $layer.qmap_bases.mlp_silu_mul
            "full_down_qmap_base" = $layer.qmap_bases.mlp_down
            "full_residual_qmap_base" = $layer.qmap_bases.mlp_residual_add
            "full_oproj_weight_base" = $layer.persistent_bases.o_proj_weight
            "full_oproj_scale_base" = $layer.persistent_bases.o_proj_scale
            "full_gate_weight_base" = $layer.persistent_bases.mlp_gate_weight
            "full_gate_scale_base" = $layer.persistent_bases.mlp_gate_scale
            "full_up_weight_base" = $layer.persistent_bases.mlp_up_weight
            "full_up_scale_base" = $layer.persistent_bases.mlp_up_scale
            "full_down_weight_base" = $layer.persistent_bases.mlp_down_weight
            "full_down_scale_base" = $layer.persistent_bases.mlp_down_scale
            "full_input_hidden_base" = $layer.input_hidden_base
            "full_input_norm_output_base" = $layer.write_bases.input_norm
            "full_q_base" = $layer.write_bases.q
            "full_k_base" = $layer.write_bases.k
            "full_v_base" = $layer.write_bases.v
            "full_q_rope_base" = $layer.write_bases.q_rope
            "full_attn_out_base" = $layer.write_bases.attn_out
            "full_o_proj_output_base" = $layer.write_bases.o_proj
            "full_post_hidden_base" = $layer.write_bases.post_hidden
            "full_post_norm_base" = $layer.write_bases.post_norm
            "full_gate_output_base" = $layer.write_bases.gate
            "full_up_output_base" = $layer.write_bases.up
            "full_silu_output_base" = $layer.write_bases.silu
            "full_down_output_base" = $layer.write_bases.down
            "full_layer_output_base" = $layer.write_bases.layer
        }.GetEnumerator()) {
            $includeLines.Add("                $($assignment.Key)[$layerIndex] = $(Format-SvAddress ([UInt64]$assignment.Value));")
        }
        foreach ($load in @($qmapLoads + $dataLoads)) {
            $key = [string]$load[0]
            $arrayName = [string]$load[1]
            $wordCount = [int]$load[2]
            $path = Resolve-VectorFile $vectorDirFull ([string]$layer.files.$key)
            Add-ReadMemLine -Lines $includeLines -Path $path -ArrayName $arrayName `
                -Start ($layerIndex * $wordCount) -WordCount $wordCount
        }
    }

    $includePath = Join-Path $xsimDir "full_chain_vector_loads.svh"
    [System.IO.File]::WriteAllLines($includePath, [string[]]$includeLines,
        [System.Text.Encoding]::ASCII)
    $summary.Add("PASS Temp-only manifest include :: $includePath")

    Push-Location $xsimDir
    try {
        $snapshot = "embedding_full_chain_${manifestLayerCount}_axil_tail_xsim"
        $xvlogArgs = @("--sv", "--relax", "-i", $rtlDir, "-i", $xsimDir)
        foreach ($define in @(
            "QMAP_ONE_TOKEN_TB_WITH_FINAL_TAIL",
            "QMAP_ONE_TOKEN_TB_USE_TOP",
            "QMAP_ONE_TOKEN_TB_USE_MMIO_CONTROL",
            "QMAP_ONE_TOKEN_TB_USE_AXIL_TOP",
            "QMAP_ONE_TOKEN_TB_FULL_CHAIN"
        )) {
            $xvlogArgs += @("-d", $define)
        }
        $xvlogArgs += @("--log", (Join-Path $xsimDir "xvlog.log"))
        $xvlogArgs += $rtlFiles
        $xvlogArgs += @($tbPath)
        Write-Host "[xvlog]  compile $manifestLayerCount-layer AXI-Lite full-chain snapshot"
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
            "-testplusarg", "embedding_full_chain",
            "-testplusarg", "fastmem",
            "-testplusarg", "progress",
            "-testplusarg", "notrace",
            "--log", (Join-Path $xsimDir "xsim.log"),
            "--runall"
        )
        Write-Host "[xsim]   run embedding -> $manifestLayerCount layers -> full-vocabulary tail"
        $xsimOutput = Invoke-LoggedCommand -Executable $xsim -Arguments $xsimArgs `
            -LogPath (Join-Path $xsimDir "xsim_console.log")
        $combinedOutput = $xsimOutput -join "`n"
        if ($combinedOutput -match "(?m)^FAIL:") {
            throw "XSim printed a FAIL line. See $(Join-Path $xsimDir 'xsim.log')"
        }
        if ($combinedOutput -notmatch
            "PASS: AXI-Lite tied-Q4 embedding ran through $manifestLayerCount complete layers") {
            throw "XSim did not print the full-chain PASS marker"
        }
        $summary.Add("PASS xsim")
    } finally {
        Pop-Location
    }

    $eventTrace = Join-Path $xsimDir "embedding_full_chain_events.csv"
    $auditPath = Join-Path $xsimDir "timing_audit.json"
    $auditArgs = @(
        "run", "--no-capture-output", "-n", $CondaEnvironment,
        "python", "-B", $timingChecker, $eventTrace,
        "--manifest", $manifestPath,
        "--xsim-log", (Join-Path $xsimDir "xsim.log"),
        "--output", $auditPath
    )
    Write-Host "[audit]  verify all $manifestLayerCount layer addresses, counts, and response ordering"
    $auditOutput = Invoke-LoggedCommand -Executable $conda -Arguments $auditArgs `
        -LogPath (Join-Path $xsimDir "timing_audit_console.log")
    if (($auditOutput -join "`n") -notmatch
        "PASS: AXI-Lite embedding full-chain timing and counts are exact") {
        throw "Timing auditor did not print its PASS marker"
    }
    $summary.Add("PASS timing_audit :: $auditPath")

    $summaryPath = Join-Path $sessionDir "summary.txt"
    [System.IO.File]::WriteAllLines($summaryPath, [string[]]$summary)
    $summary | ForEach-Object { Write-Host $_ }
    Write-Host "All vectors, links, generated includes, compiler products, logs, traces, and audits are under $sessionDir"
} finally {
    Pop-Location
    $env:PYTHONPYCACHEPREFIX = $savedPycachePrefix
}
