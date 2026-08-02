[CmdletBinding()]
param(
    [string]$RunRoot = "",
    [string]$VectorDir = "",
    [ValidateRange(3, 28)]
    [int]$LayerCount = 28,
    [int]$TokenId = -1,
    [switch]$RuntimeContext,
    [switch]$PersistentTwoToken,
    [string]$PersistentVectorDir = "",
    [int]$PersistentFeedbackTokenId = -1,
    [string]$RtlSourceDir = "",
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

function Resolve-ReferencedFile {
    param(
        [Parameter(Mandatory = $true)] [string]$VectorRoot,
        [Parameter(Mandatory = $true)] [string]$RepoRoot,
        [Parameter(Mandatory = $true)] [string]$Reference
    )

    $candidates = [System.Collections.Generic.List[string]]::new()
    if ([System.IO.Path]::IsPathRooted($Reference)) {
        $candidates.Add([System.IO.Path]::GetFullPath($Reference))
    } else {
        $candidates.Add([System.IO.Path]::GetFullPath((Join-Path $VectorRoot $Reference)))
        $candidates.Add([System.IO.Path]::GetFullPath((Join-Path $RepoRoot $Reference)))
    }
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }
    throw "Manifest references a missing file: $Reference"
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

function Write-HexSlice {
    param(
        [Parameter(Mandatory = $true)] [string]$Source,
        [Parameter(Mandatory = $true)] [string]$Destination,
        [Parameter(Mandatory = $true)] [int]$Start,
        [Parameter(Mandatory = $true)] [int]$Count
    )

    $lines = [System.IO.File]::ReadAllLines($Source)
    if (($Start -lt 0) -or ($Count -le 0) -or (($Start + $Count) -gt $lines.Length)) {
        throw "Hex slice $Start+$Count escapes $Source ($($lines.Length) lines)"
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
    [System.IO.File]::WriteAllLines(
        $Destination,
        [string[]]$lines[$Start..($Start + $Count - 1)],
        [System.Text.Encoding]::ASCII
    )
}

function Format-SvWord32 {
    param([Parameter(Mandatory = $true)] [UInt32]$Value)
    return "32'h{0:X8}" -f $Value
}

function Format-SvSignedDecimal {
    param(
        [Parameter(Mandatory = $true)] [Int64]$Value,
        [Parameter(Mandatory = $true)] [int]$Width
    )
    if ($Value -lt 0) {
        return "-${Width}'sd$(-$Value)"
    }
    return "${Width}'sd$Value"
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
if ($PersistentTwoToken -and !$RuntimeContext) {
    throw "-PersistentTwoToken requires -RuntimeContext"
}
if ($PersistentTwoToken -and ($TokenId -lt 0)) {
    $TokenId = 374
}
if ($PersistentTwoToken) {
    if ([string]::IsNullOrWhiteSpace($PersistentVectorDir)) {
        $persistentVectorDirFull = Join-Path $sessionDir "persistent_vectors"
    } elseif (![System.IO.Path]::IsPathRooted($PersistentVectorDir)) {
        $persistentVectorDirFull = Join-Path $repoRoot $PersistentVectorDir
    } else {
        $persistentVectorDirFull = $PersistentVectorDir
    }
    $persistentVectorDirFull = Assert-UnderTemp -Path $persistentVectorDirFull -TempRoot $tempRoot
}

if ([string]::IsNullOrWhiteSpace($RtlSourceDir)) {
    $rtlDir = Join-Path $repoRoot "FPGA_Project\rtl"
} elseif ([System.IO.Path]::IsPathRooted($RtlSourceDir)) {
    $rtlDir = [System.IO.Path]::GetFullPath($RtlSourceDir)
} else {
    $rtlDir = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $RtlSourceDir))
}
if (!(Test-Path -LiteralPath $rtlDir -PathType Container)) {
    throw "RTL source directory does not exist: $rtlDir"
}
$simDir = Join-Path $repoRoot "FPGA_Project\sim"
$tbPath = Join-Path $simDir "tb_qmap_one_token_layer_scheduler.sv"
$exporter = Join-Path $repoRoot "Qwen3-0.6B-Base\python_each_module\53_export_embedding_full28_final_chain.py"
$persistentExporter = Join-Path $repoRoot "Qwen3-0.6B-Base\python_each_module\56_export_persistent_multitoken_layer_golden.py"
$timingChecker = Join-Path $simDir "check_embedding_full_chain_axil_tail_timing.py"
$persistentTimingChecker = Join-Path $simDir "check_persistent_two_token_full_chain.py"
. (Join-Path $simDir "load_rtl_manifest.ps1")
$rtlBuild = Get-RtlBuildManifest -RtlDir $rtlDir
$rtlFiles = $rtlBuild.SourceFiles
$rtlIncludeDirs = $rtlBuild.IncludeDirs

$conda = (Get-Command conda -ErrorAction Stop).Source
$xvlog = (Get-Command xvlog -ErrorAction Stop).Source
$xelab = (Get-Command xelab -ErrorAction Stop).Source
$xsim = (Get-Command xsim -ErrorAction Stop).Source
$summary = [System.Collections.Generic.List[string]]::new()
$summary.Add("AXI-Lite tied-Q4 embedding -> full decoder chain -> final tail regression")
$summary.Add("run_id=$runId")
$summary.Add("session_dir=$sessionDir")
$summary.Add("vector_dir=$vectorDirFull")
$summary.Add("rtl_source_dir=$rtlDir")
$summary.Add("rtl_source_count=$($rtlFiles.Count)")
$summary.Add("runtime_context=$($RuntimeContext.IsPresent)")
$summary.Add("persistent_two_token=$($PersistentTwoToken.IsPresent)")
if ($PersistentTwoToken) {
    $summary.Add("persistent_vector_dir=$persistentVectorDirFull")
}

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
        if ($RuntimeContext) {
            $exportArgs += "--runtime-rope-table"
        }
        $modeLabel = if ($RuntimeContext) { "runtime-context" } else { "legacy" }
        Write-Host "[export]  tied-Q4 embedding through $LayerCount layers and full-vocabulary tail ($modeLabel)"
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

    $manifestRuntimeEnabled = ($null -ne $manifest.runtime_context) -and
        ([bool]$manifest.runtime_context.enabled)
    if ($manifestRuntimeEnabled -ne $RuntimeContext.IsPresent) {
        throw "Requested RuntimeContext=$($RuntimeContext.IsPresent) but manifest runtime_context.enabled=$manifestRuntimeEnabled"
    }

    $runtimePosition = 0
    $runtimeRopeHex = $null
    $runtimeRopeCosBase = [UInt64]0
    $runtimeRopeSinBase = [UInt64]0
    $runtimeHiddenABase = [UInt64]0
    $runtimeHiddenBBase = [UInt64]0
    $runtimeLastHiddenBase = [UInt64]$manifest.final_tail.runtime_hidden_override
    $embeddingOutputBase = [UInt64]$manifest.embedding.output_base
    if ($RuntimeContext) {
        $runtimeRopeCosBase = [UInt64]$manifest.runtime_context.rope_cos_base
        $runtimeRopeSinBase = [UInt64]$manifest.runtime_context.rope_sin_base
        $runtimeHiddenABase = [UInt64]$manifest.runtime_context.hidden_a_base
        $runtimeHiddenBBase = [UInt64]$manifest.runtime_context.hidden_b_base
        $runtimeLastHiddenBase = [UInt64]$manifest.runtime_context.last_layer_hidden_base
        $runtimeRopePlaneBytes = [UInt64](256 * 128 * 4)
        if (($runtimeRopeCosBase -eq 0) -or
            ($runtimeRopeSinBase -ne ($runtimeRopeCosBase + $runtimeRopePlaneBytes))) {
            throw "Runtime RoPE cos/sin bases are missing or non-contiguous"
        }
        if (($runtimeHiddenABase -eq 0) -or ($runtimeHiddenBBase -eq 0) -or
            ($runtimeHiddenABase -eq $runtimeHiddenBBase) -or
            ((($runtimeHiddenABase -bor $runtimeHiddenBBase) -band 3) -ne 0)) {
            throw "Runtime hidden A/B bases must be distinct, non-zero, and word aligned"
        }
        if (($manifest.runtime_context.rope_shape[0] -ne 256) -or
            ($manifest.runtime_context.rope_shape[1] -ne 128) -or
            ($manifest.runtime_context.rope_row_stride_bytes -ne 512)) {
            throw "Runtime RoPE manifest shape/stride contract mismatch"
        }
        if ($embeddingOutputBase -ne $runtimeHiddenABase) {
            throw "Runtime embedding output must select hidden buffer A"
        }
        if ([UInt64]$manifest.embedding.descriptor_golden_output_base -ne
            [UInt64]$manifest.layers[0].input_hidden_base) {
            throw "Embedding descriptor-golden output no longer matches Layer 0 static input"
        }

        $expectedLastHiddenBase = if (($manifestLayerCount % 2) -eq 0) {
            $runtimeHiddenABase
        } else {
            $runtimeHiddenBBase
        }
        if (($runtimeLastHiddenBase -ne $expectedLastHiddenBase) -or
            ([UInt64]$manifest.final_tail.runtime_hidden_override -ne $expectedLastHiddenBase)) {
            throw "Runtime final hidden address does not follow A/B layer parity"
        }
        if ([UInt64]$manifest.final_tail.descriptor_golden_hidden_base -ne
            [UInt64]$manifest.layers[$manifestLayerCount - 1].output_hidden_base) {
            throw "Final-tail descriptor-golden hidden base changed unexpectedly"
        }
        $runtimeRopeHex = Resolve-VectorFile $vectorDirFull `
            ([string]$manifest.runtime_context.files.sim_hex)

        for ($layerIndex = 0; $layerIndex -lt $manifestLayerCount; $layerIndex++) {
            $layer = $manifest.layers[$layerIndex]
            $expectedRuntimeInput = if (($layerIndex % 2) -eq 0) {
                $runtimeHiddenABase
            } else {
                $runtimeHiddenBBase
            }
            $expectedRuntimeOutput = if (($layerIndex % 2) -eq 0) {
                $runtimeHiddenBBase
            } else {
                $runtimeHiddenABase
            }
            if (([UInt64]$layer.runtime_input_hidden_base -ne $expectedRuntimeInput) -or
                ([UInt64]$layer.runtime_output_hidden_base -ne $expectedRuntimeOutput)) {
                throw "Layer $layerIndex runtime hidden A/B parity mismatch"
            }
            if (([UInt64]$layer.descriptor_input_hidden_base -ne [UInt64]$layer.input_hidden_base) -or
                ([UInt64]$layer.descriptor_output_hidden_base -ne [UInt64]$layer.output_hidden_base)) {
                throw "Layer $layerIndex descriptor-golden hidden addresses changed unexpectedly"
            }

            $layerManifestPath = Resolve-VectorFile $vectorDirFull ([string]$layer.manifest)
            $layerManifest = Get-Content -Raw -LiteralPath $layerManifestPath | ConvertFrom-Json
            $frontendManifestPath = Resolve-ReferencedFile -VectorRoot $vectorDirFull `
                -RepoRoot $repoRoot -Reference ([string]$layerManifest.files.frontend.manifest)
            $frontendManifest = Get-Content -Raw -LiteralPath $frontendManifestPath | ConvertFrom-Json
            $scoreManifestPath = Resolve-ReferencedFile -VectorRoot $vectorDirFull `
                -RepoRoot $repoRoot -Reference ([string]$layerManifest.files.score_value.manifest)
            $scoreManifest = Get-Content -Raw -LiteralPath $scoreManifestPath | ConvertFrom-Json
            $layerPosition = [int]$frontendManifest.shape.selected_position
            if ($layerIndex -eq 0) {
                $runtimePosition = $layerPosition
            } elseif ($layerPosition -ne $runtimePosition) {
                throw "Runtime token position is inconsistent across layer manifests"
            }
            if (($layerPosition -lt 0) -or ($layerPosition -ge 256) -or
                ([int]$frontendManifest.shape.layer_id -ne $layerIndex) -or
                ([int]$scoreManifest.shape.layer_id -ne $layerIndex) -or
                ([int]$scoreManifest.shape.selected_position -ne $layerPosition) -or
                ([int]$scoreManifest.shape.cache_length -ne ($layerPosition + 1))) {
                throw "Layer $layerIndex runtime layer/position/cache-length manifest mismatch"
            }
        }
        $summary.Add("PASS runtime_manifest_contract :: position=$runtimePosition hidden_a=$(Format-SvAddress $runtimeHiddenABase) hidden_b=$(Format-SvAddress $runtimeHiddenBBase)")
    }

    $persistentManifest = $null
    $persistentManifestPath = $null
    if ($PersistentTwoToken) {
        $persistentFullModel = ($LayerCount -eq 28)
        $persistentManifestName = if ($persistentFullModel) {
            "persistent_multitoken_full_model_manifest.json"
        } else {
            "persistent_multitoken_layer_prefix_manifest.json"
        }
        $persistentLayerModeArgs = if ($persistentFullModel) {
            @("--all-layers")
        } else {
            @("--layer-count", "$LayerCount")
        }
        $persistentManifestPath = Join-Path $persistentVectorDirFull `
            $persistentManifestName
        if (!(Test-Path -LiteralPath $persistentManifestPath -PathType Leaf)) {
            New-Item -ItemType Directory -Force -Path $persistentVectorDirFull | Out-Null
            $feedbackToken = $PersistentFeedbackTokenId
            if ($feedbackToken -lt 0) {
                $probeDir = Join-Path $sessionDir "persistent_probe"
                $probeManifestPath = Join-Path $probeDir `
                    $persistentManifestName
                if (!(Test-Path -LiteralPath $probeManifestPath -PathType Leaf)) {
                    New-Item -ItemType Directory -Force -Path $probeDir | Out-Null
                    $probeArgs = @(
                        "run", "--no-capture-output", "-n", $CondaEnvironment,
                        "python", "-B", $persistentExporter,
                        "--output-dir", $probeDir
                    )
                    $probeArgs += $persistentLayerModeArgs
                    $probeArgs += @("--token-ids", "$TokenId")
                    Write-Host "[golden]  discover feedback token for token=$TokenId position=0 over $LayerCount layers"
                    $probeOutput = Invoke-LoggedCommand -Executable $conda -Arguments $probeArgs `
                        -LogPath (Join-Path $sessionDir "persistent_probe.log")
                    if (($probeOutput -join "`n") -notmatch
                        "Persistent multi-token fixed-point golden: PASS") {
                        throw "Persistent feedback probe did not print its PASS marker"
                    }
                }
                $probeManifest = Get-Content -Raw -LiteralPath $probeManifestPath | ConvertFrom-Json
                if (($probeManifest.self_check.status -ne "PASS") -or
                    ([int]$probeManifest.layer_count -ne $LayerCount) -or
                    ([int]$probeManifest.token_ids[0] -ne $TokenId) -or
                    ($probeManifest.final_tail.position_results.Count -ne 1)) {
                    throw "Persistent feedback probe manifest contract mismatch"
                }
                $feedbackToken =
                    [int]$probeManifest.final_tail.position_results[0].argmax_token
                $summary.Add("PASS feedback_probe :: token=$TokenId -> $feedbackToken")
            }

            $persistentArgs = @(
                "run", "--no-capture-output", "-n", $CondaEnvironment,
                "python", "-B", $persistentExporter,
                "--output-dir", $persistentVectorDirFull
            )
            $persistentArgs += $persistentLayerModeArgs
            $persistentArgs += @("--token-ids", "$TokenId", "$feedbackToken")
            Write-Host "[golden]  export feedback-closed positions 0/1 tokens $TokenId/$feedbackToken"
            $persistentOutput = Invoke-LoggedCommand -Executable $conda -Arguments $persistentArgs `
                -LogPath (Join-Path $sessionDir "persistent_export.log")
            if (($persistentOutput -join "`n") -notmatch
                "Persistent multi-token fixed-point golden: PASS") {
                throw "Persistent two-token exporter did not print its PASS marker"
            }
            $summary.Add("PASS persistent_vector_export")
        } else {
            $summary.Add("REUSE persistent_vector_export :: $persistentManifestPath")
        }

        $persistentManifest =
            Get-Content -Raw -LiteralPath $persistentManifestPath | ConvertFrom-Json
        if (($persistentManifest.self_check.status -ne "PASS") -or
            ($persistentManifest.self_check.full_vocab_argmax_checked -ne $true) -or
            ([int]$persistentManifest.layer_count -ne $LayerCount) -or
            ($persistentManifest.positions.Count -ne 2) -or
            ([int]$persistentManifest.positions[0] -ne 0) -or
            ([int]$persistentManifest.positions[1] -ne 1) -or
            ($persistentManifest.token_ids.Count -ne 2) -or
            ([int]$persistentManifest.token_ids[0] -ne $TokenId) -or
            ($persistentManifest.final_tail.position_results.Count -ne 2)) {
            throw "Persistent two-token manifest shape/self-check contract mismatch"
        }
        if ($persistentFullModel) {
            if (($persistentManifest.model_complete -ne $true) -or
                ($persistentManifest.valid_as_full_model_decode -ne $true) -or
                ([string]$persistentManifest.tail_semantics -ne "full_model_decode")) {
                throw "Full28 persistent manifest is not classified as a complete model decode"
            }
        } elseif (($null -ne $persistentManifest.model_complete) -and
            (($persistentManifest.model_complete -ne $false) -or
             ($persistentManifest.valid_as_full_model_decode -ne $false) -or
             ([string]$persistentManifest.tail_semantics -ne "truncated_prefix_diagnostic"))) {
            throw "Partial persistent manifest is not classified as a truncated-prefix diagnostic"
        }
        $feedbackToken = [int]$persistentManifest.final_tail.position_results[0].argmax_token
        if ([int]$persistentManifest.token_ids[1] -ne $feedbackToken) {
            throw "Persistent token pair is not feedback closed: step0 argmax=$feedbackToken step1 input=$($persistentManifest.token_ids[1])"
        }
        $baseTokenPath = Resolve-VectorFile $vectorDirFull `
            ([string]$manifest.embedding.files.token)
        $baseTokenText = (Get-Content -LiteralPath $baseTokenPath -TotalCount 1).Trim()
        $baseToken = [Convert]::ToInt32($baseTokenText, 16)
        if ($baseToken -ne $TokenId) {
            throw "RuntimeContext base bundle token $baseToken does not match persistent token $TokenId"
        }
        $summary.Add(
            "PASS persistent_manifest_contract :: tokens=$TokenId/$feedbackToken positions=0/1 layers=$LayerCount"
        )
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
    $includeLines.Add("                full_chain_manifest_runtime_context = $([int]$manifestRuntimeEnabled);")
    $includeLines.Add("                full_chain_runtime_position = $runtimePosition;")
    $includeLines.Add("                full_chain_embedding_output_base_addr = $(Format-SvAddress $embeddingOutputBase);")
    $includeLines.Add("                full_chain_runtime_hidden_a_base_addr = $(Format-SvAddress $runtimeHiddenABase);")
    $includeLines.Add("                full_chain_runtime_hidden_b_base_addr = $(Format-SvAddress $runtimeHiddenBBase);")
    $includeLines.Add("                full_chain_runtime_rope_cos_base_addr = $(Format-SvAddress $runtimeRopeCosBase);")
    $includeLines.Add("                full_chain_runtime_rope_sin_base_addr = $(Format-SvAddress $runtimeRopeSinBase);")
    $includeLines.Add("                full_chain_runtime_last_hidden_base_addr = $(Format-SvAddress $runtimeLastHiddenBase);")
    if ($RuntimeContext) {
        Add-ReadMemLine -Lines $includeLines -Path $runtimeRopeHex `
            -ArrayName "full_runtime_rope_mem" -Start 0 -WordCount (2 * 256 * 128)
    }

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
        $actualInputHidden = if ($RuntimeContext) {
            [UInt64]$layer.runtime_input_hidden_base
        } else {
            [UInt64]$layer.input_hidden_base
        }
        $actualOutputHidden = if ($RuntimeContext) {
            [UInt64]$layer.runtime_output_hidden_base
        } else {
            [UInt64]$layer.output_hidden_base
        }
        $includeLines.Add("                full_actual_input_hidden_base[$layerIndex] = $(Format-SvAddress $actualInputHidden);")
        $includeLines.Add("                full_actual_layer_output_base[$layerIndex] = $(Format-SvAddress $actualOutputHidden);")
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

    if ($PersistentTwoToken) {
        $persistentConfigLines = [System.Collections.Generic.List[string]]::new()
        $persistentConfigLines.Add(
            "                    // Generated from persistent_multitoken_layer_prefix_manifest.json."
        )
        for ($step = 0; $step -lt 2; $step++) {
            $inputToken = [int]$persistentManifest.token_ids[$step]
            $outputToken =
                [int]$persistentManifest.final_tail.position_results[$step].argmax_token
            $outputScore =
                [Int64]$persistentManifest.final_tail.position_results[$step].argmax_score_q26
            $persistentConfigLines.Add(
                "                    persistent_input_token[$step] = 32'd$inputToken;"
            )
            $persistentConfigLines.Add(
                "                    persistent_expected_output_token[$step] = 32'd$outputToken;"
            )
            $persistentConfigLines.Add(
                "                    persistent_expected_output_score[$step] = $(Format-SvSignedDecimal -Value $outputScore -Width 56);"
            )
        }
        $persistentConfigLines.Add(
            "                    if (full_chain_layer_count != $LayerCount) begin"
        )
        $persistentConfigLines.Add(
            '                        $display("FAIL: persistent/full-chain layer-count mismatch");'
        )
        $persistentConfigLines.Add("                        `$finish(1);")
        $persistentConfigLines.Add("                    end")
        $persistentConfigPath = Join-Path $xsimDir "persistent_two_token_config.svh"
        [System.IO.File]::WriteAllLines(
            $persistentConfigPath,
            [string[]]$persistentConfigLines,
            [System.Text.Encoding]::ASCII
        )

        $initialHiddenPath = Resolve-VectorFile $persistentVectorDirFull `
            "initial_hidden_q14_10_words32.hex"
        $finalNormPath = Resolve-VectorFile $persistentVectorDirFull `
            "final/final_norm_q12_12_words32.hex"
        $sliceDir = Join-Path $xsimDir "persistent_slices"
        New-Item -ItemType Directory -Force -Path $sliceDir | Out-Null
        $perLayerLoads = @(
            @("input_norm_q12_12_words32.hex", "full_expected_input_norm", 1024),
            @("q_rope_q12_12_words32.hex", "full_expected_q_rope", 2048),
            @("kv_write_addr.hex", "full_expected_cache_addr", 2048),
            @("kv_write_data_words32.hex", "full_expected_cache_data", 2048),
            @("kv_write_kind.hex", "full_expected_cache_kind", 2048),
            @("attention_output_q12_12_words32.hex", "full_expected_attn_out", 2048),
            @("o_proj_q12_12_words32.hex", "full_expected_o_proj", 1024),
            @("post_attention_hidden_q14_10_words32.hex", "full_expected_post_hidden", 1024),
            @("post_attention_norm_q12_12_words32.hex", "full_expected_post_norm", 1024),
            @("mlp_gate_q12_12_words32.hex", "full_expected_gate", 3072),
            @("mlp_up_q12_12_words32.hex", "full_expected_up", 3072),
            @("mlp_hidden_q12_12_words32.hex", "full_expected_silu_hidden", 3072),
            @("mlp_down_q12_12_words32.hex", "full_expected_down", 1024),
            @("layer_output_q14_10_words32.hex", "full_expected_layer", 1024)
        )

        for ($step = 0; $step -lt 2; $step++) {
            $stepLines = [System.Collections.Generic.List[string]]::new()
            $stepLines.Add(
                "                    // Position $step exact fixed-point writes and tail result."
            )
            $inputToken = [int]$persistentManifest.token_ids[$step]
            $outputToken =
                [int]$persistentManifest.final_tail.position_results[$step].argmax_token
            $outputScore =
                [Int64]$persistentManifest.final_tail.position_results[$step].argmax_score_q26
            $scoreBytes = [BitConverter]::GetBytes($outputScore)
            $scoreLow = [BitConverter]::ToUInt32($scoreBytes, 0)
            $scoreHigh = [BitConverter]::ToUInt32($scoreBytes, 4)
            $stepLines.Add("                    embedding_token_mem[0] = 32'd$inputToken;")
            $stepLines.Add(
                "                    tail_expected_words[0] = 32'd$outputToken;"
            )
            $stepLines.Add(
                "                    tail_expected_words[1] = $(Format-SvWord32 $scoreLow);"
            )
            $stepLines.Add(
                "                    tail_expected_words[2] = $(Format-SvWord32 $scoreHigh);"
            )

            $embeddingSlice = Join-Path $sliceDir "step${step}_embedding_expected.hex"
            $normSlice = Join-Path $sliceDir "step${step}_final_norm_expected.hex"
            Write-HexSlice -Source $initialHiddenPath -Destination $embeddingSlice `
                -Start ($step * 1024) -Count 1024
            Write-HexSlice -Source $finalNormPath -Destination $normSlice `
                -Start ($step * 1024) -Count 1024
            Add-ReadMemLine -Lines $stepLines -Path $embeddingSlice `
                -ArrayName "embedding_expected" -Start 0 -WordCount 1024
            Add-ReadMemLine -Lines $stepLines -Path $normSlice `
                -ArrayName "tail_final_norm_expected" -Start 0 -WordCount 1024

            for ($layerIndex = 0; $layerIndex -lt $LayerCount; $layerIndex++) {
                $positionRoot = "layer_{0:D2}/position_{1:D3}" -f $layerIndex,$step
                $qPath = Resolve-VectorFile $persistentVectorDirFull `
                    "$positionRoot/q_proj_q12_12_words32.hex"
                $kPath = Resolve-VectorFile $persistentVectorDirFull `
                    "$positionRoot/k_proj_q12_12_words32.hex"
                $vPath = Resolve-VectorFile $persistentVectorDirFull `
                    "$positionRoot/v_proj_q12_12_words32.hex"
                $qkvStart = $layerIndex * 4096
                Add-ReadMemLine -Lines $stepLines -Path $qPath `
                    -ArrayName "full_expected_qkv" -Start $qkvStart -WordCount 2048
                Add-ReadMemLine -Lines $stepLines -Path $kPath `
                    -ArrayName "full_expected_qkv" -Start ($qkvStart + 2048) -WordCount 1024
                Add-ReadMemLine -Lines $stepLines -Path $vPath `
                    -ArrayName "full_expected_qkv" -Start ($qkvStart + 3072) -WordCount 1024

                foreach ($load in $perLayerLoads) {
                    $relativeName = [string]$load[0]
                    $arrayName = [string]$load[1]
                    $wordCount = [int]$load[2]
                    $path = Resolve-VectorFile $persistentVectorDirFull `
                        "$positionRoot/$relativeName"
                    Add-ReadMemLine -Lines $stepLines -Path $path `
                        -ArrayName $arrayName -Start ($layerIndex * $wordCount) `
                        -WordCount $wordCount
                }
            }

            $stepIncludePath = Join-Path $xsimDir `
                "persistent_step${step}_vector_loads.svh"
            [System.IO.File]::WriteAllLines(
                $stepIncludePath,
                [string[]]$stepLines,
                [System.Text.Encoding]::ASCII
            )
        }
        $summary.Add(
            "PASS persistent Temp-only includes :: $persistentConfigPath"
        )
    }

    Push-Location $xsimDir
    try {
        $snapshotMode = if ($PersistentTwoToken) {
            "persistent"
        } elseif ($RuntimeContext) {
            "runtime"
        } else {
            "legacy"
        }
        $snapshot = "embedding_full_chain_${manifestLayerCount}_${snapshotMode}_axil_tail_xsim"
        $xvlogProjectPath = Join-Path $xsimDir "full_chain_sources.prj"
        $xvlogProjectLines = [System.Collections.Generic.List[string]]::new()
        foreach ($sourcePath in @($rtlFiles) + @($tbPath)) {
            $projectPath = $sourcePath.Replace("\", "/").Replace('"', '\"')
            $xvlogProjectLines.Add(('sv work "{0}"' -f $projectPath))
        }
        [System.IO.File]::WriteAllLines(
            $xvlogProjectPath,
            [string[]]$xvlogProjectLines,
            [System.Text.Encoding]::ASCII
        )
        $summary.Add(
            "PASS xvlog project :: $xvlogProjectPath " +
            "($($xvlogProjectLines.Count) ordered sources including testbench)"
        )

        $xvlogArgs = @("--sv", "--relax", "--prj", $xvlogProjectPath)
        foreach ($includeDir in $rtlIncludeDirs) {
            $xvlogArgs += @("-i", $includeDir)
        }
        $xvlogArgs += @("-i", $xsimDir)
        $compileDefines = @(
            "QMAP_ONE_TOKEN_TB_WITH_FINAL_TAIL",
            "QMAP_ONE_TOKEN_TB_USE_TOP",
            "QMAP_ONE_TOKEN_TB_USE_MMIO_CONTROL",
            "QMAP_ONE_TOKEN_TB_USE_AXIL_TOP",
            "QMAP_ONE_TOKEN_TB_FULL_CHAIN"
        )
        if ($PersistentTwoToken) {
            $compileDefines += "QMAP_ONE_TOKEN_TB_PERSISTENT_TWO_TOKEN"
        }
        foreach ($define in $compileDefines) {
            $xvlogArgs += @("-d", $define)
        }
        $xvlogArgs += @("--log", (Join-Path $xsimDir "xvlog.log"))
        Write-Host "[xvlog]  compile $manifestLayerCount-layer AXI-Lite full-chain snapshot"
        Invoke-LoggedCommand -Executable $xvlog -Arguments $xvlogArgs `
            -LogPath (Join-Path $xsimDir "xvlog_console.log") | Out-Null

        $xelabArgs = @(
            "--relax", "--O3", "--debug", "off", "--mt", "auto",
            "--snapshot", $snapshot,
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
        if ($RuntimeContext) {
            $runtimePlusargs = @(
                "-testplusarg", "true3_mmio_top_tail_only",
                "-testplusarg", "embedding_true3",
                "-testplusarg", "embedding_full_chain",
                "-testplusarg", "runtime_context",
                "-testplusarg", "fastmem",
                "-testplusarg", "progress",
                "-testplusarg", "notrace"
            )
            if ($PersistentTwoToken) {
                $runtimePlusargs += @(
                    "-testplusarg", "persistent_two_token"
                )
            }
            $xsimArgs = @($snapshot) + $runtimePlusargs + @(
                "--log", (Join-Path $xsimDir "xsim.log"),
                "--runall"
            )
        }
        $runLabel = if ($PersistentTwoToken) {
            "two persistent RuntimeContext tokens"
        } else {
            "embedding -> $manifestLayerCount layers -> full-vocabulary tail"
        }
        Write-Host "[xsim]   run $runLabel"
        $xsimOutput = Invoke-LoggedCommand -Executable $xsim -Arguments $xsimArgs `
            -LogPath (Join-Path $xsimDir "xsim_console.log")
        $combinedOutput = $xsimOutput -join "`n"
        if ($combinedOutput -match "(?m)^FAIL:") {
            throw "XSim printed a FAIL line. See $(Join-Path $xsimDir 'xsim.log')"
        }
        $passMarker = if ($PersistentTwoToken) {
            "PASS: AXI-Lite RuntimeContext persistent two-token decode ran through $manifestLayerCount complete layers without reset."
        } else {
            "PASS: AXI-Lite tied-Q4 embedding ran through $manifestLayerCount complete layers"
        }
        if ($combinedOutput -notmatch [regex]::Escape($passMarker)) {
            throw "XSim did not print the full-chain PASS marker"
        }
        $summary.Add("PASS xsim")
    } finally {
        Pop-Location
    }

    if ($PersistentTwoToken) {
        $eventTrace = Join-Path $xsimDir "persistent_two_token_events.csv"
        $auditPath = Join-Path $xsimDir "persistent_timing_audit.json"
        $auditArgs = @(
            "run", "--no-capture-output", "-n", $CondaEnvironment,
            "python", "-B", $persistentTimingChecker, $eventTrace,
            "--full-chain-manifest", $manifestPath,
            "--persistent-manifest", $persistentManifestPath,
            "--xsim-log", (Join-Path $xsimDir "xsim.log"),
            "--output", $auditPath
        )
        Write-Host "[audit]  verify two starts, feedback, exact writes, retained K/V, and no reset"
        $auditOutput = Invoke-LoggedCommand -Executable $conda -Arguments $auditArgs `
            -LogPath (Join-Path $xsimDir "persistent_timing_audit_console.log")
        if (($auditOutput -join "`n") -notmatch
            "PASS: persistent two-token full-chain timing and retention are exact") {
            throw "Persistent timing auditor did not print its PASS marker"
        }
        $summary.Add("PASS persistent_timing_audit :: $auditPath")
    } else {
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
    }

    $summaryPath = Join-Path $sessionDir "summary.txt"
    [System.IO.File]::WriteAllLines($summaryPath, [string[]]$summary)
    $summary | ForEach-Object { Write-Host $_ }
    Write-Host "All vectors, links, generated includes, compiler products, logs, traces, and audits are under $sessionDir"
} finally {
    Pop-Location
    $env:PYTHONPYCACHEPREFIX = $savedPycachePrefix
}
